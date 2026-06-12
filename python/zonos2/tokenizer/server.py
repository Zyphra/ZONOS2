from __future__ import annotations

import multiprocessing as mp
import threading
from typing import List

import torch
from zonos2.message import (
    BaseBackendMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchTokenizerMsg,
    BatchTTSBackendMsg,
    BatchTTSFrontendMsg,
    BatchTTSTokenizerMsg,
    TTSAudioReply,
    TTSDetokenizeMsg,
    TTSTokenizeMsg,
    TTSUserMsg,
)
from zonos2.tts.prompt import TTSPromptBuilder, TTSPromptConfig
from zonos2.utils import ZmqPullQueue, ZmqPushQueue, init_logger


def _unwrap_msg(
    msg: BaseTokenizerMsg | BatchTokenizerMsg | BatchTTSTokenizerMsg,
) -> List[BaseTokenizerMsg]:
    if isinstance(msg, BatchTokenizerMsg):
        return msg.data
    if isinstance(msg, BatchTTSTokenizerMsg):
        return msg.data
    return [msg]


@torch.inference_mode()
def tokenize_worker(
    *,
    addr: str,
    create: bool,
    backend_addr: str,
    frontend_addr: str,
    local_bs: int,
    tokenizer_id: int = -1,
    ack_queue: mp.Queue[str] | None = None,
    n_codebooks: int = 9,
    audio_pad_id: int = 1025,
    text_vocab: int | None = None,
    speaking_rate_num_buckets: int = 0,
    quality_bucket_counts: List[int] | None = None,
    speaker_background_num_buckets: int = 0,
    accurate_mode_num_buckets: int = 0,
) -> None:
    send_backend = ZmqPushQueue(backend_addr, create=False, encoder=BaseBackendMsg.encoder)
    send_frontend = ZmqPushQueue(frontend_addr, create=False, encoder=BaseFrontendMsg.encoder)
    recv_listener = ZmqPullQueue(addr, create=create, decoder=BatchTokenizerMsg.decoder)
    assert local_bs > 0
    logger = init_logger(__name__, f"tokenizer_{tokenizer_id}")

    from .textnorm import TTSTextNormalizer, normalization_enabled
    from .vocoder import TTSVocoderManager

    if text_vocab is None:
        raise ValueError("TTS mode requires text_vocab from the model config.")
    textnorm_enabled = normalization_enabled()
    text_normalizer = TTSTextNormalizer() if textnorm_enabled else None
    if text_normalizer is not None:
        # Compile/load the English grammars up front so the first request does
        # not pay the one-time WFST construction cost.
        threading.Thread(
            target=text_normalizer.warmup, args=(["en"],), daemon=True
        ).start()
    tts_prompt_builder = TTSPromptBuilder(
        TTSPromptConfig(
            n_codebooks=n_codebooks,
            audio_pad_id=audio_pad_id,
            text_vocab=text_vocab,
            speaking_rate_num_buckets=speaking_rate_num_buckets,
            quality_bucket_counts=tuple(quality_bucket_counts or ()),
            speaker_background_num_buckets=speaker_background_num_buckets,
            accurate_mode_num_buckets=accurate_mode_num_buckets,
            prepend_silence=True,
        )
    )
    tts_vocoder_manager = TTSVocoderManager(
        n_codebooks=n_codebooks,
        audio_pad_id=audio_pad_id,
    )

    if ack_queue is not None:
        ack_queue.put(f"Tokenize server {tokenizer_id} is ready")

    try:
        while True:
            pending_msg = _unwrap_msg(recv_listener.get())
            while len(pending_msg) < local_bs and not recv_listener.empty():
                pending_msg.extend(_unwrap_msg(recv_listener.get()))

            logger.debug(f"Received {len(pending_msg)} messages")

            tts_tokenize_msg = [m for m in pending_msg if isinstance(m, TTSTokenizeMsg)]
            tts_detokenize_msg = [
                m for m in pending_msg if isinstance(m, TTSDetokenizeMsg)
            ]

            # Process TTS tokenize messages
            if len(tts_tokenize_msg) > 0:
                tensors = []
                for msg in tts_tokenize_msg:
                    text = msg.text
                    if text_normalizer is not None and msg.text_normalization:
                        text = text_normalizer.normalize(text, msg.language)
                    tensors.append(
                        tts_prompt_builder.build(
                            text,
                            speaking_rate_bucket=msg.speaking_rate_bucket,
                            quality_buckets=msg.quality_buckets,
                        )
                    )
                user_msgs = []
                for msg, t in zip(tts_tokenize_msg, tensors, strict=True):
                    speaker_token_position = msg.speaker_token_position
                    if msg.speaker_embedding is not None:
                        # Canonical speaker slot is token position 0, matching training.
                        speaker_slot = tts_prompt_builder.speaker_slot(
                            dtype=t.dtype,
                            device=t.device,
                        )
                        t = torch.cat([speaker_slot, t], dim=0)
                        speaker_token_position = 0
                    logger.debug(
                        "Tokenize uid=%d speaking_rate_bucket=%s quality_buckets=%s "
                        "text='%s...' frames=%d",
                        msg.uid,
                        msg.speaking_rate_bucket,
                        msg.quality_buckets,
                        msg.text[:50],
                        len(t),
                    )
                    user_msgs.append(
                        TTSUserMsg(
                            uid=msg.uid,
                            input_ids=t,
                            sampling_params=msg.sampling_params,
                            speaker_embedding=msg.speaker_embedding,
                            speaker_token_position=speaker_token_position,
                            clean_speaker_background=msg.clean_speaker_background,
                            accurate_mode=msg.accurate_mode,
                        )
                    )
                batch_output = BatchTTSBackendMsg(
                    data=user_msgs
                )
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]
                send_backend.put(batch_output)

            # Process TTS detokenize (vocoder) messages
            if len(tts_detokenize_msg) > 0:
                audio_chunks = tts_vocoder_manager.decode_frames(tts_detokenize_msg)
                batch_output = BatchTTSFrontendMsg(
                    data=[
                        TTSAudioReply(
                            uid=msg.uid,
                            audio_data=audio,
                            finished=msg.finished,
                        )
                        for msg, audio in zip(
                            tts_detokenize_msg, audio_chunks, strict=True
                        )
                    ]
                )
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]
                send_frontend.put(batch_output)

    except KeyboardInterrupt:
        pass
