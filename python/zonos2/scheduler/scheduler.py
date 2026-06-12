"""TTS Scheduler for multi-codebook audio generation."""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch
import torch.nn.functional as F
from zonos2.core import TTSBatch, TTSReq
from zonos2.env import ENV
from zonos2.message import (
    BaseTTSBackendMsg,
    BatchTTSBackendMsg,
    BatchTTSTokenizerMsg,
    ExitMsg,
    TTSDetokenizeMsg,
    TTSUserMsg,
)
from zonos2.utils import init_logger

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import TTSDecodeManager
from .io import SchedulerIOMixin
from .table import TTSTableManager

if TYPE_CHECKING:
    from zonos2.engine.sample import TTSBatchSamplingArgs

logger = init_logger(__name__)


class TTSForwardOutput(NamedTuple):
    """Output from TTS forward pass."""

    next_tokens_gpu: torch.Tensor  # (batch, frame_width)
    next_tokens_cpu: torch.Tensor  # (batch, frame_width)
    copy_done: torch.cuda.Event


class TTSForwardInput(NamedTuple):
    """Input for TTS forward pass."""

    batch: TTSBatch
    sample_args: TTSBatchSamplingArgs
    load_indices: torch.Tensor
    write_indices: torch.Tensor


TTSForwardData: TypeAlias = "Tuple[TTSForwardInput, TTSForwardOutput]"


def _make_3d_indices(table_3d: torch.Tensor, ranges: List[Tuple[int, int, int]]) -> torch.Tensor:
    """Return 1D indices for a 3D token pool.

    Args:
        table_3d: 3D tensor (max_reqs, max_seq_len, frame_width)
        ranges: List of (req_idx, begin, end) tuples

    Returns:
        1D tensor of indices into the flattened token pool.
        These indices are designed to be divided by frame_width to get
        row indices into flat_pool.view(-1, frame_width).
    """
    assert table_3d.dim() == 3 and table_3d.is_contiguous()
    req_stride, seq_stride, _ = table_3d.stride()
    # For shape (max_reqs, max_seq_len, frame_width):
    #   req_stride = max_seq_len * frame_width
    #   seq_stride = frame_width
    # We compute: entry * req_stride + pos * seq_stride
    # When divided by frame_width, this gives: entry * max_seq_len + pos
    needed_size = sum(end - begin for _, begin, end in ranges)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for entry, begin, end in ranges:
        length = end - begin
        base = entry * req_stride
        for i in range(length):
            indices_host[offset + i] = base + (begin + i) * seq_stride
        offset += length
    return indices_host.to(table_3d.device, non_blocking=True)


def _make_speaker_slot(
    input_ids: torch.Tensor,
    *,
    n_codebooks: int,
    audio_pad_id: int,
    text_vocab: int,
) -> torch.Tensor:
    """Create a reserved frame for speaker embedding injection at position 0."""
    assert input_ids.dim() == 2
    slot = torch.zeros(
        (1, input_ids.shape[-1]),
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    audio_cols = min(n_codebooks, input_ids.shape[-1])
    slot[:, :audio_cols] = int(audio_pad_id)
    if input_ids.shape[-1] > n_codebooks:
        text_pad = int(text_vocab) if int(text_vocab) > 0 else int(audio_pad_id)
        slot[:, n_codebooks] = text_pad
    return slot


def _has_reserved_speaker_slot(
    input_ids: torch.Tensor,
    *,
    n_codebooks: int,
    audio_pad_id: int,
    text_vocab: int,
) -> bool:
    """Check whether the first frame already matches the reserved speaker slot."""
    if input_ids.shape[0] == 0 or input_ids.shape[-1] <= n_codebooks:
        return False
    first = input_ids[0]
    audio_cols = min(n_codebooks, input_ids.shape[-1])
    audio_ok = bool(torch.all(first[:audio_cols] == int(audio_pad_id)))
    text_expected = int(text_vocab) if int(text_vocab) > 0 else int(audio_pad_id)
    text_ok = int(first[n_codebooks].item()) == text_expected
    return audio_ok and text_ok


def _make_marker_slot(
    input_ids: torch.Tensor,
    *,
    n_codebooks: int,
    audio_pad_id: int,
    text_token: int,
) -> torch.Tensor:
    """Create a conditioning marker frame: audio padding plus one text token."""
    assert input_ids.dim() == 2
    slot = torch.zeros(
        (1, input_ids.shape[-1]),
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    audio_cols = min(n_codebooks, input_ids.shape[-1])
    slot[:, :audio_cols] = int(audio_pad_id)
    if input_ids.shape[-1] > n_codebooks:
        slot[:, n_codebooks] = int(text_token)
    return slot


def _frame_text_token(
    input_ids: torch.Tensor,
    frame_idx: int,
    *,
    n_codebooks: int,
    audio_pad_id: int,
) -> int | None:
    """Return the text token of an audio-padded frame, or None if not a marker."""
    if input_ids.shape[0] <= frame_idx or input_ids.shape[-1] <= n_codebooks:
        return None
    frame = input_ids[frame_idx]
    audio_cols = min(n_codebooks, input_ids.shape[-1])
    if not bool(torch.all(frame[:audio_cols] == int(audio_pad_id))):
        return None
    return int(frame[n_codebooks].item())


class TTSScheduler(SchedulerIOMixin):
    """Scheduler for TTS multi-codebook generation."""

    def __init__(self, config: SchedulerConfig):
        from zonos2.engine import Engine
        from zonos2.engine.sample import TTSSampler

        self.engine = Engine(config)
        super().__init__(config, self.engine.tp_cpu_group)

        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)

        # TTS-specific configuration - read from model config
        model_config = self.engine.model_config
        self.n_codebooks = model_config.n_codebooks
        self.codebook_size = model_config.codebook_size
        self.eoa_id = model_config.eoa_id
        self.audio_pad_id = model_config.audio_pad_id
        self.text_vocab = model_config.text_vocab if model_config.text_vocab is not None else 0
        self.speaker_enabled = bool(getattr(model_config, "speaker_enabled", False))
        self.speaker_embedding_dim = int(getattr(model_config, "speaker_embedding_dim", 0))
        self.speaker_background_token_enabled = bool(
            getattr(model_config, "speaker_background_token_enabled", False)
        )
        self.accurate_mode_token_enabled = bool(
            getattr(model_config, "accurate_mode_token_enabled", False)
        )
        self.speaker_background_num_buckets = 2 if self.speaker_background_token_enabled else 0
        self.accurate_mode_num_buckets = (
            1
            if self.accurate_mode_token_enabled and self.speaker_background_num_buckets > 0
            else 0
        )
        self.speaking_rate_num_buckets = int(getattr(model_config, "speaking_rate_num_buckets", 0))
        quality_buckets = getattr(model_config, "quality_buckets", None) or {}
        quality_features = tuple(getattr(model_config, "quality_features", ()) or ())
        if not quality_features:
            quality_features = tuple(quality_buckets.keys())
        self.quality_features = quality_features
        self.quality_bucket_counts = [
            len(quality_buckets.get(feature, ())) for feature in quality_features
        ]
        # Frame width includes n_codebooks audio codes and one text token.
        self.frame_width = self.n_codebooks + 1

        # TTS sampler
        self.tts_sampler = TTSSampler(
            device=self.device,
            n_codebooks=self.n_codebooks,
            codebook_size=self.codebook_size,
            text_vocab=self.text_vocab,
        )

        # Initialize TTS-specific managers
        self.table_manager = TTSTableManager(
            config.max_running_req, self.engine.page_table, self.frame_width
        )
        self.cache_manager = CacheManager(self.device, self.engine.num_pages, config.cache_type)
        self.decode_manager = TTSDecodeManager()

        # Waiting queue for prefill
        self.waiting_reqs: List[TTSUserMsg] = []

        self.tp_info = config.tp_info
        self.finished_reqs: Set[TTSReq] = set()
        self.page_table = self.engine.page_table
        self.token_pool = self.table_manager.token_pool
        self.prefill_budget = config.max_extend_tokens

        # Metrics tracking for periodic throughput logging
        self._stats_start = time.perf_counter()
        self._stats_batches = 0
        self._stats_tokens = 0

    def _process_last_data(
        self,
        last_data: TTSForwardData | None,
        ongoing_data: TTSForwardData | None,
    ) -> None:
        if last_data is None:
            return

        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        copy_done.synchronize()
        reply = BatchTTSTokenizerMsg(data=[])
        logger.debug(
            "Processing batch: size=%d, padded_size=%d, next_tokens shape=%s",
            batch.size,
            batch.padded_size,
            next_tokens_cpu.shape,
        )

        max_seq_len = self.engine.max_seq_len
        for i, req in enumerate(batch.reqs):
            if req in self.finished_reqs:
                continue

            # Get audio codes for this frame
            next_token = next_tokens_cpu[i]  # (frame_width,)
            audio_codes = next_token[: self.n_codebooks].tolist()

            # Append to host storage
            req.append_host(next_token)

            # Log progress
            generated = req.total_generated
            if generated <= 5:
                logger.debug("TTS %d frame %d: audio_codes=%s", req.uid, generated, audio_codes)

            # Check for EOS
            finished = req.remain_len <= 0
            if req.check_eos(audio_codes):
                finished = True
                logger.debug(
                    "TTS %d EOS detected at frame %d, eos_frame=%d",
                    req.uid,
                    generated,
                    req.eos_frame,
                )

            if req.device_len >= max_seq_len - 1:
                finished = True
                logger.warning_rank0(f"TTS request {req.uid} reached {max_seq_len = }, dropped.")

            # Send detokenize message with audio codes
            reply.data.append(
                TTSDetokenizeMsg(
                    uid=req.uid,
                    audio_codes=audio_codes,
                    finished=finished,
                    eos_frame=req.eos_frame if req.eos_frame >= 0 else None,
                )
            )

            if finished:
                self.finished_reqs.add(req)
                self.decode_manager.remove_req(req)
                logger.debug("TTS %d finished: %d frames generated", req.uid, generated)

        # Free resources for finished but not ongoing reqs
        ongoing_reqs = ongoing_data[0].batch.reqs if ongoing_data else []
        for req in self.finished_reqs.difference(ongoing_reqs):
            # Free the allocated cache slots (cached_len represents actually allocated positions)
            slots = self.page_table[req.table_idx, : req.cached_len]
            self.cache_manager.free_slots(slots)
            self.cache_manager.free_handle(req.cache_handle)
            self.table_manager.free(req.table_idx)

        self.finished_reqs.intersection_update(ongoing_reqs)
        self.send_result(reply)

    def _process_one_msg(self, msg: BaseTTSBackendMsg) -> None:
        if isinstance(msg, BatchTTSBackendMsg):
            for m in msg.data:
                self._process_one_msg(m)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, TTSUserMsg):
            logger.debug_rank0("Received TTS user msg: %s", msg.uid)
            input_len = len(msg.input_ids)
            max_seq_len = self.engine.max_seq_len
            if input_len >= max_seq_len:
                return logger.warning_rank0(
                    f"TTS input len {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
            self.waiting_reqs.append(msg)
        else:
            logger.error(f"Unknown TTS message type: {type(msg)}")
            raise NotImplementedError

    def _speaker_background_token(self, clean: bool) -> int:
        # Imported lazily: zonos2.tts pulls in TTSLLM, which imports this module.
        from zonos2.tts.prompt import speaker_background_token_id

        return speaker_background_token_id(
            self.text_vocab,
            self.speaking_rate_num_buckets,
            self.quality_bucket_counts,
            clean,
            self.speaker_background_num_buckets,
            self.accurate_mode_num_buckets,
        )

    def _accurate_mode_token(self) -> int:
        from zonos2.tts.prompt import accurate_mode_token_id

        return accurate_mode_token_id(
            self.text_vocab,
            self.speaking_rate_num_buckets,
            self.quality_bucket_counts,
            self.speaker_background_num_buckets,
            self.accurate_mode_num_buckets,
        )

    def _with_speaker_frames(
        self,
        input_ids: torch.Tensor,
        msg: TTSUserMsg,
        speaker_token_position: int,
    ) -> torch.Tensor:
        """Ensure the canonical speaker slot and marker frames lead the prompt.

        Train-time prompt layout: speaker slot at frame 0, then the clean/noisy
        speaker-background marker, then (optionally) the accurate-mode marker.
        """
        needs_background = self.speaker_background_num_buckets > 0
        needs_accurate = self.accurate_mode_num_buckets > 0 and bool(msg.accurate_mode)

        def _marker(text_token: int) -> torch.Tensor:
            return _make_marker_slot(
                input_ids,
                n_codebooks=self.n_codebooks,
                audio_pad_id=self.audio_pad_id,
                text_token=text_token,
            )

        has_slot = speaker_token_position == 0 and _has_reserved_speaker_slot(
            input_ids,
            n_codebooks=self.n_codebooks,
            audio_pad_id=self.audio_pad_id,
            text_vocab=self.text_vocab,
        )
        if not has_slot:
            prefix = [
                _make_speaker_slot(
                    input_ids,
                    n_codebooks=self.n_codebooks,
                    audio_pad_id=self.audio_pad_id,
                    text_vocab=self.text_vocab,
                )
            ]
            if needs_background:
                prefix.append(_marker(self._speaker_background_token(msg.clean_speaker_background)))
                if needs_accurate:
                    prefix.append(_marker(self._accurate_mode_token()))
            return torch.cat(prefix + [input_ids], dim=0)

        if not needs_background:
            return input_ids

        background_ids = {
            self._speaker_background_token(True),
            self._speaker_background_token(False),
        }
        frame1 = _frame_text_token(
            input_ids, 1, n_codebooks=self.n_codebooks, audio_pad_id=self.audio_pad_id
        )
        if frame1 not in background_ids:
            inserted = [_marker(self._speaker_background_token(msg.clean_speaker_background))]
            if needs_accurate:
                inserted.append(_marker(self._accurate_mode_token()))
            return torch.cat([input_ids[:1]] + inserted + [input_ids[1:]], dim=0)

        if needs_accurate:
            frame2 = _frame_text_token(
                input_ids, 2, n_codebooks=self.n_codebooks, audio_pad_id=self.audio_pad_id
            )
            if frame2 != self._accurate_mode_token():
                return torch.cat(
                    [input_ids[:2], _marker(self._accurate_mode_token()), input_ids[2:]], dim=0
                )

        return input_ids

    def _schedule_prefill(self) -> TTSBatch | None:
        """Schedule a prefill batch from waiting requests."""
        if not self.waiting_reqs or self.table_manager.available_size == 0:
            return None

        reqs = []
        total_tokens = 0

        while self.waiting_reqs and self.table_manager.available_size > 0:
            if total_tokens >= self.prefill_budget:
                break

            msg = self.waiting_reqs[0]
            input_ids = msg.input_ids
            speaker_token_position = msg.speaker_token_position
            if msg.speaker_embedding is not None:
                input_ids = self._with_speaker_frames(input_ids, msg, speaker_token_position)
                speaker_token_position = 0

            input_len = len(input_ids)
            max_seq_len = self.engine.max_seq_len
            if input_len >= max_seq_len:
                logger.warning_rank0(
                    f"TTS input len {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
                self.waiting_reqs.pop(0)
                continue

            max_output_len = max_seq_len - input_len
            sampling_params = msg.sampling_params
            if msg.sampling_params.max_tokens > max_output_len:
                sampling_params = replace(msg.sampling_params, max_tokens=max_output_len)
                logger.debug_rank0(
                    "Adjust TTS max_tokens to %d for request %d.",
                    max_output_len,
                    msg.uid,
                )

            if total_tokens + input_len > self.prefill_budget and reqs:
                break  # Don't exceed budget unless this is the first request

            self.waiting_reqs.pop(0)
            total_tokens += input_len

            # Create TTSReq
            table_idx = self.table_manager.allocate()
            cache_handle = self.cache_manager.allocate_new_handle()

            # Copy input_ids to token pool
            input_ids_i32 = input_ids.to(torch.int32)
            self.token_pool[table_idx, :input_len].copy_(
                input_ids_i32.pin_memory().to(self.device), non_blocking=True
            )
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "TTS %d prefill: input_len=%d, input_ids shape=%s",
                    msg.uid,
                    input_len,
                    input_ids_i32.shape,
                )
                logger.debug("TTS %d   first frame: %s", msg.uid, input_ids_i32[0].tolist())
                logger.debug("TTS %d   last frame: %s", msg.uid, input_ids_i32[-1].tolist())
                if input_len > 20:
                    logger.debug("TTS %d   frame at -20: %s", msg.uid, input_ids_i32[-20].tolist())
                    logger.debug("TTS %d   frame at -17: %s", msg.uid, input_ids_i32[-17].tolist())

            rng = None
            if msg.sampling_params.seed is not None:
                rng = torch.Generator(device=self.device)
                rng.manual_seed(msg.sampling_params.seed)

            req = TTSReq(
                input_ids=input_ids_i32,
                table_idx=table_idx,
                cached_len=0,
                output_len=sampling_params.max_tokens,
                uid=msg.uid,
                sampling_params=sampling_params,
                cache_handle=cache_handle,
                n_codebooks=self.n_codebooks,
                eoa_id=self.eoa_id,
                rng=rng,
                speaker_embedding=msg.speaker_embedding,
                speaker_token_position=speaker_token_position,
            )
            reqs.append(req)

        if not reqs:
            return None

        return TTSBatch(reqs=reqs, phase="prefill")

    def _prepare_batch(self, batch: TTSBatch) -> TTSForwardInput:
        """Prepare a batch for forward pass."""
        needed_size = sum(r.extend_len for r in batch.reqs)
        batch.out_loc = self.cache_manager.allocate(needed_size)

        # Pad batch if needed for CUDA graph
        if padding_size := self.engine.graph_runner.pad_batch(batch):
            batch.out_loc = F.pad(batch.out_loc, (0, padding_size), value=self.engine.dummy_page)

        # For TTS we load full frames
        load_ranges = [(r.table_idx, r.cached_len, r.device_len) for r in batch.padded_reqs]
        write_ranges = [(r.table_idx, r.device_len, r.device_len + 1) for r in batch.reqs]

        load_indices = _make_3d_indices(self.token_pool, load_ranges)
        write_indices = _make_3d_indices(self.token_pool, write_ranges)

        # Write out_loc to page table
        self.page_table.view(-1)[load_indices // self.frame_width] = batch.out_loc
        self.engine.attn_backend.prepare_metadata(batch)

        return TTSForwardInput(
            batch=batch,
            sample_args=self.tts_sampler.prepare(batch, token_pool=self.token_pool),
            load_indices=load_indices,
            write_indices=write_indices,
        )

    def _schedule_next_batch(self) -> TTSForwardInput | None:
        """Schedule next batch (prefill first, then decode)."""
        batch = self._schedule_prefill() or self.decode_manager.schedule_next_batch()
        return self._prepare_batch(batch) if batch else None

    def _log_batch_stats(self, batch: TTSBatch) -> None:
        self._stats_batches += 1
        self._stats_tokens += batch.size

        if batch.is_prefill:
            logger.info_rank0(
                "TTS Prefill: reqs=%d, extend_frames=%d, waiting=%d, decode_running=%d",
                batch.size,
                sum(r.extend_len for r in batch.reqs),
                len(self.waiting_reqs),
                len(self.decode_manager.running_reqs),
            )

        now = time.perf_counter()
        elapsed = now - self._stats_start
        if elapsed >= 5.0:
            logger.info_rank0(
                "TTS Throughput (%.1fs window): batches=%d, decode_steps=%d, %.1f frames/s, "
                "decode_running=%d, waiting=%d",
                elapsed,
                self._stats_batches,
                self._stats_tokens,
                self._stats_tokens / elapsed,
                len(self.decode_manager.running_reqs),
                len(self.waiting_reqs),
            )
            self._stats_start = now
            self._stats_batches = 0
            self._stats_tokens = 0

    def _load_token_ids(self, input: TTSForwardInput) -> None:
        """Load token IDs from pool to batch."""
        # Load full frames for TTS
        flat_pool = self.token_pool.view(-1, self.frame_width)
        frame_indices = input.load_indices // self.frame_width
        input.batch.input_ids = flat_pool[frame_indices]

        # Debug: verify loaded tokens
        if logger.isEnabledFor(logging.DEBUG) and input.batch.is_decode and input.batch.size > 0:
            req = input.batch.reqs[0]
            if req.total_generated < 10:
                loaded_tokens = input.batch.input_ids[0].tolist()
                logger.debug(
                    "Load frame %d: loaded from frame_index %d, tokens = %s",
                    req.total_generated,
                    frame_indices[0].item(),
                    loaded_tokens,
                )

    def _write_token_ids(self, input: TTSForwardInput, output: TTSForwardOutput) -> None:
        """Write generated tokens back to pool."""
        flat_pool = self.token_pool.view(-1, self.frame_width)
        frame_indices = input.write_indices // self.frame_width
        flat_pool[frame_indices] = output.next_tokens_gpu

        # Debug: verify what was written
        if logger.isEnabledFor(logging.DEBUG) and input.batch.size > 0:
            req = input.batch.reqs[0]
            if req.total_generated < 10:
                written_tokens = flat_pool[frame_indices[0]].tolist()
                logger.debug(
                    "Write frame %d: wrote tokens %s at frame_index %d",
                    req.total_generated,
                    written_tokens,
                    frame_indices[0].item(),
                )

    def _prepare_speaker_conditioning(self, batch: TTSBatch) -> None:
        """Prepare per-batch speaker embeddings and injection positions."""
        if not self.speaker_enabled:
            batch.speaker_emb_values = None
            batch.speaker_token_positions = None
            return

        emb_values: list[torch.Tensor] = []
        token_positions: list[int] = []
        offset = 0

        for req in batch.reqs:
            if req.speaker_embedding is None:
                offset += req.extend_len
                continue

            rel_pos = req.speaker_token_position - req.cached_len
            if 0 <= rel_pos < req.extend_len:
                emb = req.speaker_embedding
                if emb.numel() != self.speaker_embedding_dim:
                    logger.warning_rank0(
                        "Skipping speaker embedding for uid=%d: expected dim=%d, got=%d",
                        req.uid,
                        self.speaker_embedding_dim,
                        emb.numel(),
                    )
                else:
                    emb_values.append(emb.to(torch.float32))
                    token_positions.append(offset + rel_pos)
            offset += req.extend_len

        if emb_values:
            batch.speaker_emb_values = torch.stack(emb_values, dim=0).to(
                self.device, non_blocking=True
            )
            batch.speaker_token_positions = torch.tensor(
                token_positions, dtype=torch.long, device=self.device
            )
        else:
            batch.speaker_emb_values = None
            batch.speaker_token_positions = None

    def _forward(self, forward_input: TTSForwardInput) -> TTSForwardOutput:
        """Run forward pass on batch."""
        self._load_token_ids(forward_input)
        batch, sample_args = forward_input.batch, forward_input.sample_args
        self._prepare_speaker_conditioning(batch)

        # Debug: show input shape and first request's input
        if logger.isEnabledFor(logging.DEBUG):
            if batch.is_prefill and batch.size > 0:
                logger.debug("Forward prefill input_ids shape: %s", batch.input_ids.shape)
            elif batch.is_decode and batch.size > 0:
                req = batch.reqs[0]
                if req.total_generated < 10:
                    input_tokens = batch.input_ids[0].tolist()
                    eoa_id = self.eoa_id
                    eoa_positions = [
                        i for i, t in enumerate(input_tokens[: self.n_codebooks]) if t == eoa_id
                    ]
                    logger.debug(
                        "Forward decode step %d: input tokens = %s",
                        req.total_generated,
                        input_tokens,
                    )
                    if eoa_positions:
                        logger.debug(
                            "Forward WARNING: EOA token in INPUT at codebook positions %s!",
                            eoa_positions,
                        )
                    positions = batch.attn_metadata.positions
                    logger.debug(
                        "Forward positions: %s",
                        (
                            positions[: batch.size].tolist()
                            if hasattr(positions, "tolist")
                            else positions
                        ),
                    )

        # Forward through engine - get multi-codebook logits
        logits = self.engine.forward_batch_tts(batch)

        # Debug: check logits stats for first 5 frames
        if logger.isEnabledFor(logging.DEBUG) and batch.size > 0:
            req = batch.reqs[0]
            if req.total_generated < 5:
                logger.debug(
                    "Forward frame %d: logits shape=%s, min=%.2f, max=%.2f, mean=%.2f",
                    req.total_generated,
                    logits.shape,
                    logits.min().item(),
                    logits.max().item(),
                    logits.mean().item(),
                )

        # Sample tokens
        next_tokens = self.tts_sampler.sample_to_tensor(logits, sample_args)

        # Copy to CPU
        next_tokens_cpu = torch.empty_like(next_tokens, device="cpu", pin_memory=True)
        copy_done = torch.cuda.Event()
        next_tokens_cpu.copy_(next_tokens, non_blocking=True)
        copy_done.record()

        self._write_token_ids(
            forward_input,
            TTSForwardOutput(next_tokens, next_tokens_cpu, copy_done),
        )

        # Mark all requests as available for decode
        for req in batch.reqs:
            req.complete_one()
        self.decode_manager.add_reqs(forward_input.batch.reqs)

        return TTSForwardOutput(next_tokens, next_tokens_cpu, copy_done)

    def run_when_idle(self) -> None:
        """Called when scheduler is idle."""
        logger.info_rank0("TTS Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()

    def overlap_loop(self, last_data: TTSForwardData | None) -> TTSForwardData | None:
        """Main loop with overlapping scheduling and execution."""
        blocking = not (last_data or self.waiting_reqs or self.decode_manager.runnable)
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            self._log_batch_stats(forward_input.batch)
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(last_data, ongoing_data)
        return ongoing_data

    def normal_loop(self) -> None:
        """Non-overlapping loop."""
        blocking = not (self.waiting_reqs or self.decode_manager.runnable)
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            self._log_batch_stats(forward_input.batch)
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data, None)

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        """Run scheduler loop forever."""
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                data = self.overlap_loop(data)

    def shutdown(self) -> None:
        """Shutdown the scheduler."""
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        self.engine.shutdown()
