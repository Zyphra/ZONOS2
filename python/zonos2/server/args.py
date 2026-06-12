from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch
from zonos2.distributed import DistributedInfo
from zonos2.scheduler import SchedulerConfig
from zonos2.utils import cached_load_checkpoint_config, init_logger


@dataclass(frozen=True)
class ServerArgs(SchedulerConfig):
    server_host: str = "127.0.0.1"
    server_port: int = 1919
    num_tokenizer: int = 0
    silent_output: bool = False

    # TTS-specific settings
    tts_n_codebooks: int = 9
    tts_audio_pad_id: int = 1025
    tts_text_vocab: int | None = None
    tts_sample_rate: int = 44100
    tts_speaker_background_token_enabled: bool = False
    tts_accurate_mode_token_enabled: bool = False
    tts_speaking_rate_num_buckets: int = 0
    tts_speaking_rate_buckets: Tuple[str, ...] = ()
    tts_quality_num_buckets: int = 0
    tts_quality_features: Tuple[str, ...] = ()
    tts_quality_buckets: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    tts_quality_dropout: Dict[str, float] | None = None
    tts_default_voices_dir: str | None = None

    @property
    def share_tokenizer(self) -> bool:
        return self.num_tokenizer == 0

    @property
    def zmq_frontend_addr(self) -> str:
        return "ipc:///tmp/zonos2_3" + self._unique_suffix

    @property
    def zmq_tokenizer_addr(self) -> str:
        if self.share_tokenizer:
            return self.zmq_detokenizer_addr
        result = "ipc:///tmp/zonos2_4" + self._unique_suffix
        assert result != self.zmq_detokenizer_addr
        return result

    @property
    def tokenizer_create_addr(self) -> bool:
        return self.share_tokenizer

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return not self.share_tokenizer

    @property
    def frontend_create_tokenizer_link(self) -> bool:
        return not self.share_tokenizer

    @property
    def distributed_addr(self) -> str:
        return f"tcp://127.0.0.1:{self.server_port + 1}"


def parse_args(args: List[str], run_shell: bool = False) -> Tuple[ServerArgs, bool]:
    """
    Parse command line arguments and return an EngineConfig.

    Args:
        args: Command line arguments (e.g., sys.argv[1:])

    Returns:
        EngineConfig instance with parsed arguments
    """
    from zonos2.attention import validate_backend
    from zonos2.kvcache import SUPPORTED_CACHE_MANAGER

    def validate_moe_backend(backend: str) -> str:
        """Validate MoE backend argument."""
        # Accept any non-empty string as backend name
        # Specific backend validation can be added later as backends are implemented
        if not backend:
            from argparse import ArgumentTypeError

            raise ArgumentTypeError(f"MoE backend must be a non-empty string, got: {backend}")
        return backend

    parser = argparse.ArgumentParser(description="Zonos2 Server Arguments")

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Zonos2 TTS checkpoint: a Hugging Face repo id (e.g. Zyphra/ZONOS2) or a local path.",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Data type for model weights and activations. 'auto' will use FP16 for FP32/FP16 models and BF16 for BF16 models.",
    )

    parser.add_argument(
        "--tensor-parallel-size",
        "--tp-size",
        type=int,
        default=1,
        help="The tensor parallelism size.",
    )

    parser.add_argument(
        "--max-running-requests",
        type=int,
        dest="max_running_req",
        default=ServerArgs.max_running_req,
        help="The maximum number of running requests.",
    )

    parser.add_argument(
        "--max-seq-len-override",
        type=int,
        default=ServerArgs.max_seq_len_override,
        help="The maximum sequence length override.",
    )

    parser.add_argument(
        "--memory-ratio",
        type=float,
        default=ServerArgs.memory_ratio,
        help="The fraction of GPU memory to use for KV cache.",
    )

    assert ServerArgs.use_dummy_weight == False
    parser.add_argument(
        "--dummy-weight",
        action="store_true",
        dest="use_dummy_weight",
        help="Use dummy weights for testing.",
    )

    assert ServerArgs.use_pynccl == True
    parser.add_argument(
        "--disable-pynccl",
        action="store_false",
        dest="use_pynccl",
        help="Disable PyNCCL for tensor parallelism.",
    )

    parser.add_argument(
        "--host",
        type=str,
        dest="server_host",
        default=ServerArgs.server_host,
        help="The host address for the server.",
    )

    parser.add_argument(
        "--port",
        type=int,
        dest="server_port",
        default=ServerArgs.server_port,
        help="The port number for the server to listen on.",
    )

    parser.add_argument(
        "--cuda-graph-max-bs",
        "--graph",
        type=int,
        default=ServerArgs.cuda_graph_max_bs,
        help="The maximum batch size for CUDA graph capture. None means auto-tuning based on the GPU memory.",
    )

    parser.add_argument(
        "--num-tokenizer",
        "--tokenizer-count",
        type=int,
        default=ServerArgs.num_tokenizer,
        help="The number of tokenizer processes to launch. 0 means the tokenizer is shared with the detokenizer.",
    )

    parser.add_argument(
        "--max-prefill-length",
        "--max-extend-length",
        type=int,
        dest="max_extend_tokens",
        default=ServerArgs.max_extend_tokens,
        help="Chunk Prefill maximum chunk size in tokens.",
    )

    parser.add_argument(
        "--num-pages",
        "--num-tokens",
        dest="num_page_override",
        type=int,
        default=ServerArgs.num_page_override,
        help="Set the maximum number of pages for KVCache.",
    )

    parser.add_argument(
        "--attention-backend",
        "--attn",
        type=validate_backend,
        default=ServerArgs.attention_backend,
        help="The attention backend to use. If two backends are specified,"
        " the first one is used for prefill and the second one for decode.",
    )

    parser.add_argument(
        "--cache-type",
        type=str,
        default=ServerArgs.cache_type,
        choices=SUPPORTED_CACHE_MANAGER.supported_names(),
        help="The KV cache management strategy.",
    )

    parser.add_argument(
        "--moe-backend",
        type=validate_moe_backend,
        default=ServerArgs.moe_backend,
        help="The MoE backend to use.",
    )

    parser.add_argument(
        "--shell-mode",
        action="store_true",
        help="Run the server in shell mode.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging (shorthand for --log-level DEBUG).",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the log level (default: INFO). Overrides LOG_LEVEL env var.",
    )

    # TTS arguments
    parser.add_argument(
        "--tts-n-codebooks",
        type=int,
        default=ServerArgs.tts_n_codebooks,
        help="Number of audio codebooks for TTS (default: 9).",
    )

    parser.add_argument(
        "--tts-audio-pad-id",
        type=int,
        default=ServerArgs.tts_audio_pad_id,
        help="Audio padding token ID for TTS (default: 1025).",
    )

    parser.add_argument(
        "--tts-text-vocab",
        type=int,
        default=ServerArgs.tts_text_vocab,
        help="Text vocabulary size for TTS. If not set, auto-detected from model.",
    )

    parser.add_argument(
        "--tts-sample-rate",
        type=int,
        default=ServerArgs.tts_sample_rate,
        help="Audio sample rate for TTS output (default: 44100 for DAC).",
    )

    parser.add_argument(
        "--tts-default-voices-dir",
        type=str,
        default=ServerArgs.tts_default_voices_dir,
        help="Directory of default speaker audio or .npy/.npz embedding files to pre-populate in the TTS UI.",
    )

    # Parse arguments
    kwargs = parser.parse_args(args).__dict__.copy()

    # Apply log level before any init_logger calls
    debug_mode = kwargs.pop("debug")
    log_level = kwargs.pop("log_level")
    if debug_mode:
        from zonos2.utils import set_log_level

        set_log_level("DEBUG")
    elif log_level is not None:
        from zonos2.utils import set_log_level

        set_log_level(log_level)

    # resolve some arguments
    run_shell |= kwargs.pop("shell_mode")
    if run_shell:
        kwargs["cuda_graph_max_bs"] = 1
        kwargs["max_running_req"] = 1
        kwargs["silent_output"] = True

    if kwargs["model_path"].startswith("~"):
        kwargs["model_path"] = os.path.expanduser(kwargs["model_path"])
    if kwargs.get("tts_default_voices_dir"):
        kwargs["tts_default_voices_dir"] = os.path.expanduser(kwargs["tts_default_voices_dir"])

    DTYPE_MAP = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    checkpoint_config = cached_load_checkpoint_config(kwargs["model_path"])

    if (dtype_str := kwargs["dtype"]) != "auto":
        kwargs["dtype"] = DTYPE_MAP[dtype_str]
    else:
        dtype_or_str = getattr(checkpoint_config, "dtype", "bfloat16")
        if isinstance(dtype_or_str, str):
            kwargs["dtype"] = DTYPE_MAP.get(dtype_or_str, torch.bfloat16)
        else:
            kwargs["dtype"] = dtype_or_str

    kwargs["tp_info"] = DistributedInfo(0, kwargs["tensor_parallel_size"])
    del kwargs["tensor_parallel_size"]

    # Auto-detect TTS settings from model config if not provided
    if kwargs.get("tts_text_vocab") is None:
        text_vocab = getattr(checkpoint_config, "text_vocab", None)
        if text_vocab is not None:
            kwargs["tts_text_vocab"] = text_vocab

    # Auto-detect n_codebooks
    if kwargs.get("tts_n_codebooks") == ServerArgs.tts_n_codebooks:  # still default
        n_codebooks = getattr(checkpoint_config, "n_codebooks", None)
        if n_codebooks is not None:
            kwargs["tts_n_codebooks"] = n_codebooks

    # Auto-detect audio_pad_id
    if kwargs.get("tts_audio_pad_id") == ServerArgs.tts_audio_pad_id:  # still default
        audio_pad_id = getattr(checkpoint_config, "audio_pad_id", None)
        if audio_pad_id is not None:
            kwargs["tts_audio_pad_id"] = audio_pad_id

    speaking_rate_num_buckets = int(getattr(checkpoint_config, "speaking_rate_num_buckets", 0) or 0)
    if speaking_rate_num_buckets > 0:
        kwargs["tts_speaking_rate_num_buckets"] = speaking_rate_num_buckets
        kwargs["tts_speaking_rate_buckets"] = tuple(
            getattr(checkpoint_config, "speaking_rate_buckets", None) or ()
        )

    if bool(getattr(checkpoint_config, "speaker_background_token_enabled", False)):
        kwargs["tts_speaker_background_token_enabled"] = True
        if bool(getattr(checkpoint_config, "accurate_mode_token_enabled", False)):
            kwargs["tts_accurate_mode_token_enabled"] = True

    quality_num_buckets = int(getattr(checkpoint_config, "quality_num_buckets", 0) or 0)
    if quality_num_buckets > 0:
        raw_quality_buckets = getattr(checkpoint_config, "quality_buckets", None) or {}
        raw_quality_features = (
            getattr(checkpoint_config, "quality_features", None) or raw_quality_buckets.keys()
        )
        quality_features = tuple(str(item) for item in (raw_quality_features or ()))
        kwargs["tts_quality_num_buckets"] = quality_num_buckets
        kwargs["tts_quality_features"] = quality_features
        kwargs["tts_quality_buckets"] = {
            str(feature): tuple(
                str(item) for item in (raw_quality_buckets.get(feature, ()) or ())
            )
            for feature in quality_features
        }
        kwargs["tts_quality_dropout"] = getattr(checkpoint_config, "quality_dropout", None)

    # Debug: print TTS settings
    logger = init_logger(__name__)
    logger.info(
        f"TTS settings: n_codebooks={kwargs.get('tts_n_codebooks')}, "
        f"text_vocab={kwargs.get('tts_text_vocab')}, "
        f"speaking_rate_num_buckets={kwargs.get('tts_speaking_rate_num_buckets', 0)}, "
        f"quality_num_buckets={kwargs.get('tts_quality_num_buckets', 0)}, "
        f"speaker_background_token={kwargs.get('tts_speaker_background_token_enabled', False)}, "
        f"accurate_mode_token={kwargs.get('tts_accurate_mode_token_enabled', False)}"
    )

    result = ServerArgs(**kwargs)
    logger = init_logger(__name__)
    logger.info(f"Parsed arguments:\n{result}")
    return result, run_shell
