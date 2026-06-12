from __future__ import annotations

import logging
import multiprocessing as mp
import sys
from dataclasses import replace
from typing import TYPE_CHECKING

from zonos2.distributed import DistributedInfo
from zonos2.utils import init_logger

if TYPE_CHECKING:
    from .args import ServerArgs


def _run_scheduler(args: ServerArgs, ack_queue: mp.Queue[str]) -> None:
    import torch

    # Bind this process to its GPU before importing the model: importing the
    # model eagerly pulls in sgl_kernel, which creates the CUDA context, and it
    # must land on this rank's device rather than the default cuda:0.
    if torch.cuda.is_available():
        torch.cuda.set_device(args.tp_info.rank)

    from zonos2.scheduler import TTSScheduler

    with torch.inference_mode():
        scheduler = TTSScheduler(args)
        scheduler.sync_all_ranks()

        if args.tp_info.is_primary():
            ack_queue.put("Scheduler is ready")

        if args.silent_output:
            logging.disable(logging.INFO)

        try:
            scheduler.run_forever()
        except KeyboardInterrupt:
            logger = init_logger(__name__)
            if scheduler.tp_info.is_primary():
                print()  # for a clean newline after ^C
                logger.info("Scheduler exiting gracefully...")
            scheduler.shutdown()


def launch_server(run_shell: bool = False) -> None:
    from .api_server import run_api_server
    from .args import parse_args

    server_args, run_shell = parse_args(sys.argv[1:], run_shell)
    logger = init_logger(__name__, "initializer")

    def start_subprocess() -> None:
        import multiprocessing as mp

        from zonos2.tokenizer import tokenize_worker

        mp.set_start_method("spawn", force=True)

        world_size = server_args.tp_info.size
        # a multiprocessing queue to receive ack from subprocesses
        # so that we can guarantee all subprocesses are ready
        ack_queue: mp.Queue[str] = mp.Queue()

        for i in range(world_size):
            new_args = replace(
                server_args,
                tp_info=DistributedInfo(i, world_size),
            )
            mp.Process(
                target=_run_scheduler,
                args=(new_args, ack_queue),
                daemon=False,
                name=f"zonos2-TP{i}-scheduler",
            ).start()

        num_tokenizers = server_args.num_tokenizer

        # Common tokenizer kwargs
        tokenizer_common_kwargs = {
            "backend_addr": server_args.zmq_backend_addr,
            "frontend_addr": server_args.zmq_frontend_addr,
            "local_bs": 1,
            "ack_queue": ack_queue,
            "n_codebooks": server_args.tts_n_codebooks,
            "audio_pad_id": server_args.tts_audio_pad_id,
            "text_vocab": server_args.tts_text_vocab,
            "speaking_rate_num_buckets": server_args.tts_speaking_rate_num_buckets,
            "quality_bucket_counts": [
                len(server_args.tts_quality_buckets.get(feature, ()))
                for feature in server_args.tts_quality_features
            ],
            "speaker_background_num_buckets": (
                2 if server_args.tts_speaker_background_token_enabled else 0
            ),
            "accurate_mode_num_buckets": (
                1
                if server_args.tts_speaker_background_token_enabled
                and server_args.tts_accurate_mode_token_enabled
                else 0
            ),
        }

        # DeTokenizer, only 1
        mp.Process(
            target=tokenize_worker,
            kwargs={
                **tokenizer_common_kwargs,
                "addr": server_args.zmq_detokenizer_addr,
                "create": server_args.tokenizer_create_addr,
                "tokenizer_id": num_tokenizers,
            },
            daemon=False,
            name="zonos2-detokenizer-0",
        ).start()
        for i in range(num_tokenizers):
            mp.Process(
                target=tokenize_worker,
                kwargs={
                    **tokenizer_common_kwargs,
                    "addr": server_args.zmq_tokenizer_addr,
                    "create": server_args.tokenizer_create_addr,
                    "tokenizer_id": i,
                },
                daemon=False,
                name=f"zonos2-tokenizer-{i}",
            ).start()

        # Wait for acknowledgments from all worker processes:
        # - world_size schedulers (but only primary rank sends ack)
        # - num_tokenizers tokenizers
        # - 1 detokenizer
        # Total acks expected: 1 + num_tokenizers + 1 = num_tokenizers + 2
        for _ in range(num_tokenizers + 2):
            logger.info(ack_queue.get())

    run_api_server(server_args, start_subprocess, run_shell=run_shell)


if __name__ == "__main__":
    launch_server()
