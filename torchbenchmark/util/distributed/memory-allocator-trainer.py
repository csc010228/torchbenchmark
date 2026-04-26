import os
import argparse
from pathlib import Path
import os

import torch
import torch.distributed as dist
from torchbenchmark.util.e2emodel import E2EBenchmarkModel
from .trainer import Trainer


class ArgumentError(Exception):
    pass

class _RaisingParser(argparse.ArgumentParser):
    def error(self, message: str):
        raise ArgumentError(message)


class MemoryAllocatorTrainer(Trainer):
    def __init__(self, args, model_class, mode="SPMD", model_args=None):
        self.args = args
        self.model_args = model_args
        self.model_class = model_class
        self.mode = mode

        # 提取所需参数
        self.extract_model_args()

        # 指定的 memory allocator
        if self.allocator:
            torch.cuda.memory.change_current_allocator(torch.cuda.memory.CUDAPluggableAllocator(str(self.allocator), self.malloc_func, self.free_func))

        self.local_rank = int(os.getenv("LOCAL_RANK", -1))
        self.setup()

        extra_args = [
            "--distributed",
            self.args.distributed,
        ]
        extra_args.extend(self.model_args)

        # create model instance after Trainer setup, so that
        # visible devices won't be revised in model constructor
        self.e2e_benchmark: E2EBenchmarkModel = model_class(
            "train", batch_size=self.batch_size, extra_args=extra_args
        )

        expected_attrs = [
            "model",
            "optimizer",
            "train_dataloader",
            "accelerator",
            "run_contexts",
        ]
        assert all(attr in dir(self.e2e_benchmark) for attr in expected_attrs), (
            "Missing attributes in the input E2EBenchmarkModel implementation: "
            f"{[attr for attr in expected_attrs if attr not in dir(self.e2e_benchmark)]}"
        )

        self.rank = dist.get_rank()

    def extract_model_args(self):
        """
        Extract known args from model_args list, remove them from model_args,
        and apply them to self.
        """
        parser = _RaisingParser(add_help=False)
        parser.add_argument("-a",   "--allocator",    type=Path, default=None)
        parser.add_argument("-mf",  "--malloc-func",  type=str,  default=None)
        parser.add_argument("-ff",  "--free-func",    type=str,  default=None)
        parser.add_argument("-b",   "--batch-size",   type=int,  default=None)
        parser.add_argument("-i",   "--iterations",   type=int,  default=None)

        parsed, self.model_args = parser.parse_known_args(self.model_args)

        # Validate allocator args: all or none
        allocator_args = {
            "--allocator":   parsed.allocator,
            "--malloc-func": parsed.malloc_func,
            "--free-func":   parsed.free_func,
        }
        provided = {k for k, v in allocator_args.items() if v is not None}
        missing  = {k for k, v in allocator_args.items() if v is None}

        if provided and missing:
            raise ArgumentError(
                f"Allocator args must all be specified or none. "
                f"Provided: {provided}, Missing: {missing}"
            )

        self.allocator    = parsed.allocator
        self.malloc_func  = parsed.malloc_func
        self.free_func    = parsed.free_func
        self.batch_size   = parsed.batch_size
        if parsed.iterations:
            self.DEFAULT_MEASURE_ITERATIONS   = parsed.iterations


class OriginalTrainer(MemoryAllocatorTrainer):
    def __init__(self, args, model_class, mode="SPMD", model_args=None):
        super().__init__(args, model_class, mode=mode, model_args=model_args)