import argparse
import sys
import json
import nvtx
from typing import List, Optional
from pathlib import Path
import torch
from dataclasses import asdict
import time

from ..utils import add_path, dump_output, get_output_dir, get_output_json, REPO_PATH

from .nvtx_config import TorchBenchModelWithNvtxConfig, TorchBenchModelWithNvtxResult

from .train_with_nvtx import load_model_with_nvtx, run_model_train_with_nvtx

with add_path(REPO_PATH):
    from torchbenchmark.util.experiment.instantiator import (
        list_models,
        TorchBenchModelConfig,
    )

BM_NAME = "nvtx"
DEFAULT_WARNUP_ITERATIONS = 3
DEFAULT_ITERATIONS = 15
DEFAULT_BATCH_SIZE = 4


def parse_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--iterations",
        default=DEFAULT_ITERATIONS,
        type=int,
        help="Number of iterations to run model stage test.",
    )
    parser.add_argument(
        "-wi",
        "--warmup-iterations",
        default=DEFAULT_WARNUP_ITERATIONS,
        type=int,
        help="Number of warm up iterations to run before model stage test.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Specify the model to run",
    )
    parser.add_argument("-d", "--device", default="cuda", help="Specify the device.")
    parser.add_argument("-t", "--test", default="train", help="Specify the test.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "-b",
        "--batch-size",
        default=DEFAULT_BATCH_SIZE,
        type=int,
        help="Batch size for model running.",
    )
    parser.add_argument(
        "-o", 
        "--output", 
        default=None, 
        type=str, 
        help="The default output json file.", 
    )
    parser.add_argument(
        "--domain", 
        type=str, 
        default=None, 
        help="The nvtx domain name.", 
    )
    args = parser.parse_args(args)
    return args

def run_model_with_nvtx(cfg: TorchBenchModelWithNvtxConfig, verbose: bool = False) -> TorchBenchModelWithNvtxResult:
    start_timestamp = time.time_ns()

    model = load_model_with_nvtx(cfg.model_cfg, cfg.domain)

    # 进行 warmup
    with nvtx.annotate(message="warmup", domain=cfg.domain):
        for warmup_iteration in range(cfg.warmup_iterations):
            if verbose:
                print(f"[warmup iteration {warmup_iteration}/{cfg.warmup_iterations}] Running {cfg}")
            with nvtx.annotate(message="warmup_iteration" + str(warmup_iteration), domain=cfg.domain):
                run_model_train_with_nvtx(model)
                torch.cuda.synchronize()

    # 实际运行
    with nvtx.annotate(message="actual_test", domain=cfg.domain):
        for iteration in range(cfg.iterations):
            if verbose:
                print(f"[iteration {iteration}/{cfg.iterations}] Running {cfg}")
            with nvtx.annotate(message="iteration" + str(iteration), domain=cfg.domain):
                run_model_train_with_nvtx(model)
                torch.cuda.synchronize()

    end_timestamp = time.time_ns()

    return TorchBenchModelWithNvtxResult(start_timestamp, end_timestamp)


def run(args: List[str]):
    # 解析命令行
    args = parse_args(args)
    models = list_models()
    if args.model not in models:
        print(f"Error: Unknown model '{args.model}'.")
        print("Available models:")
        for m in models:
            print(f"  - {m}")
        sys.exit(1)


    # 生成要执行的配置
    cfg = TorchBenchModelWithNvtxConfig(
        model_cfg = TorchBenchModelConfig(
            name=args.model,
            device=args.device,
            test=args.test,
            batch_size=args.batch_size,
            extra_args=[],
            extra_env=None,
        ),
        output=args.output,
        domain=args.domain,
        warmup_iterations=args.warmup_iterations,
        iterations=args.iterations,
    )

    # 运行配置，获取结果
    result = run_model_with_nvtx(cfg, args.verbose)

    # 将结果转换成可输出格式
    result_for_output = {
        "cfg": asdict(cfg),
        "res": asdict(result)
    }

    # 如果没有输出文件路径，那么就直接将结果输出到命令行
    if args.output is None:
        if args.verbose:
            print(result_for_output)
        return
    
    # 如果有输出文件路径，将结果保存到文件中
    output_json = get_output_json(BM_NAME, result_for_output)
    output = Path(args.output)
    if output.suffix != ".json":
        output = output.with_suffix(".json")
    output.parent.mkdir(exist_ok=True, parents=True)
    with open(output, "w") as f:
        json.dump(result_for_output, f, indent=4)
    dump_output(BM_NAME, output_json)
