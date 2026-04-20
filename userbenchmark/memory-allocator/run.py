import argparse
import sys
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path
import torch
from dataclasses import asdict

from ..utils import add_path, dump_output, get_output_dir, get_output_json, REPO_PATH

from .train_with_memory_allocator import IterationEventPairs, load_model_with_memory_allocator, run_model_train_with_memory_allocator

with add_path(REPO_PATH):
    from torchbenchmark.util.experiment.instantiator import (
        list_models,
        TorchBenchModelConfig,
    )

BM_NAME = "memory-allocator"
DEFAULT_WARNUP_ITERATIONS = 3
DEFAULT_ITERATIONS = 15
DEFAULT_BATCH_SIZE = 4

@dataclass
class IterationDuration:
    all: int
    forward: int
    backward: int
    optimize: int

def parse_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a", 
        "--allocator", 
        type=Path, 
        default=None, 
        help="Path to the .so file of the memory allocator.",
    )
    parser.add_argument(
        "-mf", 
        "--malloc_func", 
        default="malloc", 
        type=str, 
        help="Memory allocation function name, default is malloc.",
    )
    parser.add_argument(
        "-ff", 
        "--free_func", 
        default="free", 
        type=str, 
        help="Memory release function name, default is free.",
    )
    parser.add_argument(
        "-b", 
        "--batch-size", 
        default=DEFAULT_BATCH_SIZE, 
        type=int, 
        help="Batch size for model running.",
    )
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
        "-o", 
        "--output", 
        default=None, 
        type=str, 
        help="The default output json file.",
    )
    args = parser.parse_args(args)
    return args

def run_model_with_memory_allocator(model_cfg: TorchBenchModelConfig, warmup_iterations: int, iterations: int, verbose: bool = False) -> Tuple[List[IterationEventPairs], ...]:
    model = load_model_with_memory_allocator(model_cfg)

    # 进行 warmup
    warmup_iteration_event_pairs: List[IterationEventPairs] = []
    for warmup_iteration in range(warmup_iterations):
        if verbose:
            print(f"[warmup iteration {warmup_iteration}/{warmup_iterations}] Running {model_cfg}")
        warmup_iteration_event_pairs.append(run_model_train_with_memory_allocator(model))

    # 实际运行
    iteration_event_pairs: List[IterationEventPairs] = []
    for iteration in range(iterations):
        if verbose:
            print(f"[iteration {iteration}/{iterations}] Running {model_cfg}")
        iteration_event_pairs.append(run_model_train_with_memory_allocator(model))

    return (warmup_iteration_event_pairs, iteration_event_pairs)

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
    model_cfg = TorchBenchModelConfig(
        name=args.model,
        device=args.device,
        test=args.test,
        batch_size=args.batch_size,
        extra_args=[],
        extra_env=None,
    )

    # 启动指定的 memory allocator
    if args.allocator:
        torch.cuda.memory.change_current_allocator(torch.cuda.memory.CUDAPluggableAllocator(str(args.allocator), args.malloc_func, args.free_func))

    # 运行模型
    warmup_iteration_event_pairs, iteration_event_pairs = run_model_with_memory_allocator(model_cfg, args.warmup_iterations, args.iterations, args.verbose)

    # 等待设备执行结束
    torch.cuda.synchronize()

    # 获取结果
    warmup_iteration_durations: List[IterationDuration] = []
    for warmup_iteration_event_pair in warmup_iteration_event_pairs:
        warmup_iteration_durations.append(IterationDuration(
            all=warmup_iteration_event_pair.all_event_pair.start_event.elapsed_time(warmup_iteration_event_pair.all_event_pair.end_event),
            forward=warmup_iteration_event_pair.forward_event_pair.start_event.elapsed_time(warmup_iteration_event_pair.forward_event_pair.end_event),
            backward=warmup_iteration_event_pair.backward_event_pair.start_event.elapsed_time(warmup_iteration_event_pair.backward_event_pair.end_event),
            optimize=warmup_iteration_event_pair.optimize_event_pair.start_event.elapsed_time(warmup_iteration_event_pair.optimize_event_pair.end_event)
        ))

    iteration_durations: List[IterationDuration] = []
    for iteration_event_pair in iteration_event_pairs:
        iteration_durations.append(IterationDuration(
            all=iteration_event_pair.all_event_pair.start_event.elapsed_time(iteration_event_pair.all_event_pair.end_event),
            forward=iteration_event_pair.forward_event_pair.start_event.elapsed_time(iteration_event_pair.forward_event_pair.end_event),
            backward=iteration_event_pair.backward_event_pair.start_event.elapsed_time(iteration_event_pair.backward_event_pair.end_event),
            optimize=iteration_event_pair.optimize_event_pair.start_event.elapsed_time(iteration_event_pair.optimize_event_pair.end_event)
        ))

    # 将结果转换成可输出格式
    result_for_output = {
        "allocator": str(args.allocator),
        "malloc_func": args.malloc_func,
        "free_func": args.free_func,
        "model_cfg": asdict(model_cfg),
        "warmup_iterations": args.warmup_iterations,
        "iterations": args.iterations,
        "result": {
            "warmup_iterations_total_duration": sum([warmup_iteration_duration.all for warmup_iteration_duration in warmup_iteration_durations]),
            "iterations_total_duration": sum([iteration_duration.all for iteration_duration in iteration_durations]),
            "warmup_iteration_durations": [asdict(warmup_iteration_duration) for warmup_iteration_duration in warmup_iteration_durations],
            "iteration_durations": [asdict(iteration_duration) for iteration_duration in iteration_durations],
        }
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
