import argparse
import itertools
import json
from typing import List
from pathlib import Path

from ..utils import add_path, dump_output, get_output_dir, get_output_json, REPO_PATH

from .stage_train import load_model_with_stage_timer, get_model_train_stage_latency

with add_path(REPO_PATH):
    from torchbenchmark.util.experiment.instantiator import (
        list_models,
        TorchBenchModelConfig,
    )

BM_NAME = "stage-latency"
DEFAULT_ITERATIONS = 15
DEFAULT_BATCH_SIZE = 4


def generate_model_config(model_name: str, batch_size: int = None) -> List[TorchBenchModelConfig]:
    devices = ["cpu", "cuda"]
    tests = ["train", "eval"]
    cfgs = itertools.product(*[devices, tests])
    result = [
        TorchBenchModelConfig(
            name=model_name,
            device=device,
            test=test,
            batch_size=batch_size,
            extra_args=[],
            extra_env=None,
        )
        for device, test in cfgs
    ]
    return result


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
        "-m",
        "--models",
        default="",
        help="Specify the models to run, default (empty) runs all models.",
    )
    parser.add_argument("-d", "--device", default="cuda", help="Specify the device.")
    parser.add_argument("-t", "--test", default="train", help="Specify the test.")
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
    args = parser.parse_args(args)
    return args


def generate_filter(args: argparse.Namespace):
    allowed_models = args.models
    if allowed_models:
        allowed_models = (
            allowed_models.split(",") if "," in allowed_models else [allowed_models]
        )
    allowed_devices = args.device
    allowed_devices = (
        allowed_devices.split(",") if "," in allowed_devices else [allowed_devices]
    )
    allowed_tests = args.test
    allowed_tests = (
        allowed_tests.split(",") if "," in allowed_tests else [allowed_tests]
    )

    def cfg_filter(cfg: TorchBenchModelConfig) -> bool:
        if cfg.device in allowed_devices and cfg.test in allowed_tests:
            if not allowed_models:
                return True
            else:
                return cfg.name in allowed_models
        return False

    return cfg_filter


def run(args: List[str]):
    args = parse_args(args)
    # output_dir = get_output_dir(BM_NAME)
    output = args.output
    models = list_models()
    batch_sizes = [args.batch_size] * len(models)
    cfgs = list(itertools.chain(*map(generate_model_config, models, batch_sizes)))
    cfg_filter = generate_filter(args)
    # run a model cfg and get latencies
    full_results = []
    for cfg in filter(cfg_filter, cfgs):
        cfg_dict = cfg.__dict__
        cfg_dict["output"] = output
        cfg_dict["iterations"] = args.iterations
        single_cfg_result = {
            "cfg": cfg_dict, 
            "all_latencies": [], 
            "forward_latencies": [], 
            "backward_latencies": [], 
            "optimizer_latencies": [], 
        }
        try:
            model = load_model_with_stage_timer(cfg)
            for _iteration in range(args.iterations):
                print(f"[iteration {_iteration}/{args.iterations}] Running {cfg}")
                metrics = get_model_train_stage_latency(model)
                single_cfg_result["all_latencies"].append(metrics["all"])
                single_cfg_result["forward_latencies"].append(metrics["forward"])
                single_cfg_result["backward_latencies"].append(metrics["backward"])
                single_cfg_result["optimizer_latencies"].append(metrics["optimizer"])
        finally:
            # Remove model reference to trigger deletion in gc
            model = None
        full_results.append(single_cfg_result)

    if output is None:
        print(full_results)
        return
    
    output_json = get_output_json(BM_NAME, full_results)
    output = Path(output)
    if output.suffix != ".json":
        output = output.with_suffix(".json")
    output.parent.mkdir(exist_ok=True, parents=True)
    with open(output, "w") as f:
        json.dump(full_results, f, indent=4)
    dump_output(BM_NAME, output_json)
