"""
memory-consume userbenchmark
=============================
Measures training performance and detailed CUDA memory consumption across
all torchbenchmark models using PyTorch's official memory snapshot API:
  https://docs.pytorch.org/docs/stable/torch_cuda_memory.html

Per-model outputs
-----------------
  Metrics (in canonical JSON):
    - latency_ms            : ms per iteration
    - throughput_samples_s  : samples / second
    - peak_gpu_mb           : peak GPU memory allocated (MB)
    - reserved_gpu_mb       : peak GPU memory reserved from cudaMalloc (MB)
    - peak_cpu_mb           : peak CPU RSS delta (MB)
    - memory_timeline_mb    : GPU allocated MB sampled at each iteration

  Artifacts (per model, in output dir):
    - {model}_{test}_snapshot.pickle  : full CUDA memory snapshot, drag-drop
                                        into https://pytorch.org/memory_viz

Usage
-----
    python run_benchmark.py memory-consume
    python run_benchmark.py memory-consume --device cuda --test train
    python run_benchmark.py memory-consume --model resnet50 --iterations 10
    python run_benchmark.py memory-consume --model resnet50,hf_bert --no-snapshot
"""

import argparse
import gc
import importlib
import json
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.cuda

# ── Repo root & output dir ────────────────────────────────────────────────────

REPO_ROOT  = Path(__file__).parent.parent.parent.resolve()
BM_NAME    = "memory-consume"
OUTPUT_DIR = REPO_ROOT / ".userbenchmark" / BM_NAME

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_TEST         = "train"
DEFAULT_ITERATIONS   = 10
DEFAULT_WARMUP       = 3
DEFAULT_MAX_ENTRIES  = 100_000   # max alloc/free events kept in snapshot history

# ── Memory helpers ────────────────────────────────────────────────────────────

def reset_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

def allocated_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0

def peak_allocated_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0

def peak_reserved_mb() -> float:
    """Reserved = total cudaMalloc'd memory (includes free blocks in cache)."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_reserved() / 1024 / 1024
    return 0.0

def cpu_rss_mb() -> float:
    try:
        import resource
        # ru_maxrss is in KB on Linux
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        return 0.0

def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()

# ── Snapshot helpers ──────────────────────────────────────────────────────────

def start_memory_history(max_entries: int) -> None:
    """Enable CUDA memory history recording (alloc/free trace + stack frames)."""
    if not torch.cuda.is_available():
        return
    torch.cuda.memory._record_memory_history(
        enabled="all",       # record alloc + free
        context="all",       # include Python + C++ frames
        stacks="all",        # all stack frames
        max_entries=max_entries,
    )

def stop_memory_history() -> None:
    if not torch.cuda.is_available():
        return
    torch.cuda.memory._record_memory_history(enabled=None)

def dump_snapshot(path: Path) -> None:
    """
    Save a pickled CUDA memory snapshot to `path`.
    The file can be dragged into https://pytorch.org/memory_viz for
    interactive exploration of the Active Memory Timeline and Allocator State.
    """
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.memory._dump_snapshot(str(path))
    except Exception as e:
        print(f"    [warn] Failed to dump snapshot: {e}")

# ── Model discovery ───────────────────────────────────────────────────────────

def discover_models() -> List[str]:
    models_dir = REPO_ROOT / "torchbenchmark" / "models"
    if not models_dir.exists():
        raise RuntimeError(f"Models directory not found: {models_dir}")
    return sorted([
        d.name for d in models_dir.iterdir()
        if d.is_dir()
        and (d / "__init__.py").exists()
        and not d.name.startswith("_")
    ])

def load_model_class(model_name: str):
    module = importlib.import_module(f"torchbenchmark.models.{model_name}")
    return module.Model

# ── Core benchmark ────────────────────────────────────────────────────────────

def run_one_model(
    model_name:   str,
    device:       str,
    test:         str,
    batch_size:   Optional[int],
    iterations:   int,
    warmup:       int,
    snapshot_dir: Path,
    save_snapshot: bool,
    max_entries:  int,
) -> Dict[str, Any]:
    """
    Run one model and return a result dict with all metrics.
    If save_snapshot=True and device=cuda, also writes a .pickle snapshot file.
    """
    result: Dict[str, Any] = {
        "latency_ms":           None,
        "throughput_samples_s": None,
        "peak_gpu_mb":          None,
        "reserved_gpu_mb":      None,
        "peak_cpu_mb":          None,
        "memory_timeline_mb":   [],
        "batch_size":           None,
        "snapshot_path":        None,
        "error":                None,
    }

    try:
        ModelClass = load_model_class(model_name)

        # ── Construct model ───────────────────────────────────────────────────
        kwargs: Dict[str, Any] = dict(test=test, device=device)
        if batch_size is not None:
            kwargs["batch_size"] = batch_size
        model_instance = ModelClass(**kwargs)
        result["batch_size"] = model_instance.batch_size

        step_fn = model_instance.invoke

        # ── Warmup (no recording) ─────────────────────────────────────────────
        reset_stats()
        for _ in range(warmup):
            step_fn()
        sync()

        # ── Enable memory history recording ───────────────────────────────────
        if save_snapshot and device == "cuda":
            start_memory_history(max_entries)

        # ── Timed + traced iterations ─────────────────────────────────────────
        reset_stats()
        cpu_before = cpu_rss_mb()
        timeline:  List[float] = []
        sync()

        import time
        t_start = time.perf_counter()

        for _ in range(iterations):
            step_fn()
            sync()
            timeline.append(allocated_mb())

        t_end = time.perf_counter()

        # ── Collect metrics ───────────────────────────────────────────────────
        elapsed_s  = t_end - t_start
        latency_ms = elapsed_s / iterations * 1000.0
        throughput = (model_instance.batch_size * iterations) / elapsed_s

        result["latency_ms"]           = round(latency_ms, 3)
        result["throughput_samples_s"] = round(throughput, 3)
        result["peak_gpu_mb"]          = round(peak_allocated_mb(), 2)
        result["reserved_gpu_mb"]      = round(peak_reserved_mb(), 2)
        result["peak_cpu_mb"]          = round(max(cpu_rss_mb() - cpu_before, 0.0), 2)
        result["memory_timeline_mb"]   = [round(v, 2) for v in timeline]

        # ── Dump snapshot BEFORE stopping history ─────────────────────────────
        if save_snapshot and device == "cuda":
            snapshot_path = (
                snapshot_dir / f"{model_name}_{test}_snapshot.pickle"
            )
            dump_snapshot(snapshot_path)
            result["snapshot_path"] = str(snapshot_path)

    except NotImplementedError:
        result["error"] = f"{test} not implemented for {model_name}"
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
    finally:
        # Always stop history and clean up
        if save_snapshot and device == "cuda":
            stop_memory_history()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return result

# ── Metrics flattening ────────────────────────────────────────────────────────

def build_metrics(
    results: Dict[str, Dict[str, Any]],
    test:    str,
) -> Dict[str, float]:
    """
    Flatten all per-model results into the canonical float-only metrics dict
    required by the userbenchmark interface.

    Key pattern:  {model}_{test}_{metric}
    Timeline:     {model}_{test}_memory_timeline_iter{N}_mb
    """
    metrics: Dict[str, float] = {}

    for model_name, r in results.items():
        if r["error"] is not None:
            metrics[f"{model_name}_{test}_error"] = float("nan")
            continue

        prefix = f"{model_name}_{test}"
        metrics[f"{prefix}_latency_ms"]           = r["latency_ms"]
        metrics[f"{prefix}_throughput_samples_s"]  = r["throughput_samples_s"]
        metrics[f"{prefix}_peak_gpu_mb"]           = r["peak_gpu_mb"]
        metrics[f"{prefix}_reserved_gpu_mb"]       = r["reserved_gpu_mb"]
        metrics[f"{prefix}_peak_cpu_mb"]           = r["peak_cpu_mb"]

        for i, mb in enumerate(r["memory_timeline_mb"]):
            metrics[f"{prefix}_memory_timeline_iter{i}_mb"] = mb

    return metrics

# ── Output writing ────────────────────────────────────────────────────────────

def write_output(
    metrics:   Dict[str, float],
    results:   Dict[str, Dict[str, Any]],
    args:      argparse.Namespace,
    out_dir:   Path,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # ── Canonical userbenchmark JSON ──────────────────────────────────────────
    output = {
        "name": BM_NAME,
        "environ": {
            "metrics_version":     "v0.1",
            "pytorch_git_version": getattr(torch.version, "git_version", "unknown"),
            "pytorch_version":     torch.__version__,
            "cuda_version":        torch.version.cuda or "N/A",
            "device":              args.device,
            "test":                args.test,
            "iterations":          args.iterations,
            "warmup":              args.warmup,
            "save_snapshot":       args.save_snapshot,
        },
        "metrics": metrics,
    }

    metrics_path = out_dir / f"metrics-{timestamp}.json"
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2)

    # ── Detailed sidecar (per-model full results, errors, snapshot paths) ─────
    detail_path = out_dir / f"details-{timestamp}.json"
    # memory_timeline_mb can be large — keep it in details only
    with open(detail_path, "w") as f:
        json.dump(results, f, indent=2)

    return metrics_path

# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "memory-consume: benchmark latency, throughput, and full CUDA "
            "memory consumption across all torchbenchmark models."
        ),
        add_help=True,
    )
    parser.add_argument(
        "--device", "-d",
        default=DEFAULT_DEVICE,
        choices=["cpu", "cuda"],
        help=f"Device to run on (default: {DEFAULT_DEVICE})",
    )
    parser.add_argument(
        "--test", "-t",
        default=DEFAULT_TEST,
        choices=["train", "eval"],
        help=f"Test mode (default: {DEFAULT_TEST})",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        type=str,
        help=(
            "Comma-separated list of model names to run. "
            "Runs all discovered models if not set."
        ),
    )
    parser.add_argument(
        "--batch-size", "-b",
        default=None,
        type=int,
        help="Batch size override. Uses each model's default if not set.",
    )
    parser.add_argument(
        "--iterations", "-i",
        default=DEFAULT_ITERATIONS,
        type=int,
        help=f"Number of timed iterations per model (default: {DEFAULT_ITERATIONS})",
    )
    parser.add_argument(
        "--warmup", "-w",
        default=DEFAULT_WARMUP,
        type=int,
        help=f"Number of warmup iterations (default: {DEFAULT_WARMUP})",
    )
    parser.add_argument(
        "--max-snapshot-entries",
        default=DEFAULT_MAX_ENTRIES,
        type=int,
        help=(
            f"Max alloc/free events kept in snapshot history "
            f"(default: {DEFAULT_MAX_ENTRIES}). "
            "Larger values = bigger .pickle files."
        ),
    )
    parser.add_argument(
        "--no-snapshot",
        dest="save_snapshot",
        action="store_false",
        default=True,
        help=(
            "Disable saving .pickle memory snapshots. "
            "Snapshots are saved by default when device=cuda."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        type=str,
        help=f"Directory to write output files (default: {OUTPUT_DIR})",
    )
    return parser.parse_args(args)

# ── Entry point (required by userbenchmark interface) ─────────────────────────

def run(args: List[str]) -> None:
    parsed = parse_args(args)

    out_dir = Path(parsed.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve model list
    if parsed.model:
        model_names = [m.strip() for m in parsed.model.split(",")]
    else:
        model_names = discover_models()

    snapshot_enabled = parsed.save_snapshot and parsed.device == "cuda"

    print(f"\n{'='*65}")
    print(f"  {BM_NAME}")
    print(f"  device={parsed.device}  test={parsed.test}")
    print(f"  iterations={parsed.iterations}  warmup={parsed.warmup}")
    print(f"  models={len(model_names)}")
    print(f"  memory snapshots={'enabled' if snapshot_enabled else 'disabled'}")
    if snapshot_enabled:
        print(f"  snapshot max_entries={parsed.max_snapshot_entries}")
        print(f"  visualize at: https://pytorch.org/memory_viz")
    print(f"{'='*65}\n")

    results: Dict[str, Dict[str, Any]] = {}

    for i, model_name in enumerate(model_names, 1):
        print(f"[{i:3d}/{len(model_names)}] {model_name} ...", end=" ", flush=True)

        result = run_one_model(
            model_name    = model_name,
            device        = parsed.device,
            test          = parsed.test,
            batch_size    = parsed.batch_size,
            iterations    = parsed.iterations,
            warmup        = parsed.warmup,
            snapshot_dir  = out_dir,
            save_snapshot = parsed.save_snapshot,
            max_entries   = parsed.max_snapshot_entries,
        )
        results[model_name] = result

        if result["error"]:
            print(f"FAILED — {result['error'].splitlines()[0]}")
        else:
            snap_note = ""
            if result["snapshot_path"]:
                snap_note = f"  snapshot=saved"
            print(
                f"OK  "
                f"latency={result['latency_ms']:.1f}ms  "
                f"throughput={result['throughput_samples_s']:.1f}samp/s  "
                f"gpu_peak={result['peak_gpu_mb']:.0f}MB  "
                f"gpu_reserved={result['reserved_gpu_mb']:.0f}MB  "
                f"cpu_peak={result['peak_cpu_mb']:.0f}MB"
                f"{snap_note}"
            )

    # Build flat metrics and write JSON outputs
    metrics      = build_metrics(results, parsed.test)
    metrics_path = write_output(metrics, results, parsed, out_dir)

    # Summary
    ok     = sum(1 for r in results.values() if r["error"] is None)
    failed = len(results) - ok
    snaps  = sum(1 for r in results.values() if r.get("snapshot_path"))

    print(f"\n{'='*65}")
    print(f"  Done: {ok} succeeded, {failed} failed")
    if snaps:
        print(f"  Snapshots saved: {snaps}  (drag .pickle into pytorch.org/memory_viz)")
    print(f"  Metrics: {metrics_path}")
    print(f"{'='*65}\n")
