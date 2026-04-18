from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from ..utils import add_path, REPO_PATH

with add_path(REPO_PATH):
    from torchbenchmark.util.experiment.instantiator import (
        TorchBenchModelConfig,
    )

@dataclass
class TorchBenchModelWithNvtxConfig:
    model_cfg: TorchBenchModelConfig
    output: Path
    domain: Optional[str]
    warmup_iterations: int
    iterations: int

@dataclass
class TorchBenchModelWithNvtxResult:
    start_timestamp: int
    end_timestamp: int