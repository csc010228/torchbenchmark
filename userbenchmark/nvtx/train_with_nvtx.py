import time
import nvtx
from typing import Optional
import torch

from ..utils import add_path, REPO_PATH

with add_path(REPO_PATH):
    from torchbenchmark.util.experiment.instantiator import (
        load_model,
        TorchBenchModelConfig,
    )
    from torchbenchmark.util.model import (
        BenchmarkModel
    )
    from torchbenchmark.util.extra_args import (
        TEST_STAGE
    )
    from torchbenchmark.util.env_check import (
        is_staged_train_test
    )

domain: str = None

class AllRange:
    def __enter__(self):
        global domain
        nvtx.push_range(message="all", domain=domain)
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        global domain
        torch.cuda.synchronize()
        nvtx.pop_range(domain=domain)

class ForwardRange:
    def __enter__(self):
        global domain
        nvtx.push_range(message="forward", domain=domain)
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        global domain
        torch.cuda.synchronize()
        nvtx.pop_range(domain=domain)

class BackwardRange:
    def __enter__(self):
        global domain
        nvtx.push_range(message="backward", domain=domain)
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        global domain
        torch.cuda.synchronize()
        nvtx.pop_range(domain=domain)

class OptimizerRange:
    def __enter__(self):
        global domain
        nvtx.push_range(message="optimizer", domain=domain)
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        global domain
        torch.cuda.synchronize()
        nvtx.pop_range(domain=domain)


def load_model_with_nvtx(config: TorchBenchModelConfig, domain_name: Optional[str]) -> BenchmarkModel:
    """
    根据配置加载带有不同阶段计数器的模型实例
    这个模型实例后续运行时会在本进程中运行

    Args:
        config (TorchBenchModelConfig): 模型配置

    Returns:
        BenchmarkModel: 加载出来的模型实例
    """
    global domain
    domain = domain_name

    model = load_model(config)

    model.add_context(AllRange, TEST_STAGE.ALL)
    model.add_context(ForwardRange, TEST_STAGE.FORWARD)
    model.add_context(BackwardRange, TEST_STAGE.BACKWARD)
    model.add_context(OptimizerRange, TEST_STAGE.OPTIMIZER)

    return model

def run_model_train_with_nvtx(model: BenchmarkModel) -> None:
    """
    执行一个迭代的模型分阶段训练过程，并收集每个阶段的时间开销（单位：纳秒）

    Args:
        model (BenchmarkModel): 需要运行分阶段训练流程的模型实例。

    Returns:
        Dict[str, List[int]]: 
            返回包含训练各阶段耗时的字典，目前字段包括：
            
            - "all": List[int]  
                一个列表，每一个元素都是一次训练迭代的总耗时（单位：纳秒）
            - "forward": List[int]  
                一个列表，每一个元素都是一次训练迭代中的前向传播的总耗时（单位：纳秒）
            - "backward": List[int]  
                一个列表，每一个元素都是一次训练迭代中的反向传播总耗时（单位：纳秒）
            - "optimizer": List[int]  
                一个列表，每一个元素都是一次训练迭代中的优化器优化总耗时（单位：纳秒）

    Raises:
        ValueError:
            - model 不是 BenchmarkModel 实例
            - model.test 不是 "train"
            - 模型不支持 staged train
            - 模型定义了自带的 train 方法（与 staged 模式冲突）
    """
    if not (isinstance(model, BenchmarkModel)):
        raise ValueError(
            f"Expected BenchmarkModel, get type: {type(model)}"
        )
    
    if model.test != "train":
        raise ValueError(
            f"Expected train model, get test type: {model.test}"
        )
    
    if not is_staged_train_test(model):
        raise ValueError(
            f"Model {model.name} does not support staged train test."
        )
    
    if getattr(model, "train", None) != None:
        raise ValueError(
            f"Model {model.name} already has train method defined."
        )

    # 运行训练任务
    # 会执行 BenchmarkModel 的 _invoke_staged_train_test() 方法
    model.invoke()