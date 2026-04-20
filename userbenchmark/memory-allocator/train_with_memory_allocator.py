import torch
from dataclasses import dataclass

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

@dataclass
class EventPair:
    start_event: torch.cuda.Event = None
    end_event: torch.cuda.Event = None

@dataclass
class IterationEventPairs:
    all_event_pair: EventPair = None
    forward_event_pair: EventPair = None
    backward_event_pair: EventPair = None
    optimize_event_pair: EventPair = None

iteration_event_pairs: IterationEventPairs = None

class AllRange:
    def __enter__(self):
        global iteration_event_pairs
        iteration_event_pairs.all_event_pair.start_event.record()
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        global iteration_event_pairs
        iteration_event_pairs.all_event_pair.end_event.record()
        

class ForwardRange:
    def __enter__(self):
        global iteration_event_pairs
        iteration_event_pairs.forward_event_pair.start_event.record()
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        global iteration_event_pairs
        iteration_event_pairs.forward_event_pair.end_event.record()
        

class BackwardRange:
    def __enter__(self):
        global iteration_event_pairs
        iteration_event_pairs.backward_event_pair.start_event.record()
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        global iteration_event_pairs
        iteration_event_pairs.backward_event_pair.end_event.record()
        

class OptimizerRange:
    def __enter__(self):
        global iteration_event_pairs
        iteration_event_pairs.optimize_event_pair.start_event.record()
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        global iteration_event_pairs
        iteration_event_pairs.optimize_event_pair.end_event.record()


def load_model_with_memory_allocator(config: TorchBenchModelConfig) -> BenchmarkModel:
    """
    根据配置加载带有不同阶段计数器的模型实例
    这个模型实例后续运行时会在本进程中运行

    Args:
        config (TorchBenchModelConfig): 模型配置

    Returns:
        BenchmarkModel: 加载出来的模型实例
    """
    model = load_model(config)

    model.add_context(AllRange, TEST_STAGE.ALL)
    model.add_context(ForwardRange, TEST_STAGE.FORWARD)
    model.add_context(BackwardRange, TEST_STAGE.BACKWARD)
    model.add_context(OptimizerRange, TEST_STAGE.OPTIMIZER)

    return model

def run_model_train_with_memory_allocator(model: BenchmarkModel) -> IterationEventPairs:
    """
    执行一个迭代的模型分阶段训练过程，并收集每个阶段的时间开销（单位：纳秒）

    Args:
        model (BenchmarkModel): 需要运行分阶段训练流程的模型实例。

    Returns:
        IterationEventPairs: 
            返回包含训练各阶段的起始和结束 event：

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

    global iteration_event_pairs
    iteration_event_pairs = IterationEventPairs(
        all_event_pair=EventPair(
            start_event=torch.cuda.Event(enable_timing=True), 
            end_event=torch.cuda.Event(enable_timing=True)
        ),
        forward_event_pair=EventPair(
            start_event=torch.cuda.Event(enable_timing=True), 
            end_event=torch.cuda.Event(enable_timing=True)
        ),
        backward_event_pair=EventPair(
            start_event=torch.cuda.Event(enable_timing=True), 
            end_event=torch.cuda.Event(enable_timing=True)
        ),
        optimize_event_pair=EventPair(
            start_event=torch.cuda.Event(enable_timing=True), 
            end_event=torch.cuda.Event(enable_timing=True)
        ),
    )
    
    # 运行训练任务
    # 会执行 BenchmarkModel 的 _invoke_staged_train_test() 方法
    model.invoke()

    return iteration_event_pairs