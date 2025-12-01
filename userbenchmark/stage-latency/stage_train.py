import argparse
import itertools
import json
import time
from datetime import datetime
from typing import Dict, List

import numpy as np
import yaml

from ..utils import add_path, dump_output, get_output_dir, get_output_json, REPO_PATH

with add_path(REPO_PATH):
    from torchbenchmark import (
        ModelTask,
    )
    from torchbenchmark._components._impl.workers.subprocess_rpc import (
        ChildTraceException,
        UnserializableException,
    )
    from torchbenchmark.util.experiment.instantiator import (
        list_models,
        load_model,
        TorchBenchModelConfig,
    )
    from torchbenchmark.util.experiment.metrics import (
        get_model_test_metrics,
        TorchBenchModelMetrics,
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

all_latency = None
class AllTimer:
    def __enter__(self):
        self.t0 = time.time_ns()
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        self.t1 = time.time_ns()
        global all_latency
        all_latency = (self.t1 - self.t0)

forward_latency = None
class ForwardTimer:
    def __enter__(self):
        self.t0 = time.time_ns()
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        self.t1 = time.time_ns()
        global forward_latency
        forward_latency = (self.t1 - self.t0)

backward_latency = None
class BackwardTimer:
    def __enter__(self):
        self.t0 = time.time_ns()
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        self.t1 = time.time_ns()
        global backward_latency
        backward_latency = (self.t1 - self.t0)

optimizer_latency = None
class OptimizerTimer:
    def __enter__(self):
        self.t0 = time.time_ns()
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        self.t1 = time.time_ns()
        global optimizer_latency
        optimizer_latency = (self.t1 - self.t0)


def load_model_with_stage_timer(config: TorchBenchModelConfig) -> BenchmarkModel:
    """
    根据配置加载带有不同阶段计数器的模型实例
    这个模型实例后续运行时会在本进程中运行

    Args:
        config (TorchBenchModelConfig): 模型配置

    Returns:
        BenchmarkModel: 加载出来的模型实例
    """
    model = load_model(config)

    model.add_context(AllTimer, TEST_STAGE.ALL)
    model.add_context(ForwardTimer, TEST_STAGE.FORWARD)
    model.add_context(BackwardTimer, TEST_STAGE.BACKWARD)
    model.add_context(OptimizerTimer, TEST_STAGE.OPTIMIZER)

    return model

# def load_model_task_with_stage_timer(config: TorchBenchModelConfig) -> ModelTask:
#     """
#     根据配置加载带有不同阶段计数器的模型任务实例
#     这个模型任务实例后续运行时会新启动一个进程运行

#     Args:
#         config (TorchBenchModelConfig): 模型配置

#     Returns:
#         ModelTask: 加载出来的模型任务实例
#     """
#     task = ModelTask(
#         config.name,
#         timeout=timeout,
#         extra_env=config.extra_env,
#         save_output_dir=config.output_dir,
#     )

#     task.worker.run("import time")
#     task.worker.run("from torchbenchmark.util.extra_args import (TEST_STAGE)")

#     task.worker.run("""class RunTimer:
#     def __enter__(self):
#         self.t0 = time.time_ns()
#         return self
        
#     def __exit__(self, exc_type, exc_value, traceback):
#         self.t1 = time.time_ns()
#         run_times.append(self.t1 - self.t0)""")
    
#     task.worker.run("""class ForwardTimer:
#     def __enter__(self):
#         self.t0 = time.time_ns()
#         return self
        
#     def __exit__(self, exc_type, exc_value, traceback):
#         self.t1 = time.time_ns()
#         forward_times.append(self.t1 - self.t0)""")
    
#     task.worker.run("""class BackwardTimer:
#     def __enter__(self):
#         self.t0 = time.time_ns()
#         return self
        
#     def __exit__(self, exc_type, exc_value, traceback):
#         self.t1 = time.time_ns()
#         backward_times.append(self.t1 - self.t0)""")
    
#     task.worker.run("""class OptimizerTimer:
#     def __enter__(self):
#         self.t0 = time.time_ns()
#         return self
        
#     def __exit__(self, exc_type, exc_value, traceback):
#         self.t1 = time.time_ns()
#         optimizer_times.append(self.t1 - self.t0)""")
    
#     task.worker.store("run_times", [])
#     task.worker.store("forward_times", [])
#     task.worker.store("backward_times", [])
#     task.worker.store("optimizer_times", [])

#     return task

def get_model_train_stage_latency(model: BenchmarkModel) -> Dict[str, int]:
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

    # 清空之前的计时数据
    global all_latency
    # all_latency = []
    global forward_latency
    # forward_latency = []
    global backward_latency
    # backward_latency = []
    global optimizer_latency
    # optimizer_latency = []

    # 运行训练任务
    # 会执行 BenchmarkModel 的 _invoke_staged_train_test() 方法
    model.invoke()

    # 获取各个阶段的时间开销
    stage_latency = {
        "all": all_latency, 
        "forward": forward_latency, 
        "backward": backward_latency, 
        "optimizer": optimizer_latency, 
    }
    
    return stage_latency