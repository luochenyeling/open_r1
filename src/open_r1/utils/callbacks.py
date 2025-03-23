#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
训练回调函数模块，实现了训练过程中的各种回调功能。

主要功能：
1. 检查SLURM队列系统可用性
2. 提供配置对象的临时替代方案
3. 实现模型检查点推送到Hub的回调
4. 管理训练过程中的各种回调函数

主要组件：
1. is_slurm_available: 检查SLURM队列系统是否可用
2. DummyConfig: 用于临时存储配置信息的类
3. PushToHubRevisionCallback: 处理模型检查点推送到Hub的回调类
4. CALLBACKS: 回调函数注册表
5. get_callbacks: 获取训练配置中指定的回调函数列表
"""

import subprocess
from typing import List

from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from .evaluation import run_benchmark_jobs
from .hub import push_to_hub_revision


def is_slurm_available() -> bool:
    """
    检查SLURM队列系统是否可用
    
    返回:
        bool: 如果SLURM可用返回True，否则返回False
    """
    try:
        subprocess.run(["sinfo"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False


class DummyConfig:
    """
    用于临时存储配置信息的类
    
    用于解决在使用dataclasses.replace或实例化新的SFTConfig时
    可能破坏accelerator分布式状态的问题
    """
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class PushToHubRevisionCallback(TrainerCallback):
    """
    处理模型检查点推送到Hub的回调类
    
    在模型保存时触发，将检查点推送到HuggingFace Hub，
    并在SLURM可用时运行基准测试任务
    """
    def __init__(self, model_config) -> None:
        """
        初始化回调函数
        
        参数:
            model_config: 模型配置对象
        """
        self.model_config = model_config

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        在模型保存时触发的回调函数
        
        参数:
            args: 训练参数
            state: 训练状态
            control: 训练控制对象
            **kwargs: 其他参数
        """
        if state.is_world_process_zero:
            global_step = state.global_step

            # 创建临时配置对象，避免破坏accelerator分布式状态
            dummy_config = DummyConfig(
                hub_model_id=args.hub_model_id,
                hub_model_revision=f"{args.hub_model_revision}-step-{global_step:09d}",
                output_dir=f"{args.output_dir}/checkpoint-{global_step}",
                system_prompt=args.system_prompt,
            )

            # 推送检查点到Hub，忽略优化器状态文件
            future = push_to_hub_revision(
                dummy_config, extra_ignore_patterns=["*.pt"]
            )

            # 如果SLURM可用，设置基准测试任务
            if is_slurm_available():
                dummy_config.benchmarks = args.benchmarks

                def run_benchmark_callback(_):
                    print(f"Checkpoint {global_step} pushed to hub.")
                    run_benchmark_jobs(dummy_config, self.model_config)

                future.add_done_callback(run_benchmark_callback)


# 回调函数注册表
CALLBACKS = {
    "push_to_hub_revision": PushToHubRevisionCallback,
}


def get_callbacks(train_config, model_config) -> List[TrainerCallback]:
    """
    获取训练配置中指定的回调函数列表
    
    参数:
        train_config: 训练配置对象
        model_config: 模型配置对象
        
    返回:
        List[TrainerCallback]: 回调函数列表
        
    异常:
        ValueError: 当指定的回调函数未在CALLBACKS中注册时抛出
    """
    callbacks = []
    for callback_name in train_config.callbacks:
        if callback_name not in CALLBACKS:
            raise ValueError(f"Callback {callback_name} not found in CALLBACKS.")
        callbacks.append(CALLBACKS[callback_name](model_config))

    return callbacks
