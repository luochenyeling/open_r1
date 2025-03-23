# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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
配置模块，定义了训练过程中使用的各种配置类。

主要包含：
1. GRPOConfig: GRPO训练方法的配置类
2. SFTConfig: 监督微调训练的配置类
3. GRPOScriptArguments: GRPO训练脚本的参数配置类
"""

from dataclasses import dataclass, field
from typing import Optional

import trl


@dataclass
class GRPOConfig(trl.GRPOConfig):
    """
    GRPO训练方法的配置类，继承自trl.GRPOConfig
    
    参数:
        benchmarks (list[str]): 训练后要运行的基准测试列表
        callbacks (list[str]): 训练过程中要运行的回调函数列表
        chat_template (Optional[str]): 使用的聊天模板
        system_prompt (Optional[str]): 可选的系统提示词
        hub_model_revision (Optional[str]): 要推送到的Hub模型分支，默认为"main"
        overwrite_hub_revision (bool): 是否覆盖Hub上的修订版本
        push_to_hub_revision (bool): 是否推送到Hub的修订版本/分支
        wandb_entity (Optional[str]): 存储运行记录的实体
        wandb_project (Optional[str]): 存储运行记录的项目
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )
    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use."},
    )
    hub_model_revision: Optional[str] = field(
        default="main", metadata={"help": "The Hub model branch to push the model to."}
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )


@dataclass
class SFTConfig(trl.SFTConfig):
    """
    监督微调训练的配置类，继承自trl.SFTConfig
    
    参数:
        benchmarks (list[str]): 训练后要运行的基准测试列表
        callbacks (list[str]): 训练过程中要运行的回调函数列表
        chat_template (Optional[str]): 使用的聊天模板
        system_prompt (Optional[str]): 用于基准测试的可选系统提示词
        hub_model_revision (Optional[str]): 要推送到的Hub模型分支，默认为"main"
        overwrite_hub_revision (bool): 是否覆盖Hub上的修订版本
        push_to_hub_revision (bool): 是否推送到Hub的修订版本/分支
        wandb_entity (Optional[str]): 存储运行记录的实体
        wandb_project (Optional[str]): 存储运行记录的项目
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )
    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use for benchmarking."},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The Hub model branch to push the model to."},
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )


@dataclass
class GRPOScriptArguments(trl.ScriptArguments):
    """
    GRPO训练脚本的参数配置类，继承自trl.ScriptArguments
    
    参数:
        reward_funcs (list[str]): 奖励函数列表，可选值包括：
            - 'accuracy': 准确性奖励
            - 'format': 格式奖励
            - 'reasoning_steps': 推理步骤奖励
            - 'cosine': 余弦相似度奖励
            - 'repetition_penalty': 重复惩罚
            - 'length': 长度奖励
            - 'tag_count': 标签计数奖励
            - 'code': 代码奖励
            - 'code_format': 代码格式奖励
        cosine_min_value_wrong (float): 错误答案的余弦缩放最小奖励值
        cosine_max_value_wrong (float): 错误答案的余弦缩放最大奖励值
        cosine_min_value_correct (float): 正确答案的余弦缩放最小奖励值
        cosine_max_value_correct (float): 正确答案的余弦缩放最大奖励值
        cosine_max_len (int): 余弦缩放的最大长度
        code_language (str): 代码格式奖励使用的语言
        code_eval_test_batch_size (int): 代码评估的测试批次大小
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "tag_count"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', tag_count', 'code', 'code_format'"
        },
    )
    cosine_min_value_wrong: float = field(
        default=0.0,
        metadata={"help": "Minimum reward for wrong answers"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "Maximum reward for wrong answers"},
    )
    cosine_min_value_correct: float = field(
        default=0.5,
        metadata={"help": "Minimum reward for correct answers"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward for correct answers"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "Maximum length for scaling"},
    )
    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "Number of n-grams for repetition penalty reward"},
    )
    repetition_max_penalty: float = field(
        default=-1.0,
        metadata={"help": "Maximum (negative) penalty for for repetition penalty reward"},
    )
    code_language: str = field(
        default="python",
        metadata={
            "help": "Language for code format reward. Based on E2B supported languages https://e2b.dev/docs/code-interpreting/supported-languages",
            "choices": ["python", "javascript", "r", "java", "bash", "cpp"],
        },
    )
    code_eval_test_batch_size: int = field(
        default=1,
        metadata={
            "help": "for each generation, evaluate these many test cases in parallel, then check if any of them failed (0 score): if so stop evaluating; otherwise continue with the next batch of test cases. Useful to avoid overloading the eval server + save time on wrong solutions"
        },
    )
