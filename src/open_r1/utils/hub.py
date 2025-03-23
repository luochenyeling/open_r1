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
HuggingFace Hub交互模块，实现了与HuggingFace Hub的交互功能。

主要功能：
1. 模型检查点推送到Hub
2. 检查Hub版本是否存在
3. 从仓库ID获取模型参数数量
4. 计算vLLM所需的GPU数量

主要组件：
1. push_to_hub_revision: 将模型推送到Hub仓库的分支
2. check_hub_revision_exists: 检查Hub版本是否存在
3. get_param_count_from_repo_id: 从仓库ID获取模型参数数量
4. get_gpu_count_for_vllm: 计算vLLM所需的GPU数量
"""

import logging
import re
from concurrent.futures import Future

from transformers import AutoConfig

from huggingface_hub import (
    create_branch,
    create_repo,
    get_safetensors_metadata,
    list_repo_commits,
    list_repo_files,
    list_repo_refs,
    repo_exists,
    upload_folder,
)
from trl import GRPOConfig, SFTConfig


logger = logging.getLogger(__name__)


def push_to_hub_revision(training_args: SFTConfig | GRPOConfig, extra_ignore_patterns=[]) -> Future:
    """
    将模型推送到Hub仓库的分支
    
    参数:
        training_args: 训练参数配置
        extra_ignore_patterns: 额外的忽略文件模式列表
        
    返回:
        Future: 异步上传任务的Future对象
    """
    # 如果仓库不存在则创建
    repo_url = create_repo(repo_id=training_args.hub_model_id, private=True, exist_ok=True)
    # 获取初始提交作为分支起点
    initial_commit = list_repo_commits(training_args.hub_model_id)[-1]
    # 创建要推送到的分支
    create_branch(
        repo_id=training_args.hub_model_id,
        branch=training_args.hub_model_revision,
        revision=initial_commit.commit_id,
        exist_ok=True,
    )
    logger.info(f"Created target repo at {repo_url}")
    logger.info(f"Pushing to the Hub revision {training_args.hub_model_revision}...")
    # 设置要忽略的文件模式
    ignore_patterns = ["checkpoint-*", "*.pth"]
    ignore_patterns.extend(extra_ignore_patterns)
    # 异步上传文件夹
    future = upload_folder(
        repo_id=training_args.hub_model_id,
        folder_path=training_args.output_dir,
        revision=training_args.hub_model_revision,
        commit_message=f"Add {training_args.hub_model_revision} checkpoint",
        ignore_patterns=ignore_patterns,
        run_as_future=True,
    )
    logger.info(f"Pushed to {repo_url} revision {training_args.hub_model_revision} successfully!")

    return future


def check_hub_revision_exists(training_args: SFTConfig | GRPOConfig):
    """
    检查Hub版本是否存在
    
    参数:
        training_args: 训练参数配置
        
    异常:
        ValueError: 当版本已存在且未设置覆盖选项时抛出
    """
    if repo_exists(training_args.hub_model_id):
        if training_args.push_to_hub_revision is True:
            # 首先检查版本是否存在
            revisions = [rev.name for rev in list_repo_refs(training_args.hub_model_id).branches]
            # 如果版本存在，检查是否有README文件
            if training_args.hub_model_revision in revisions:
                repo_files = list_repo_files(
                    repo_id=training_args.hub_model_id, revision=training_args.hub_model_revision
                )
                if "README.md" in repo_files and training_args.overwrite_hub_revision is False:
                    raise ValueError(
                        f"Revision {training_args.hub_model_revision} already exists. "
                        "Use --overwrite_hub_revision to overwrite it."
                    )


def get_param_count_from_repo_id(repo_id: str) -> int:
    """
    从仓库ID获取模型参数数量
    
    尝试从safetensors元数据获取参数数量，或从仓库ID中查找类似42m、1.5b、0.5m或8x7b这样的模式
    
    参数:
        repo_id: HuggingFace仓库ID
        
    返回:
        int: 模型参数数量，如果无法获取则返回-1
    """
    try:
        # 尝试从safetensors元数据获取参数数量
        metadata = get_safetensors_metadata(repo_id)
        return list(metadata.parameter_count.values())[0]
    except Exception:
        # 匹配产品名称（如8x7b）和单个值（如42m）的模式
        pattern = r"((\d+(\.\d+)?)(x(\d+(\.\d+)?))?)([bm])"
        matches = re.findall(pattern, repo_id.lower())

        param_counts = []
        for full_match, number1, _, _, number2, _, unit in matches:
            if number2:  # 如果有第二个数字，说明是乘积形式
                number = float(number1) * float(number2)
            else:  # 否则是单个值
                number = float(number1)

            if unit == "b":
                number *= 1_000_000_000  # 转换为十亿
            elif unit == "m":
                number *= 1_000_000  # 转换为百万

            param_counts.append(number)

        if len(param_counts) > 0:
            # 返回最大的数字
            return int(max(param_counts))
        else:
            # 如果没有匹配则返回-1
            return -1


def get_gpu_count_for_vllm(model_name: str, revision: str = "main", num_gpus: int = 8) -> int:
    """
    计算vLLM所需的GPU数量
    
    vLLM要求注意力头数必须能被GPU数量整除，且64必须能被GPU数量整除。
    此函数根据模型中的注意力头数计算用于解码的GPU数量。
    
    参数:
        model_name: 模型名称
        revision: 模型版本，默认为"main"
        num_gpus: 初始GPU数量，默认为8
        
    返回:
        int: 计算得到的GPU数量
    """
    config = AutoConfig.from_pretrained(model_name, revision=revision, trust_remote_code=True)
    # 获取注意力头数
    num_heads = config.num_attention_heads
    # 减少GPU数量直到满足条件
    while num_heads % num_gpus != 0 or 64 % num_gpus != 0:
        logger.info(f"Reducing num_gpus from {num_gpus} to {num_gpus - 1} to make num_heads divisible by num_gpus")
        num_gpus -= 1
    return num_gpus
