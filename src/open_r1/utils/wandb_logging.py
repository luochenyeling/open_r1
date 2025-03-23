"""
Weights & Biases日志记录工具模块，用于配置和管理训练过程中的日志记录。

主要功能：
1. 初始化Weights & Biases日志记录环境
2. 设置W&B实体和项目名称

主要组件：
1. init_wandb_training: 初始化W&B训练日志记录环境的函数
"""

import os


def init_wandb_training(training_args):
    """
    初始化Weights & Biases训练日志记录环境
    
    设置W&B的实体和项目名称环境变量，用于训练过程中的日志记录
    
    参数:
        training_args: 训练参数配置，包含wandb_entity和wandb_project设置
    """
    # 设置W&B实体
    if training_args.wandb_entity is not None:
        os.environ["WANDB_ENTITY"] = training_args.wandb_entity
    # 设置W&B项目名称
    if training_args.wandb_project is not None:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project
