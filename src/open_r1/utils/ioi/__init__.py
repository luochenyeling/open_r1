"""
IOI模块的初始化文件，定义了模块的公共接口。

主要功能：
1. 导出IOI模块的主要组件
2. 提供便捷的导入方式

主要组件：
1. get_piston_client_from_env: 从环境变量获取Piston客户端
2. get_slurm_piston_endpoints: 从SLURM获取Piston工作节点列表
3. score_subtask: 评分子任务
4. add_includes: 添加必要的头文件
5. SubtaskResult: 子任务结果类
"""

from .piston_client import get_piston_client_from_env, get_slurm_piston_endpoints
from .scoring import SubtaskResult, score_subtask
from .utils import add_includes


# 定义模块的公共接口
__all__ = [
    "get_piston_client_from_env",  # 从环境变量获取Piston客户端
    "get_slurm_piston_endpoints",  # 从SLURM获取Piston工作节点列表
    "score_subtask",               # 评分子任务
    "add_includes",                # 添加必要的头文件
    "SubtaskResult",               # 子任务结果类
]
