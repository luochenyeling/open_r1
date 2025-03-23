"""
评估模块，实现了模型性能评估和基准测试功能。

主要功能：
1. 注册和管理LightEval评估任务
2. 配置SLURM环境以在训练作业中启动vLLM
3. 运行基准测试任务
4. 管理GPU资源分配

主要组件：
1. register_lighteval_task: 注册LightEval任务配置
2. run_lighteval_job: 运行单个LightEval评估任务
3. run_benchmark_jobs: 运行多个基准测试任务
"""

import subprocess
from typing import TYPE_CHECKING, Dict, Union

from .hub import get_gpu_count_for_vllm, get_param_count_from_repo_id


if TYPE_CHECKING:
    from trl import GRPOConfig, SFTConfig, ModelConfig

import os


# 为在SLURM训练作业中启动vLLM需要特殊的环境设置
# - 参考代码: https://github.com/huggingface/brrr/blob/c55ba3505686d690de24c7ace6487a5c1426c0fd/brrr/lighteval/one_job_runner.py#L105
# - Slack讨论: https://huggingface.slack.com/archives/C043JTYE1MJ/p1726566494958269
user_home_directory = os.path.expanduser("~")
VLLM_SLURM_PREFIX = [
    "env",
    "-i",
    "bash",
    "-c",
    f"for f in /etc/profile.d/*.sh; do source $f; done; export HOME={user_home_directory}; sbatch ",
]


def register_lighteval_task(
    configs: Dict[str, str], eval_suite: str, task_name: str, task_list: str, num_fewshot: int = 0
):
    """
    注册LightEval任务配置
    
    核心任务可以从以下表格添加：
    https://github.com/huggingface/lighteval/blob/main/src/lighteval/tasks/tasks_table.jsonl
    
    需要自定义指标/脚本的任务应存储在scripts/evaluation/extended_lighteval_tasks中
    
    参数:
        configs: 存储任务配置的字典
        eval_suite: 评估套件名称
        task_name: 任务名称
        task_list: 逗号分隔的任务列表，格式为"extended|{task_name}|{num_fewshot}|0"或"lighteval|{task_name}|{num_fewshot}|0"
        num_fewshot: few-shot示例数量，默认为0
    """
    # 将任务列表格式化为lighteval格式
    task_list = ",".join(f"{eval_suite}|{task}|{num_fewshot}|0" for task in task_list.split(","))
    configs[task_name] = task_list


# 初始化LightEval任务字典
LIGHTEVAL_TASKS = {}

# 注册预定义的评估任务
register_lighteval_task(LIGHTEVAL_TASKS, "custom", "math_500", "math_500", 0)
register_lighteval_task(LIGHTEVAL_TASKS, "custom", "aime24", "aime24", 0)
register_lighteval_task(LIGHTEVAL_TASKS, "custom", "aime25", "aime25", 0)
register_lighteval_task(LIGHTEVAL_TASKS, "custom", "gpqa", "gpqa:diamond", 0)
register_lighteval_task(LIGHTEVAL_TASKS, "extended", "lcb", "lcb:codegeneration", 0)
register_lighteval_task(LIGHTEVAL_TASKS, "extended", "lcb_v4", "lcb:codegeneration_v4", 0)


def get_lighteval_tasks():
    """
    获取所有已注册的LightEval任务列表
    
    返回:
        list: 任务名称列表
    """
    return list(LIGHTEVAL_TASKS.keys())


# 获取支持的基准测试列表
SUPPORTED_BENCHMARKS = get_lighteval_tasks()


def run_lighteval_job(
    benchmark: str, training_args: Union["SFTConfig", "GRPOConfig"], model_args: "ModelConfig"
) -> None:
    """
    运行单个LightEval评估任务
    
    参数:
        benchmark: 基准测试名称
        training_args: 训练参数配置
        model_args: 模型参数配置
    """
    task_list = LIGHTEVAL_TASKS[benchmark]
    model_name = training_args.hub_model_id
    model_revision = training_args.hub_model_revision
    
    # 对于参数大于等于30B的模型或运行MATH基准测试的模型，需要在GPU间分片以避免OOM
    num_gpus = get_gpu_count_for_vllm(model_name, model_revision)
    if get_param_count_from_repo_id(model_name) >= 30_000_000_000:
        tensor_parallel = True
    else:
        num_gpus = 8
        tensor_parallel = False

    # 构建SLURM命令
    cmd = VLLM_SLURM_PREFIX.copy()
    cmd_args = [
        f"--gres=gpu:{num_gpus}",
        f"--job-name=or1_{benchmark}_{model_name.split('/')[-1]}_{model_revision}",
        "slurm/evaluate.slurm",
        benchmark,
        f'"{task_list}"',
        model_name,
        model_revision,
        f"{tensor_parallel}",
        f"{model_args.trust_remote_code}",
    ]
    if training_args.system_prompt is not None:
        cmd_args.append(f"--system_prompt={training_args.system_prompt}")
    cmd[-1] += " " + " ".join(cmd_args)
    subprocess.run(cmd, check=True)


def run_benchmark_jobs(training_args: Union["SFTConfig", "GRPOConfig"], model_args: "ModelConfig") -> None:
    """
    运行多个基准测试任务
    
    参数:
        training_args: 训练参数配置
        model_args: 模型参数配置
        
    异常:
        ValueError: 当指定的基准测试未知时抛出
    """
    benchmarks = training_args.benchmarks
    if len(benchmarks) == 1 and benchmarks[0] == "all":
        benchmarks = get_lighteval_tasks()
        # 在所有支持的基准测试上评估。之后可能会添加一个'chat'选项，
        # 仅评估'ifeval'和'mt_bench'等

    for benchmark in benchmarks:
        print(f"Launching benchmark `{benchmark}`")
        if benchmark in get_lighteval_tasks():
            run_lighteval_job(benchmark, training_args, model_args)
        else:
            raise ValueError(f"Unknown benchmark {benchmark}")
