"""
评分模块，实现了IOI（国际信息学奥林匹克）题目的代码评分功能。

主要功能：
1. 执行和评分单个测试用例
2. 计算子任务的得分和状态
3. 管理测试用例的缓存和批处理
4. 提供详细的评分反馈

主要组件：
1. TestResult: 单个测试用例的执行结果
2. SubtaskResult: 包含多个测试用例的子任务结果
3. score_single_test_case: 评分单个测试用例
4. score_subtask: 评分整个子任务
5. score_subtasks: 评分多个子任务
"""

import asyncio
from dataclasses import asdict, dataclass, field
from typing import Union

from .piston_client import PistonClient
from .utils import batched, load_ioi_tests


@dataclass
class TestResult:
    """
    表示单个测试用例的执行结果
    
    属性:
        test_name: 测试用例名称
        score: 测试得分（0.0到1.0）
        status: 测试结果状态码（如'AC'、'WA'、'TLE'等）
        feedback: 来自评判系统的详细反馈或错误信息
    """

    test_name: str
    score: float = 0.0
    status: str = "SKIPPED"
    feedback: str = None


@dataclass
class SubtaskResult:
    """
    表示包含多个测试用例的子任务结果
    
    属性:
        problem: 题目标识符
        subtask: 子任务标识符
        points: 该子任务的最大分值
        score_precision: 分数舍入的小数位数
        test_results: 各个测试用例的结果列表
    """

    problem: str = None
    subtask: str = None

    points: float = 0.0
    score_precision: int = 2

    test_results: list[TestResult] = field(default_factory=list)

    @property
    def status(self):
        """
        根据测试结果中最差的状态确定子任务的总体状态
        
        状态优先级从差到好排序
        
        返回:
            str: 优先级最高的状态（最低值）
        """
        status_prios = {"CE": -1, "RE": 0, "WA": 1, "MLE": 2, "TLE": 3, "PA": 4, "AC": 5, "SKIPPED": 999}
        return min([x.status for x in self.test_results], key=lambda x: status_prios[x])

    @property
    def score(self):
        """
        计算子任务的原始分数，取所有测试结果中的最低分
        
        返回:
            float: 舍入后的最低分数
        """
        return (
            0
            if not self.test_results
            else round(min([test_result.score for test_result in self.test_results]), self.score_precision)
        )

    @property
    def weighted_score(self):
        """
        计算加权分数，将原始分数乘以可用分值
        
        返回:
            float: 舍入后的加权分数
        """
        return (
            0
            if not self.test_results
            else round(
                min([test_result.score for test_result in self.test_results]) * self.points, self.score_precision
            )
        )

    def to_dict(self):
        """
        将SubtaskResult转换为字典表示
        
        返回:
            dict: 包含所有子任务结果数据的字典
        """
        return {
            "problem": self.problem,
            "subtask": self.subtask,
            "score": self.score,
            "weighted_score": self.weighted_score,
            "points": self.points,
            "score_precision": self.score_precision,
            "status": self.status,
            "test_results": [asdict(test_result) for test_result in self.test_results],
        }


def _extract_single_status(score: float, feedback: str) -> str:
    """
    根据分数和反馈信息确定状态码
    
    参数:
        score: 数值分数（0.0到1.0）
        feedback: 执行反馈信息
        
    返回:
        str: 状态码（'CE'、'MLE'、'TLE'、'WA'、'RE'、'AC'或'PA'）
    """
    if score == 0.0:
        if "Compilation error" in feedback:
            return "CE"
        elif "Memory limit exceeded" in feedback:
            return "MLE"
        elif "Time limit exceeded" in feedback:
            return "TLE"
        elif "Output isn't correct" in feedback:
            return "WA"
        else:
            return "RE"
    elif score == 1.0:
        return "AC"
    else:
        return "PA"


async def score_single_test_case(
    client: PistonClient, subtask: dict, test_name: str, test_input: str, test_output: str, submission: str
) -> TestResult:
    """
    对单个测试用例进行评分
    
    运行提交的代码并比较输出结果
    
    参数:
        client: 用于执行代码的PistonClient实例
        subtask: 包含子任务配置的字典
        test_name: 测试用例名称
        test_input: 测试用例的输入数据
        test_output: 测试用例的期望输出
        submission: 提交的源代码
        
    返回:
        TestResult: 测试用例的执行结果
    """
    # 运行提交的代码进行测试
    score, feedback = await run_submission(client, subtask, test_input, submission, test_output)
    score = float(score)

    return TestResult(
        test_name=test_name, score=score, status=_extract_single_status(score, feedback), feedback=feedback
    )


async def score_subtask(
    client: PistonClient,
    subtask: dict,
    submission: str,
    test_case_run_cache: Union[dict, None] = None,
    test_batch_size: int = 1,
) -> SubtaskResult:
    """
    对子任务中的所有测试用例进行评分
    
    参数:
        client: 用于执行代码的PistonClient实例
        subtask: 包含子任务配置的字典
        submission: 提交的源代码
        test_case_run_cache: 可选，之前运行的测试用例缓存
        test_batch_size: 并行评估的测试用例数量，如果任何一个失败（得分为0）则停止评估；否则继续评估下一批测试用例。-1表示并行评估所有测试用例
        
    返回:
        SubtaskResult: 子任务的评估结果
    """
    subtask_result = SubtaskResult(
        problem=subtask["id"],
        subtask=subtask["subtask"],
        points=subtask["score"],
        score_precision=subtask["score_precision"],
        test_results=[],
    )

    # 未缓存的测试用例
    tests_to_run = [
        (ti, test_name)
        for ti, test_name in enumerate(subtask["test_names"])
        if test_case_run_cache is None or test_name not in test_case_run_cache
    ]

    # 用缓存的结果或空的（SKIPPED）TestResult对象初始化测试结果
    subtask_result.test_results = [
        test_case_run_cache[test_name]
        if test_case_run_cache is not None and test_name in test_case_run_cache
        else TestResult(test_name=test_name)
        for test_name in subtask["test_names"]
    ]

    # 跳过没有提取到代码的提交
    # 不需要做任何事，因为我们有一个失败的缓存结果
    if not submission or any(
        test_result.status != "SKIPPED" and test_result.score == 0.0 for test_result in subtask_result.test_results
    ):
        return subtask_result

    if "test_cases" in subtask:
        test_cases = subtask["test_cases"]
        if isinstance(subtask["test_cases"], list):
            test_cases = {test_name: test for test_name, test in zip(subtask["test_names"], subtask["test_cases"])}
    else:
        test_cases = load_ioi_tests(subtask["year"], subtask["id"])

    # 运行一批测试用例，检查是否有任何一个失败（得分为0）：如果有则停止评估；否则继续评估下一批测试用例
    for test_batch_to_run in batched(tests_to_run, test_batch_size):
        results = await asyncio.gather(
            *[
                asyncio.create_task(
                    score_single_test_case(
                        client, subtask, test_name, test_cases[test_name][0], test_cases[test_name][1], submission
                    )
                )
                for _, test_name in test_batch_to_run
            ]
        )
        for (ti, test_name), test_result in zip(test_batch_to_run, results):
            if test_case_run_cache is not None:
                test_case_run_cache[test_name] = test_result
            subtask_result.test_results[ti] = test_result

        # 如果有失败的测试用例则提前停止
        if any(test_result.score == 0.0 for test_result in results):
            break

    return subtask_result


async def score_subtasks(
    client: PistonClient, subtasks: list[dict], submission: str, skip_mode: bool = True
) -> list[SubtaskResult]:
    """
    对提交的代码进行多个子任务的评分
    
    参数:
        client: 用于执行代码的PistonClient实例
        subtasks: 子任务配置列表
        submission: 提交的源代码
        skip_mode: 是否启用跳过模式，默认为True
        
    返回:
        list[SubtaskResult]: 子任务评分结果列表
    """
    # 初始化测试用例运行缓存
    test_case_run_cache = {}
    subtask_results = []

    # 对每个子任务进行评分
    for subtask in subtasks:
        subtask_result = await score_subtask(client, subtask, submission, test_case_run_cache)
        subtask_results.append(subtask_result)

        # 在跳过模式下，如果当前子任务失败则停止后续子任务的评分
        if skip_mode and subtask_result.score == 0.0:
            break

    return subtask_results


async def run_submission(
    client: PistonClient, problem: dict, test_input: str, submission: str, test_output: str | None = None
) -> tuple[str, str]:
    """
    运行提交的代码并获取执行结果
    
    参数:
        client: 用于执行代码的PistonClient实例
        problem: 题目配置字典
        test_input: 测试用例的输入数据
        submission: 提交的源代码
        test_output: 可选的测试用例期望输出
        
    返回:
        tuple[str, str]: (分数, 反馈信息)
    """
    # 准备执行数据
    data = {
        "files": [
            {"name": problem["filename"], "content": submission},
            {"name": "input.txt", "content": test_input},
        ],
        "stdin": test_input,
        "command": problem["command"],
        "compile_timeout": problem["compile_timeout"],
        "run_timeout": problem["run_timeout"],
        "memory_limit": problem["memory_limit"],
    }

    # 如果有期望输出，添加到执行数据中
    if test_output is not None:
        data["files"].append({"name": "output.txt", "content": test_output})

    # 执行代码并返回结果
    return await client.execute(data)
