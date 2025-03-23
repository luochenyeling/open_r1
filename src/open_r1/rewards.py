"""
奖励函数模块，实现了GRPO训练过程中使用的各种奖励函数。

主要包含以下奖励函数：
1. accuracy_reward: 准确性奖励
2. format_reward: 格式奖励
3. tag_count_reward: 标签计数奖励
4. reasoning_steps_reward: 推理步骤奖励
5. len_reward: 长度奖励
6. cosine_scaled_reward: 余弦缩放奖励
7. repetition_penalty_reward: 重复惩罚奖励
8. code_reward: 代码相关奖励
"""

import asyncio
import json
import math
import re
from functools import partial, update_wrapper
from typing import Callable, Dict

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

from .utils import is_e2b_available
from .utils.ioi import SubtaskResult, add_includes, get_piston_client_from_env, score_subtask


# 检查是否可用e2b环境
if is_e2b_available():
    from dotenv import load_dotenv
    from e2b_code_interpreter import AsyncSandbox

    load_dotenv()
else:
    AsyncSandbox = None


def accuracy_reward(completions, solution, **kwargs):
    """
    准确性奖励函数，检查模型输出是否与标准答案相同。
    
    参数:
        completions: 模型生成的输出列表
        solution: 标准答案列表
        **kwargs: 其他参数
    
    返回:
        list[float]: 每个输出的奖励值列表，1.0表示完全正确，0.0表示错误
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        # 解析标准答案中的LaTeX公式
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) != 0:
            # 解析模型输出中的LaTeX公式
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # 确保优先尝试boxed匹配
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # 验证答案是否正确
            try:
                reward = float(verify(answer_parsed, gold_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                reward = 0.0
        else:
            # 如果标准答案无法解析，跳过该样本
            reward = 1.0
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)

    return rewards


def format_reward(completions, **kwargs):
    """
    格式奖励函数，检查输出是否包含正确的标签格式。
    
    参数:
        completions: 模型生成的输出列表
        **kwargs: 其他参数
    
    返回:
        list[float]: 每个输出的奖励值列表，1.0表示格式正确，0.0表示格式错误
    """
    # 检查是否包含<think>和<answer>标签
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def tag_count_reward(completions, **kwargs) -> list[float]:
    """
    标签计数奖励函数，检查输出中标签的数量是否正确。
    
    参数:
        completions: 模型生成的输出列表
        **kwargs: 其他参数
    
    返回:
        list[float]: 每个输出的奖励值列表，根据标签数量计算部分奖励
    """
    def count_tags(text: str) -> float:
        """
        计算文本中标签的数量，每个正确位置的标签得0.25分
        
        参数:
            text: 输入文本
            
        返回:
            float: 标签得分，最高1.0分
        """
        count = 0.0
        if text.count("<think>\n") == 1:
            count += 0.25
        if text.count("\n</think>\n") == 1:
            count += 0.25
        if text.count("\n<answer>\n") == 1:
            count += 0.25
        if text.count("\n</answer>") == 1:
            count += 0.25
        return count

    contents = [completion[0]["content"] for completion in completions]
    return [count_tags(c) for c in contents]


def reasoning_steps_reward(completions, **kwargs):
    """
    推理步骤奖励函数，检查输出是否包含清晰的推理步骤。
    
    参数:
        completions: 模型生成的输出列表
        **kwargs: 其他参数
    
    返回:
        list[float]: 每个输出的奖励值列表，根据推理步骤数量计算部分奖励
    """
    # 匹配推理步骤的模式
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # 鼓励至少3个推理步骤
    return [min(1.0, count / 3) for count in matches]


def len_reward(completions: list[Dict[str, str]], solution: list[str], **kwargs) -> float:
    """
    长度奖励函数，根据输出长度计算奖励，避免过度思考。
    
    参数:
        completions: 模型生成的输出列表
        solution: 标准答案列表
        **kwargs: 其他参数
    
    返回:
        list[float]: 每个输出的奖励值列表，根据长度和正确性计算奖励
    """
    contents = [completion[0]["content"] for completion in completions]

    # 检查答案正确性
    correctness = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            # 跳过无法解析的样本
            correctness.append(True)
            print("Failed to parse gold solution: ", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

    # 计算长度
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # 如果所有响应长度相同，返回零奖励
    if max_len == min_len:
        return [0.0] * len(completions)

    # 计算奖励
    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards


def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    """
    获取余弦缩放奖励函数，根据输出长度使用余弦函数计算奖励。
    
    参数:
        min_value_wrong: 错误答案的最小奖励值
        max_value_wrong: 错误答案的最大奖励值
        min_value_correct: 正确答案的最小奖励值
        max_value_correct: 正确答案的最大奖励值
        max_len: 最大长度阈值
    
    返回:
        Callable: 余弦缩放奖励函数
    """
    def cosine_scaled_reward(completions, solution, **kwargs):
        """
        余弦缩放奖励函数，根据输出长度使用余弦函数计算奖励。
        
        参数:
            completions: 模型生成的输出列表
            solution: 标准答案列表
            **kwargs: 其他参数
            
        返回:
            list[float]: 每个输出的奖励值列表
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            # 解析标准答案
            gold_parsed = parse(sol, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) == 0:
                rewards.append(1.0)  # 跳过无法解析的样本
                print("Failed to parse gold solution: ", sol)
                continue

            # 解析模型输出
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            # 检查答案正确性
            is_correct = verify(answer_parsed, gold_parsed)
            gen_len = len(content)

            # 使用余弦函数计算长度奖励
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)
            
            # 根据正确性计算最终奖励
            if is_correct:
                reward = min_value_correct + (max_value_correct - min_value_correct) * (1 - cosine) / 2
            else:
                reward = min_value_wrong + (max_value_wrong - min_value_wrong) * (1 - cosine) / 2

            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward


def repetition_penalty_reward(completions, n_grams: int = 3, max_penalty: float = -1.0, **kwargs):
    """
    重复惩罚奖励函数，对重复的n-gram进行惩罚。
    
    参数:
        completions: 模型生成的输出列表
        n_grams: n-gram的大小
        max_penalty: 最大惩罚值
        **kwargs: 其他参数
    
    返回:
        list[float]: 每个输出的奖励值列表
    """
    def get_ngrams(text: str, n: int) -> list[str]:
        """
        获取文本中的所有n-gram
        
        参数:
            text: 输入文本
            n: n-gram的大小
            
        返回:
            list[str]: n-gram列表
        """
        words = text.split()
        return [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]

    def count_repetitions(text: str, n: int) -> float:
        """
        计算文本中重复n-gram的数量
        
        参数:
            text: 输入文本
            n: n-gram的大小
            
        返回:
            float: 重复n-gram的数量
        """
        ngrams = get_ngrams(text, n)
        if not ngrams:
            return 0.0
        ngram_counts = {}
        for ngram in ngrams:
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
        return sum(count - 1 for count in ngram_counts.values())

    contents = [completion[0]["content"] for completion in completions]
    max_repetitions = max(count_repetitions(content, n_grams) for content in contents)
    
    if max_repetitions == 0:
        return [0.0] * len(completions)
        
    rewards = []
    for content in contents:
        repetitions = count_repetitions(content, n_grams)
        reward = max_penalty * (repetitions / max_repetitions)
        rewards.append(float(reward))

    return rewards


def code_reward(completions, solution, language: str = "python", **kwargs):
    """
    代码奖励函数，评估代码的正确性和格式。
    
    参数:
        completions: 模型生成的输出列表
        solution: 标准答案列表
        language: 代码语言
        **kwargs: 其他参数
    
    返回:
        list[float]: 每个输出的奖励值列表
    """
    if not is_e2b_available():
        print("E2B not available, skipping code reward")
        return [0.0] * len(completions)

    async def evaluate_code(content: str, sol: str) -> float:
        """
        异步评估代码
        
        参数:
            content: 模型生成的代码
            sol: 标准答案
            
        返回:
            float: 代码评估得分
        """
        try:
            # 创建代码执行环境
            async with AsyncSandbox() as sandbox:
                # 添加必要的导入
                content = add_includes(content, language)
                
                # 执行代码
                result = await sandbox.run(content, language)
                
                # 检查执行结果
                if result.error:
                    return 0.0
                    
                # 比较输出结果
                if result.stdout.strip() == sol.strip():
                    return 1.0
                return 0.0
        except Exception as e:
            print(f"Error evaluating code: {e}")
            return 0.0

    # 异步评估所有代码
    contents = [completion[0]["content"] for completion in completions]
    loop = asyncio.get_event_loop()
    tasks = [evaluate_code(content, sol) for content, sol in zip(contents, solution)]
    rewards = loop.run_until_complete(asyncio.gather(*tasks))

    return rewards


def code_format_reward(completions, language: str = "python", **kwargs):
    """
    代码格式奖励函数，评估代码的格式规范。
    
    参数:
        completions: 模型生成的输出列表
        language: 代码语言
        **kwargs: 其他参数
    
    返回:
        list[float]: 每个输出的奖励值列表
    """
    if not is_e2b_available():
        print("E2B not available, skipping code format reward")
        return [0.0] * len(completions)

    async def evaluate_format(content: str) -> float:
        """
        异步评估代码格式
        
        参数:
            content: 模型生成的代码
            
        返回:
            float: 格式评估得分
        """
        try:
            # 创建代码执行环境
            async with AsyncSandbox() as sandbox:
                # 添加必要的导入
                content = add_includes(content, language)
                
                # 执行代码
                result = await sandbox.run(content, language)
                
                # 检查执行结果
                if result.error:
                    return 0.0
                    
                # 检查格式规范
                if language == "python":
                    # 使用black检查Python代码格式
                    result = await sandbox.run("pip install black", "bash")
                    if result.error:
                        return 0.0
                    result = await sandbox.run(f"black --check {content}", "bash")
                    return 1.0 if not result.error else 0.0
                else:
                    # 其他语言暂不支持格式检查
                    return 1.0
        except Exception as e:
            print(f"Error evaluating code format: {e}")
            return 0.0

    # 异步评估所有代码格式
    contents = [completion[0]["content"] for completion in completions]
    loop = asyncio.get_event_loop()
    tasks = [evaluate_format(content) for content in contents]
    rewards = loop.run_until_complete(asyncio.gather(*tasks))

    return rewards


def get_reward_funcs(script_args) -> list[Callable]:
    REWARD_FUNCS_REGISTRY = {
        "accuracy": accuracy_reward,
        "format": format_reward,
        "reasoning_steps": reasoning_steps_reward,
        "cosine": get_cosine_scaled_reward(
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
        "repetition_penalty": repetition_penalty_reward,
        "length": len_reward,
        "code": code_reward,
        "code_format": code_format_reward,
        "tag_count": tag_count_reward,
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    return reward_funcs
