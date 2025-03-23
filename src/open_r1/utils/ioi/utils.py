"""
IOI工具模块，提供了处理IOI（国际信息学奥林匹克）题目相关的工具函数。

主要功能：
1. 修复常见的编译错误
2. 加载IOI测试用例
3. 批处理数据

主要组件：
1. add_includes: 添加必要的头文件和命名空间声明
2. load_ioi_tests_for_year: 加载指定年份的IOI测试用例
3. load_ioi_tests: 加载指定年份和题目的测试用例
4. batched: 将数据分批处理
"""

from collections import defaultdict
from functools import lru_cache
from itertools import islice

from datasets import load_dataset


def add_includes(code: str, problem_id: str) -> str:
    """
    修复IOI题目中常见的编译错误
    
    添加必要的头文件和命名空间声明，包括：
    1. bits/stdc++.h（包含大多数常用函数）
    2. 题目特定的头文件
    3. using namespace std声明（如果代码中没有使用std::）
    
    参数:
        code: 源代码
        problem_id: 题目标识符
        
    返回:
        str: 添加了必要头文件的源代码
    """
    if not code:
        return code
    # 包含大多数常用函数
    code_header = "#include <bits/stdc++.h>\n"
    # 包含题目头文件
    problem_header_include = f'#include "{problem_id}.h"'
    if problem_header_include not in code:
        code_header += problem_header_include + "\n"
    # 使用std命名空间，因为模型经常忘记std::
    if "using namespace std;" not in code and "std::" not in code:
        code_header += "\nusing namespace std;\n\n"
    return code_header + code


@lru_cache
def load_ioi_tests_for_year(year: int) -> dict[str, dict[str, tuple[str, str]]]:
    """
    加载指定年份的IOI测试用例
    
    从HuggingFace数据集加载测试用例，并按题目ID和测试名称组织
    
    参数:
        year: IOI年份
        
    返回:
        dict: 嵌套字典，外层键为题目ID，内层键为测试名称，值为(输入, 输出)元组
    """
    tests_dataset = load_dataset("open-r1/ioi-test-cases", name=f"{year}", split="train")
    test_cases = defaultdict(dict)
    for test_case in tests_dataset:
        test_cases[test_case["problem_id"]][test_case["test_name"]] = test_case["test_input"], test_case["test_output"]
    return test_cases


def load_ioi_tests(year: int, problem_id: str) -> dict[str, tuple[str, str]]:
    """
    加载指定年份和题目的IOI测试用例
    
    参数:
        year: IOI年份
        problem_id: 题目标识符
        
    返回:
        dict: 字典，键为测试名称，值为(输入, 输出)元组
    """
    return load_ioi_tests_for_year(year)[problem_id]


def batched(iterable, n):
    """
    将数据分批处理成指定长度的列表
    
    最后一个批次可能更短
    
    参数:
        iterable: 可迭代对象
        n: 每批次的长度
        
    返回:
        generator: 生成批次列表的生成器
        
    示例:
        >>> list(batched('ABCDEFG', 3))
        [['A', 'B', 'C'], ['D', 'E', 'F'], ['G']]
    """
    if n < 1:
        return iterable
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch
