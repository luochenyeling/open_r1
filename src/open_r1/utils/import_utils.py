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
导入工具模块，用于检查和管理依赖包的可用性。

主要功能：
1. 检查e2b包的可用性
2. 提供统一的包可用性检查接口

主要组件：
1. _e2b_available: e2b包的可用性标志
2. is_e2b_available: 检查e2b包是否可用的函数
"""

from transformers.utils.import_utils import _is_package_available


# 使用与transformers.utils.import_utils相同的检查方法
_e2b_available = _is_package_available("e2b")


def is_e2b_available() -> bool:
    """
    检查e2b包是否可用
    
    返回:
        bool: 如果e2b包可用返回True，否则返回False
    """
    return _e2b_available
