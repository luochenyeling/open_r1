"""
模型工具模块，提供了模型相关的工具函数。

主要功能：
1. 获取和配置模型的分词器
2. 设置默认的对话模板

主要组件：
1. DEFAULT_CHAT_TEMPLATE: 默认的对话模板
2. get_tokenizer: 获取和配置模型分词器的函数
"""

from transformers import AutoTokenizer, PreTrainedTokenizer

from trl import ModelConfig

from ..configs import GRPOConfig, SFTConfig


# 默认的对话模板，支持用户、系统和助手角色的消息
DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"


def get_tokenizer(
    model_args: ModelConfig, training_args: SFTConfig | GRPOConfig, auto_set_chat_template: bool = True
) -> PreTrainedTokenizer:
    """
    获取和配置模型的分词器
    
    参数:
        model_args: 模型配置参数
        training_args: 训练配置参数
        auto_set_chat_template: 是否自动设置对话模板，默认为True
        
    返回:
        PreTrainedTokenizer: 配置好的分词器对象
    """
    # 从预训练模型加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    # 设置对话模板
    if training_args.chat_template is not None:
        tokenizer.chat_template = training_args.chat_template
    elif auto_set_chat_template and tokenizer.get_chat_template() is None:
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    return tokenizer
