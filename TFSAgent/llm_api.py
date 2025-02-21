"""
LLM Interface 模块
-------------------
本模块实现了一个基于 OpenAI API 的 LLM 查询接口。
函数 query_llm 接受一个 prompt 字符串（应另外传入，包含上下文信息），
并构造符合 OpenAI ChatCompletion API 格式的消息列表，调用 OpenAI API 获取回复。

注意：
  - 请确保环境变量中已配置 OPENAI_API_KEY（必填）和 OPENAI_BASE_URL（可选）。
    OPENAI_BASE_URL 默认为 "https://api.openai.com/v1"。
"""

import os
import openai
import logging
from openai import OpenAI
from dotenv import load_dotenv

# 配置日志记录
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.warning("OPENAI_API_KEY is not set. Please configure your OpenAI API key in the environment variables.")

openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

openai.api_key = openai_api_key
client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)

def query_llm(sys_prompt: str,
              user_prompt: str,
              model: str = "gpt-4o-mini",
              temperature: float = 0.8,
              max_tokens: int = 1500) -> str:
    """
    调用 OpenAI ChatCompletion API 获取 LLM 的回复。

    参数:
      sys_prompt (str): 系统消息，用于引导 LLM，通常从环境或配置中读取。
      user_prompt (str): 要发送给 LLM 的用户消息（包含上下文信息）。
      model (str): 使用的模型名称，默认为 "gpt-4o-mini"。
      temperature (float): 控制生成文本多样性的温度参数，默认为 0.5。
      max_tokens (int): 最大返回 token 数量，默认为 1500。

    返回:
      str: LLM 返回的文本内容。
    """
    try:
       
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ]
        logger.debug("Querying LLM with messages: %s", messages)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        content = response.choices[0].message.content.strip()
        logger.debug("LLM response: %s", content)
        return content
    except Exception as e:
        logger.error("Error querying LLM: %s", e)
        return f"Error querying LLM: {e}"


