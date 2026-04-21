# src/utils/init_llm.py
import os
import logging
from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import openai

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------
# 1. 单例模式：全局共享 LLM 实例
# --------------------------
class LLMManager:
    _instance: Optional["LLMManager"] = None
    _models: Dict[str, ChatOpenAI] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((openai.APIError, openai.Timeout)),
        reraise=True
    )
    def _create_chat_openai(
        self,
        model_name: str,
        api_key: str,
        base_url: str,
        temperature: float = 0.3,
        timeout: int = 30
    ) -> ChatOpenAI:
        """带重试机制的 ChatOpenAI 实例创建"""
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            request_timeout=timeout
        )

    def get_llm(
        self,
        model_key: str = "deepseek",
        temperature: Optional[float] = None
    ) -> ChatOpenAI:
        """
        获取全局 LLM 实例
        :param model_key: 模型标识，支持 "deepseek", "siliconflow"
        :param temperature: 临时覆盖温度
        """
        # 如果实例不存在则创建
        if model_key not in self._models:
            if model_key == "deepseek":
                self._models[model_key] = self._create_chat_openai(
                    model_name="deepseek-chat",
                    api_key=os.getenv("DEEPSEEK_API_KEY"),
                    base_url="https://api.deepseek.com/v1",
                    temperature=0.3
                )
            elif model_key == "siliconflow":
                self._models[model_key] = self._create_chat_openai(
                    model_name="deepseek-vl2",  # 示例：多模态模型
                    api_key=os.getenv("SILICONFLOW_API_KEY"),
                    base_url="https://api.siliconflow.cn/v1",
                    temperature=0.1
                )
            else:
                raise ValueError(f"不支持的模型: {model_key}")

        # 支持临时修改 temperature
        llm = self._models[model_key]
        if temperature is not None:
            llm.temperature = temperature
        return llm

    def get_for_task(self, task_type: str) -> ChatOpenAI:
        """根据任务类型获取预设配置的 LLM"""
        config_map = {
            "diagnosis": ("deepseek", 0.1),    # 问诊/诊断：低温度，严谨
            "chat": ("deepseek", 0.7),         # 闲聊：高温度，自然
            "multimodal": ("siliconflow", 0.1) # 影像分析：多模态+低温度
        }
        model_key, temp = config_map.get(task_type, ("deepseek", 0.3))
        return self.get_llm(model_key, temperature=temp)


# 全局单例导出
llm_manager = LLMManager()

# 便捷导出，兼容你现有代码
def get_deepseek_llm(temperature: float = 0.3) -> ChatOpenAI:
    return llm_manager.get_llm("deepseek", temperature=temperature)

def get_multimodal_llm() -> ChatOpenAI:
    return llm_manager.get_llm("siliconflow")
