# reflectors.py
import json

from src.graph.reflectors.config import AGENT_REFLECTION_CONFIG  # 导入配置
from src.until.init_llm import llm
def  reflect_task_result(state):
    # 1. 获取当前执行智能体
    last_agent = state.get("last_executed_agent")
    # 2. 获取当前任务
    messages = state.get("messages", [])

    # 3. 获取当前智能体的配置
    if not last_agent or last_agent not in AGENT_REFLECTION_CONFIG:
        print(f"❌ 无法获取当前智能体的配置: {last_agent}")
        return {
            "is_qualified": True,
            "problem": "",
            "suggestion": "",
            "need_retry": False
        }

    config = AGENT_REFLECTION_CONFIG[last_agent]
    check_items = "\n".join(f"- {item}" for item in config["check_items"])

    standard = config["standard"]

    last_content = messages[-1].content if messages else ""
    patient_info = state.get("patient_info", {})

    prompt = f"""
你是医疗系统专业反思专家，只做严谨审查。

【当前执行智能体】{last_agent}
【专业标准】{standard}
【检查项】
{check_items}

【患者信息】{patient_info}
【上一步输出】{last_content}

请输出严格的JSON格式，不要多余内容：
{{

    "is_qualified": true / false,
    "problem": "",
    "suggestion": "",
    "need_retry": true/false
}}
"""
    print(f"[DEBUG] 反思提示: {prompt}")
    try:
        response = llm.invoke(prompt)
        print(f"[DEBUG] 反思结果: {response.content}")
        return json.loads(response.content)
    except:
        return {
            "is_qualified": True,
            "problem": "",
            "suggestion": "",
            "need_retry": None
        }