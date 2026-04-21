# memory_load_node.py
from typing import Dict, Any
from langchain_core.messages import SystemMessage

def memory_load_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    记忆加载节点：
    - 加载用户长期记忆/病史信息
    - 插入到上下文最前面，作为系统背景信息
    """
    # 从状态或数据库中加载用户记忆
    patient_info = state.get("patient_info", {})
    if not patient_info:
        print("📝 无用户记忆可加载")
        return {}

    # 格式化记忆信息为系统提示
    memory_text = "用户历史信息：\n"
    for key, value in patient_info.items():
        memory_text += f"- {key}: {value}\n"

    # 生成系统消息，插入到上下文最前面
    memory_message = SystemMessage(content=memory_text)

    print("🧠 用户记忆加载完成")

    # 插入到消息列表最前面（系统消息位置）
    messages = state.get("messages", [])
    messages.insert(0, memory_message)

    return {
        "messages": messages
    }