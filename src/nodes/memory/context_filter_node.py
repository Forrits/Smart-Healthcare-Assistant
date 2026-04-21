# context_filter_node.py
from typing import Dict, Any, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage

def context_filter_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    上下文过滤节点：
    - 过滤掉空消息、格式错误的消息
    - 可选：过滤掉系统提示词以外的敏感内容
    """
    messages: List[BaseMessage] = state.get("messages", [])
    filtered_messages = []

    for msg in messages:
        # 跳过空内容的消息
        if not msg.content or (isinstance(msg.content, str) and msg.content.strip() == ""):
            continue

        # 跳过格式错误的消息（如非预期的类型）
        if not isinstance(msg, (HumanMessage, AIMessage, ToolMessage)):
            continue

        filtered_messages.append(msg)

    print(f"🔍 上下文过滤完成：{len(messages)} → {len(filtered_messages)} 条消息")

    return {
        "messages": filtered_messages
    }