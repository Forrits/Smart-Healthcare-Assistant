# context_trim_node.py
from typing import Dict, Any, List
from langchain_core.messages import BaseMessage

def context_trim_node(state: Dict[str, Any], max_messages: int = 10, max_length: int = 8000) -> Dict[str, Any]:
    """
    上下文裁剪节点：
    - 按消息数量限制裁剪
    - 按总字符数限制裁剪
    - 优先保留最新消息
    """
    messages: List[BaseMessage] = state.get("messages", [])

    # 1. 先按消息数量裁剪
    if len(messages) > max_messages:
        messages = messages[-max_messages:]

    # 2. 再按总字符数裁剪（简单实现，实际可用 tiktoken 优化）
    total_chars = sum(len(msg.content) for msg in messages if isinstance(msg.content, str))
    while total_chars > max_length and len(messages) > 1:
        # 从最旧的消息开始删除
        removed = messages.pop(0)
        if isinstance(removed.content, str):
            total_chars -= len(removed.content)

    print(f"✂️ 上下文裁剪完成：最终保留 {len(messages)} 条消息，字符数约 {total_chars}")

    return {
        "messages": messages
    }