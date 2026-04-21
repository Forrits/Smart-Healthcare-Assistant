#规划路由（跳转规则）

from langgraph.graph import END
from langgraph.prebuilt import ToolNode,tools_condition
# 图像分析节点路由
def should_continue_image(state):
    messages = state["messages"]
    if not messages:
        return END
    last_message = messages[-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "image_analysis_tools"
    return END

# 医生节点路由
def should_continue_doctor(state):
    """
    决定医生节点之后去哪：
    1. 有 tool_calls -> 继续执行工具 (doctor_tools)
    2. 无 tool_calls -> 本轮结束，让用户输入 (END)
    """
    messages = state["messages"]
    if not messages:
        return "supervisor"  # 兜底，如果没消息，回调度中心

    last_message = messages[-1]

    # 1. 如果检测到工具调用 -> 继续走工具节点
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "doctor_tools"

        # 2. 如果没有工具调用 -> 直接结束本轮 (返回 END)
    # 这就是你要的！用户可以继续输入！
    return END

#医疗科普agent节点路由
def should_continue_tutor(state):
    messages = state["messages"]
    if not messages:
        return "supervisor"
    last_message = messages[-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "medical_tutor_tools"
    return "supervisor"


# def should_continue_tutor(state):
#     """
#     科普Agent专用：只允许调用1次工具，强制结束，永不循环
#     """
#     # 🔥 第一步：限制【只能调用1次工具】—— 核心！
#     messages = state["messages"]
#
#     # 统计已经调用过多少次工具
#     tool_call_count = 0
#     for msg in messages:
#         if hasattr(msg, "tool_calls") and msg.tool_calls:
#             tool_call_count += 1
#
#     # 🔥 只要调用过 ≥1 次，直接结束！彻底断循环！
#     if tool_call_count >= 2:
#         return "__end__"
#
#     # 🔥 没调用过 → 走官方工具判断逻辑
#     return tools_condition(state)



# （可选）辩论逻辑路由，也放这
# def should_continue_challenger(state):
#     round_count = state.get("debate_round_count", 0)
#     if round_count < 1:
#         return "doctor"
#     return END