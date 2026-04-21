# src/nodes/challenger_node.py
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.graph.state import AgentState, DebateTurn
from langchain_openai import ChatOpenAI
import os

# 挑战者不需要工具，因为它不执行操作
challenger_llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0.8  # 温度调高，让它思维更发散，更容易发现问题
)

# --- 专门为挑战者设计的 Prompt ---
CHALLENGER_PROMPT = """
你是一位著名的医疗审查专家，以喜欢挑刺和发现盲点而闻名。

你的任务：
1. 阅读主治医生的初步诊断和患者的病历。
2. **寻找漏洞**：主治医生是否忽略了某个症状？是否有其他可能性（鉴别诊断）？
3. **寻找风险**：这个诊断方案是否会导致误诊？是否存在药物禁忌？
4. **强制提问**：即使你觉得诊断是对的，也必须提出至少一个刁钻的问题。
5. 你的输出只能是质疑和问题，不需要给出治疗方案。
"""


def challenger_node(state: AgentState):
    print("\n⚖️ --- 质疑者节点开始工作 ---")

    # 获取患者信息和辩论历史
    patient_info = state.get("patient_info", {})
    debate_history = state.get("debate_history", [])

    # 获取最新的医生诊断（从消息历史中提取）
    messages = state.get("messages", [])
    last_doctor_diagnosis = ""

    # 从消息历史中找到最近的医生回复
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            last_doctor_diagnosis = msg.content
            break

    # 构建消息：给挑战者看主治医生的方案
    context_text = f"患者信息：{patient_info}\n\n主治医生的诊断：{last_doctor_diagnosis}"

    messages_for_challenger = [
        SystemMessage(content=CHALLENGER_PROMPT),
        HumanMessage(content=context_text)
    ]

    response = challenger_llm.invoke(messages_for_challenger)

    # 创建新的辩论记录
    new_debate_turn: DebateTurn = {
        "role": "challenger",
        "content": response.content,
        "diagnosis_proposal": None
    }

    # 更新辩论历史和轮次计数
    current_debate_history = state.get("debate_history", []) or []
    updated_debate_history = current_debate_history + [new_debate_turn]

    return {
        "debate_history": updated_debate_history,
        "debate_round_count": state.get("debate_round_count", 0) + 1
    }
