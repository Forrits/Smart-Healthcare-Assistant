# joker_chat.py

from src.graph.state import AgentState
from src.graph.task_filter import get_current_task
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from src.until.init_llm import llm


# ====================== ✅ 修改提示词 ======================


with open("src/prompts/joker.txt", "r", encoding="utf-8") as f:
    JokerPrompt = f.read()
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是闲聊助手，严格执行任务。"),
    ("human", JokerPrompt)
])

chat_chain = prompt | llm

def chat_node(state: AgentState):
    current_task = get_current_task(state, "CHAT")
    if not current_task:
        return {}

    print("\n💬 --- 闲聊节点工作 ---")
    print(f"📋 执行任务: {current_task['description']}")

    # ====================== ✅ 读取反思 ======================
    reflection = state.get("reflection", {})
    problem = reflection.get("problem", "")
    suggestion = reflection.get("suggestion", "")

    # ====================== ✅ 传入反思 ======================
    response = chat_chain.invoke({
        "task_desc": current_task["description"], # 任务描述，还是从任务队列current_task里获取
        "problem": problem,                  # 反思问题，这个是从state里获取
        "suggestion": suggestion              # 反思建议，这个是从state里获取
    })

    # 更新任务
    updated_task_list = []
    for task in state["task_list"]:
        if task["task_id"] == current_task["task_id"]:
            updated_task = task.copy()
            updated_task["status"] = "completed"
            updated_task["just_finished"] = True
            updated_task_list.append(updated_task)
        else:
            updated_task_list.append(task)

    print(f"出结果: {response}")

    return {
        "messages": [response],
        "task_list": updated_task_list,
        "last_executed_agent": "CHAT"
    }