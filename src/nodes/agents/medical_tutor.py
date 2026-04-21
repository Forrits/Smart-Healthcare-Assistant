from src.graph.task_filter import get_current_task
from langchain_core.messages import AIMessage,ToolMessage,HumanMessage
import json
from langchain_core.prompts import ChatPromptTemplate
from src.tools.medical_tutor_tools import medical_tutor_tools
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key="sk-31f0754bbffa4c76bbf0d82d790a5ba3",
    base_url="https://api.deepseek.com/v1",
    temperature=0.1  # 闲聊可以温度高一点
)

# 绑定工具
llm_with_tools = llm.bind_tools(
    medical_tutor_tools,
    tool_choice="required",
    strict=True
)

# ---------------------------------------------------------
# 2. 提示词 + 链（我给你补齐 tutor_chain）
# ---------------------------------------------------------
TUTOR_PROMPT = """
你是专业、亲切、严谨的医学科普讲师，只负责用通俗语言解释医学常识。
根据问题，调用local_knowledge_search工具进行检索。
"""

tutor_prompt = ChatPromptTemplate.from_messages([
    ("system", TUTOR_PROMPT),
    ("placeholder", "{messages}")
])
tutor_chain = tutor_prompt | llm_with_tools
def medical_tutor_node(state):
    # ==============================================
    # 🔥 【关键改动 1】只拿分配给 TUTOR 的任务，不看完整消息
    # ==============================================
    current_task = get_current_task(state, "TUTOR")
    if not current_task:
        return {}  # 没有自己的任务 → 不执行

    task_desc = current_task["description"]
    print(f"\n📚 科普节点执行任务：{task_desc}")

    messages = state["messages"]
    # ==============================================
    # 1. 检查是否已有工具结果
    # ==============================================
    has_tool_result = any(isinstance(m, ToolMessage) for m in messages)

    if has_tool_result:
        tool_msg = next(m for m in reversed(messages) if isinstance(m, ToolMessage))
        data = json.loads(tool_msg.content)
        knowledge = data["content"]

        print(f"📚 获取工具结果：{knowledge}")
        # ==============================================
        # 🔥 【关键改动 2】只基于当前任务回答，不看完整用户问题
        # ==============================================
        rag_prompt = f"""
                    你是专业医学科普助手。
                    
                    任务：{task_desc}
                    参考资料：{knowledge}
                    
                    如果参考资料无法回答问题，请直接返回 "我不知道"。
                    输出在100字以内，请用中文回答。
                    """
        answer = llm.invoke(rag_prompt)

        # ==============================================
        # 🔥 【关键改动3】任务完成 → 标记状态，不使用 is_task_completed
        # ==============================================
        updated_tasks = []
        for t in state["task_list"]:
            if t["task_id"] == current_task["task_id"]:
                new_t = t.copy()
                new_t["status"] = "completed"
                updated_tasks.append(new_t)
            else:
                updated_tasks.append(t)

        return {
            "messages": [answer],
            "task_list": updated_tasks  # 👈 必须回写任务列表
        }

    # ==============================================
    # 2. 第一次调用：只传入任务，不传完整上下文
    # ==============================================
    # 构造仅包含任务的用户消息，不传入完整上下文
    tool_call = tutor_chain.invoke({
        "messages": [
            HumanMessage(content=f"请执行任务：{task_desc}")
        ]
    })

    return {"messages": [tool_call]}