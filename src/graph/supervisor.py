from src.graph.state import AgentState
from langchain_openai import ChatOpenAI
from src.graph.reflectors.core import reflect_task_result
from langchain_core.prompts import ChatPromptTemplate
import json

# ---------------------------------------------------------
# 1. LLM 配置
# ---------------------------------------------------------
llm = ChatOpenAI(
    model="Qwen/Qwen3-VL-32B-Instruct",
    api_key="sk-nnbtrcocdpjbbksvonxsgorqgigqsckefstlztcmboafdvwx",
    base_url="https://api.siliconflow.cn/v1",
    temperature=0
)

# ---------------------------------------------------------
# 2. 加载规划提示词
# ---------------------------------------------------------
with open("src/prompts/plan.txt", "r", encoding="utf-8") as f:
    SUPERVISOR_PROMPT = f.read()

# ---------------------------------------------------------
# 意图相关性判断
# ---------------------------------------------------------
def is_intent_related(state):
    messages = state["messages"]
    if len(messages) < 2:
        return False

    # 1. 提取用户最新输入
    user_input = ""
    for msg in reversed(messages):
        if msg.type == "human":
            user_input = msg.content
            break

    if not user_input:
        return False

    # 2. 直接把最近N轮对话给模型，让它自己判断
    recent_conversation = messages[-6:]  # 最近6轮对话，足够判断上下文
    conversation_str = "\n".join([
        f"用户: {m.content}" if m.type == "human" else f"助手: {m.content}"
        for m in recent_conversation
    ])

    prompt = f"""
你是意图判断器。
判断：用户最新输入是否是上一轮对话任务的补充、追问或修改需求？

对话历史：
{conversation_str}

用户最新输入：{user_input}

只输出 True 或 False。
"""
    try:
        resp = llm.invoke(prompt).content.strip().lower()
        return "true" in resp
    except:
        return False
# ---------------------------------------------------------
# 【最终完整版】Supervisor 节点
# ---------------------------------------------------------
def supervisor_node(state: AgentState):
    print("\n📌 --- 规划层：动态任务拆分 + 调度 ---")

    task_list = state.get("task_list", [])
    print(f"当前任务列表: {task_list}")

    # ==========================================
    # 反思机制
    # ==========================================
    # just_completed = any(
    #     t.get("status") == "completed"
    #     and t.get("just_finished") is True
    #     for t in task_list
    # )
    reflection = {}
    # if just_completed:
    #     print("🔍 任务刚完成，执行反思")
    #     reflection = reflect_task_result(state)
    #
    #     if reflection.get("need_retry"):
    #         last_agent = state.get("last_executed_agent")
    #         print(f"🔄 反思建议重试 → {last_agent}")
    #         for task in task_list:
    #             if task.get("just_finished"):
    #                 task["status"] = "pending"
    #                 task["just_finished"] = False
    #                 break
    #         return {
    #             "next_agent": last_agent,
    #             "reflection": reflection,
    #             "task_list": task_list
    #         }
    #
    # # 清除 just_finished 标记
    # for task in task_list:
    #     if task.get("just_finished"):
    #         task["just_finished"] = False






    # ==========================================
    # 所有任务完成 → 核心逻辑
    # ==========================================
    if task_list:
        all_completed = all(t["status"] == "completed" for t in task_list)
        if all_completed:
            print("✅ 所有任务已完成")
            last_msg = state["messages"][-1]
            if last_msg.type == "human":
                print("🧨 用户新输入 → 准备重新规划")

                # 🔥 统一清空队列，强制走重新规划
                task_list = []
                return {
                    "next_agent": "SUPERVISOR",
                    "task_list": task_list,
                    "reflection": reflection
                }

            else:
                print("🔚 无新输入 → END")
                return {
                    "next_agent": "END",
                    "task_list": task_list,
                    "reflection": reflection
                }









    # ==========================================
    # 无任务 → 全新规划
    # ==========================================
    if not task_list:
        print("🔍 首次/新一轮对话：拆分任务")
        recent_messages = state["messages"][-6:]

        related = is_intent_related(state)
        print(f"🤔 意图相关性判断结果: {related}")
        mode = "update" if related else "new"


        prompt = ChatPromptTemplate.from_messages([
            ("system", SUPERVISOR_PROMPT),
            ("placeholder", "{messages}")
        ])
        chain = prompt | llm
        response = chain.invoke({
            "messages": recent_messages,
            "mode": mode,
        }).content.strip()
        print(f"[DEBUG] 拆分结果: {response}")

        try:
            result = json.loads(response)
            if not result.get("task_list"):
                raise ValueError("任务列表为空")
            return {
                "intent": result.get("intent"),
                "task_list": result.get("task_list"),
                "next_agent": result.get("next_agent"),
                "reflection": reflection
            }
        except Exception as e:
            print(f"❌ 拆分失败: {e}")
            default_task = [
                {"task_id": 1, "description": "回应用户", "assigned_agent": "CHAT", "status": "pending"}
            ]
            return {
                "intent": "CHAT",
                "task_list": default_task,
                "next_agent": "CHAT",
                "reflection": reflection
            }

    # ==========================================
    # 调度待执行任务
    # ==========================================
    pending_tasks = [t for t in task_list if t["status"] == "pending"]
    if pending_tasks:
        next_task = pending_tasks[0]
        next_agent = next_task["assigned_agent"]
        print(f"➡️ 调度: {next_task['description']} → {next_agent}")
        return {
            "next_agent": next_agent,
            "reflection": reflection,
            "task_list": task_list
        }

    # ==========================================
    # 兜底 END
    # ==========================================
    return {
        "next_agent": "END",
        "reflection": reflection,
        "task_list": task_list
    }
