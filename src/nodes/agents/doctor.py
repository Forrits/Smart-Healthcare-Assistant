import os
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI
from src.graph.state import AgentState
from src.tools.doctor_tools import doctor_tools
from langchain_core.prompts import ChatPromptTemplate
from src.graph.task_filter import get_current_task
import json

# 初始化LLM
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0.5
)

# 读取提示词
with open("src/prompts/doctor.txt", "r", encoding="utf-8") as f:
    DOCTOR_PROMPT = f.read()

# ==============================================
# 🔥 工业级问诊医生节点（无死循环 + 多轮交互 + 状态机）
# ==============================================
def doctor_node(state: AgentState):
    print("\n👨‍⚕️ --- 医生节点（多轮问诊模式）---")
    state_updates = {}
    messages = state.get("messages", [])
    last_msg = messages[-1] if messages else None

    if last_msg and last_msg.type == "human":
        # 处理多模态内容，提取纯文本
        if isinstance(last_msg.content, list):
            # 遍历列表，拼接所有文本内容
            user_text = "".join(
                part["text"] for part in last_msg.content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        else:
            # 普通字符串内容
            user_text = last_msg.content

        user_text = user_text.strip()

        if "退出问诊" in user_text or "结束问诊" in user_text:
            print("🔴 用户主动退出问诊 → 直接结束任务")

            # 标记完成
            state_updates["is_diagnosed"] = True

            # 找到当前医生任务 → 直接标记完成
            updated_tasks = []
            for t in state["task_list"]:
                if t["assigned_agent"] == "DOCTOR":
                    new_t = t.copy()
                    new_t["status"] = "completed"
                    new_t["just_finished"] = True
                    updated_tasks.append(new_t)
                else:
                    updated_tasks.append(t)

            # 🔥 直接返回！！！后面代码绝对不执行！
            return {
                **state_updates,
                "task_list": updated_tasks,
                # 可以返回一句结束语
                "messages": [{"role": "assistant", "content": "好的，已经退出问诊！"}],
            }


    # 当前任务
    current_task = get_current_task(state, "DOCTOR")
    if not current_task:
        return {}

    # --------------------------
    # 1. 处理工具返回结果（你原有逻辑不变）
    # --------------------------
    if isinstance(last_msg, ToolMessage):# 处理工具返回结果
        print(f"🔧 处理工具返回: {last_msg.name}")
        try:
            tool_result = json.loads(last_msg.content) if isinstance(last_msg.content, str) else last_msg.content

            if last_msg.name == "update_patient_record":
                current_info = state.get("patient_info", {}) or {}
                updates = tool_result.get("patient_info_updates", {})
                append_fields = tool_result.get("_append_fields", [])
                merged_info = {}
                for key, value in updates.items():
                    if key in append_fields and key in current_info and current_info[key]:
                        existing = current_info[key]
                        merged_info[key] = f"{existing}；{value}" if value not in existing else existing
                    else:
                        merged_info[key] = value
                state_updates["patient_info"] = {**current_info, **merged_info}

            elif last_msg.name == "order_lab_test":
                current_orders = state.get("lab_orders", []) or []
                current_orders.append(tool_result["new_lab_order"])
                state_updates["lab_orders"] = current_orders
                state_updates["waiting_for_result"] = tool_result.get("waiting_for_result", True)

            elif last_msg.name == "make_diagnosis":
                state_updates["final_diagnosis"] = tool_result.get("diagnosis_report")
                state_updates["is_diagnosed"] = tool_result.get("is_diagnosed", False)
                print("✅ 诊断完成！")

        except Exception as e:
            print(f"⚠️ 工具解析失败: {e}")

    # --------------------------
    # 2. 状态机：问诊阶段控制（工业核心）
    # --------------------------
    is_diagnosed = state_updates.get("is_diagnosed", state.get("is_diagnosed", False))

    # --------------------------
    # 3. 调用医生LLM
    # --------------------------
    llm_with_tools = llm.bind_tools(doctor_tools)
    doctor_prompt = ChatPromptTemplate.from_messages([
        ("system", DOCTOR_PROMPT),
        ("placeholder", "{messages}")
    ])
    doctor_chain = doctor_prompt | llm_with_tools
    response = doctor_chain.invoke({"messages": messages})

    # --------------------------
    # 4. 诊断完成 → 本轮结束
    # 未完成 → 继续等待用户输入
    # --------------------------

    updated_tasks = []
    for t in state["task_list"]:
        if t["task_id"] == current_task["task_id"]:
            new_t = t.copy()
            # 只有真正诊断完成，才标记任务完成
            new_t["status"] = "completed" if is_diagnosed else "pending"
            updated_tasks.append(new_t)
        else:
            updated_tasks.append(t)

        # 🔚 节点只返回数据，不返回路由指令
    return {
        **state_updates,
        "messages": [response],
        "task_list": updated_tasks,
        "is_diagnosed": is_diagnosed, #会保留在 state 里，供下次轮次使用
        "last_executed_agent": "DOCTOR"
    }
