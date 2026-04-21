from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from src.graph.state import AgentState
from src.tools.image_analyst_tools import image_analysis_tools
from src.graph.task_filter import get_current_task
llm = ChatOpenAI(
    model="Qwen/Qwen3.5-122B-A10B",  # 建议使用支持视觉的模型
    api_key="sk-nnbtrcocdpjbbksvonxsgorqgigqsckefstlztcmboafdvwx",
    base_url="https://api.siliconflow.cn/v1",
    temperature=0.3
)

llm_with_tools = llm.bind_tools(image_analysis_tools)

IMAGE_ANALYST_PROMPT = """
你是一位专业的医学影像分诊专家。
你的任务是根据用户提供的图片，进行分析，并总结结果。
"""

# 可用工具：
# 1. analyze_x_ray_tool: 用于 X 光、CT、骨骼。
# 2. analyze_skin_tool: 用于皮肤、外伤、红肿。

# 工作流程：
# 1. 观察图片类型。
# 2. 调用对应的工具获取专业分析结果。
# 3. 将工具返回的结果总结为通俗易懂的中文报告回复给用户。
def medical_image_analyst_node(state: AgentState):

    current_task = get_current_task(state, "MEDICAL_IMAGE_ANALYST")
    if not current_task:
        return {}
    print("\n图片分析节点工作...")
    print(f"任务描述：{current_task['description']}")
    # ------------------------------
    # 🔥 在这里提取图片，这是唯一正确的地方
    # ------------------------------
    messages = state["messages"]
    user_text = ""
    image_base64 = None

    # 从后往前找，只取本轮最新的用户消息（避免历史污染）
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            if isinstance(msg.content, list):
                for item in msg.content:
                    if item.get("type") == "text":
                        user_text = item.get("text", "")
                    elif item.get("type") == "image_url":
                        image_base64 = item.get("image_url", {}).get("url")
            else:
                user_text = msg.content
            break

    # ------------------------------
    # 🔥 把「任务描述 + 文本 + 图片」一起传给多模态模型
    # ------------------------------
    multimodal_prompt = [
        {"type": "text", "text": f"任务：{current_task['description']}\n用户问题：{user_text}"},
    ]
    if image_base64:
        multimodal_prompt.append({
            "type": "image_url",
            "image_url": {"url": image_base64}
        })

    response = llm.invoke([HumanMessage(content=multimodal_prompt)])

    # ------------------------------
    # 标记任务为 completed
    # ------------------------------
    updated_tasks = []
    for t in state["task_list"]:
        if t["task_id"] == current_task["task_id"]:
            new_t = t.copy()
            new_t["status"] = "completed"
            updated_tasks.append(new_t)
        else:
            updated_tasks.append(t)

    return {
        "messages": [response],
        "task_list": updated_tasks
    }