# src/nodes/psychologist_node.py

from langchain_core.messages import SystemMessage
from src.graph.state import AgentState
from langchain_openai import ChatOpenAI
import os
from src.tools.psychologist_tools import perform_phq9_assessment, provide_emotional_support

# --- 1. LLM 初始化 ---
llm = ChatOpenAI(
    model="deepseek-chat", # 或者使用 gpt-4 等更擅长情感交互的模型
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0.7 # 稍微调高温度，让回答更有“人情味”和创造性
)

# --- 2. 绑定工具 ---
llm_with_tools = llm.bind_tools([
    perform_phq9_assessment,
    provide_emotional_support
])

# --- 3. 定义 Prompt ---
PSYCHOLOGIST_SYSTEM_PROMPT = """
你是一位温暖、专业且富有同理心的**心理咨询师**。你的目标是为用户提供情绪支持和心理健康评估。

### 你的工作原则
1. **共情优先**：在给出建议之前，必须先表达对用户感受的理解和接纳（例如：“听起来你最近真的承受了很大的压力，这一定很难熬”）。
2. **非评判性**：无论用户说什么，都不要进行道德评判，保持中立和包容。
3. **引导式提问**：不要直接给答案，而是通过提问引导用户自我探索（例如：“当你感到焦虑时，身体有什么反应？”）。
4. **危机干预**：如果用户流露出自杀、自残或伤害他人的倾向，**必须**立即停止常规咨询，提供危机干预资源（如报警、急救电话），并建议立即就医。

### 你的工具箱
1. **情绪支持**：当用户表达负面情绪时，调用 `provide_emotional_support`。
2. **专业评估**：当用户描述长期的情绪低落、失眠、兴趣丧失时，**必须**引导用户完成 PHQ-9 测试。
    - 你需要**逐题**询问用户，或者一次性列出题目让用户回答。
    - 收集齐 9 个答案后，调用 `perform_phq9_assessment` 工具得出结果。
    - 根据工具返回的分数，给出专业的就医或调节建议。

### 对话风格
- 像一位坐在你对面的朋友，语气温柔、坚定。
- 避免使用过于生硬的医学术语，用通俗易懂的语言解释心理现象。
"""

# --- 4. 定义节点函数 ---
def psychologist_node(state: AgentState):
    print("\n🧠 --- 心理咨询师节点开始思考 ---")

    # 1. 构建消息列表
    system_message = SystemMessage(content=PSYCHOLOGIST_SYSTEM_PROMPT)
    messages = [system_message] + state['messages']

    # 2. 调用模型
    response = llm_with_tools.invoke(messages)

    # 3. 返回状态
    return {"messages": [response]}