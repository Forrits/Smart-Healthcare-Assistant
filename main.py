import streamlit as st
from dotenv import load_dotenv
import base64
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
from src.Rag.setup import init_medical_rag# 加载环境
init_medical_rag()  # 项目启动只执行1次！
# 🔥 只导入编译好的图
from src.graph.builder import get_compiled_graph

# ==========================================
# 全局初始化
# ==========================================
app = get_compiled_graph()

# ==========================================
# Streamlit UI 界面
# ==========================================
st.title("🏥 模块化医疗问诊系统")

# 侧边栏：图片上传
with st.sidebar:
    st.header("📷 影像/症状上传")
    uploaded_file = st.file_uploader(
        "上传病历或患处照片",
        type=['jpg', 'png', 'jpeg']
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="待发送图片", use_column_width=True)
        try:
            bytes_data = uploaded_file.getvalue()
            base64_str = base64.b64encode(bytes_data).decode("utf-8")
            image_content = {
                "type": "image_url",
                "image_url": {"url": f"data:{uploaded_file.type};base64,{base64_str}"}
            }
            st.session_state.uploaded_image = image_content
            st.success("✅ 图片已准备")
        except Exception as e:
            st.error(f"❌ 图片处理失败：{e}")

# 会话 ID
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "medical-chat-001"

config = {"configurable": {"thread_id": st.session_state.thread_id}}

# 读取聊天历史
state = app.get_state(config)
history = state.values.get("messages", []) if state else []

# 渲染历史消息
for msg in history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user", avatar="👤"):
            content = msg.content
            if isinstance(content, list):
                for item in content:
                    if item["type"] == "text":
                        st.write(item["text"])
                    elif item["type"] == "image_url":
                        st.image(item["image_url"]["url"], width=300)
            else:
                st.write(content)

    elif isinstance(msg, AIMessage) and msg.content:
        with st.chat_message("assistant", avatar="🏥"):
            content = msg.content
            if isinstance(content, list):
                text = "".join([item.get("text", "") for item in content if isinstance(item, dict)])
                st.write(text)
            else:
                st.write(content)

# ==========================================
# 用户输入处理
# ==========================================
if prompt := st.chat_input("请输入症状或问题..."):
    with st.chat_message("user", avatar="👤"):
        st.write(prompt)
        if "uploaded_image" in st.session_state:
            st.image(st.session_state.uploaded_image["image_url"]["url"], width=300)

    content_list = [{"type": "text", "text": prompt}]
    if "uploaded_image" in st.session_state:
        content_list.append(st.session_state.uploaded_image)
        del st.session_state.uploaded_image

    with st.chat_message("assistant", avatar="🏥"):
        placeholder = st.empty()
        full_resp = ""

        for event in app.stream(
                {
                    "messages": [HumanMessage(content=content_list)],
                    "debate_round_count": 0
                },
                config,
                stream_mode="values"
        ):
            if "messages" not in event:
                continue

            messages = event["messages"]
            current_ai_answers = []

            # ==============================================
            # 🔥 核心修复1：精准定位【本轮用户消息】的索引
            # ==============================================
            # 找到最后一条（本轮）用户输入的位置
            user_msg_index = -1
            for i, msg in reversed(list(enumerate(messages))):
                if isinstance(msg, HumanMessage):
                    # 匹配本轮用户输入（严格匹配内容+角色）
                    if msg.content == content_list:
                        user_msg_index = i
                        break

            # ==============================================
            # 🔥 核心修复2：只取【本轮用户消息之后】的AI消息
            # ==============================================
            if user_msg_index != -1:
                # 只遍历本轮用户消息之后的所有消息
                for msg in messages[user_msg_index + 1:]:
                    if isinstance(msg, AIMessage) and msg.content:
                        # 提取纯文本，过滤工具调用/空消息
                        if isinstance(msg.content, list):
                            text = "".join([item.get("text", "") for item in msg.content])
                        else:
                            text = str(msg.content)

                        # 过滤工具调用消息（仅保留最终回答）
                        if not text.strip() or "tool_call" in str(msg.additional_kwargs):
                            continue

                        current_ai_answers.append(text.strip())

            # ==============================================
            # 🔥 核心修复3：去重 + 合并
            # ==============================================
            # 去重，避免同一任务重复输出
            unique_answers = list(dict.fromkeys(current_ai_answers))
            full_resp = "\n\n".join(unique_answers)
            placeholder.markdown(full_resp + "▌")

        placeholder.markdown(full_resp)
