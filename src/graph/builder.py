##规划总编排，构图




from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st

from src.graph.state import AgentState
# 导入所有节点
from src.graph.supervisor import supervisor_node
from src.nodes.agents.joker_chat import chat_node
from src.nodes.agents.doctor import doctor_node
from src.nodes.agents.image_analyst import medical_image_analyst_node
from src.nodes.agents.psychologist_node import psychologist_node
from src.nodes.agents.challenger_node import challenger_node
from src.nodes.agents.medical_tutor import medical_tutor_node
# 导入所有工具
from src.tools.doctor_tools import doctor_tools
from src.tools.image_analyst_tools import image_analysis_tools
from src.tools.psychologist_tools import psychologist_tools
from src.tools.medical_tutor_tools import medical_tutor_tools
# 导入所有路由规则（从edges.py）
from src.graph.edges import (
    should_continue_image,
    should_continue_doctor,
    should_continue_tutor
)

memory = MemorySaver()

@st.cache_resource
def get_compiled_graph():
    builder = StateGraph(AgentState)

    # 1. 加所有节点
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("chat", chat_node)
    builder.add_node("medical_image_analyst", medical_image_analyst_node)
    builder.add_node("psychologist", psychologist_node)
    builder.add_node("doctor", doctor_node)
    builder.add_node("challenger", challenger_node)
    builder.add_node("medical_tutor", medical_tutor_node)

    # 2. 加工具节点
    builder.add_node("doctor_tools", ToolNode(doctor_tools))
    builder.add_node("image_analysis_tools", ToolNode(image_analysis_tools))
    builder.add_node("psychologist_tools", ToolNode(psychologist_tools))
    builder.add_node("medical_tutor_tools", ToolNode(medical_tutor_tools))

    # 3. 入口
    builder.add_edge(START, "supervisor")

    # 4. supervisor 条件路由
    builder.add_conditional_edges(
        "supervisor",
        lambda x: x["next_agent"],
        {
            "DOCTOR": "doctor",
            "CHAT": "chat",
            "MEDICAL_IMAGE_ANALYST": "medical_image_analyst",
            "PSYCHOLOGIST": "psychologist",
            "TUTOR": "medical_tutor",
            "END":END,
            "SUPERVISOR": "supervisor"
        }
    )

    # 5. 普通边
    builder.add_edge("chat", "supervisor")
    builder.add_edge("medical_image_analyst", "supervisor")
    builder.add_edge("doctor_tools", "doctor")
    builder.add_edge("medical_tutor_tools", "medical_tutor")

    # 6. 条件边（全部从edges.py导入，builder里只做连线）
    # builder.add_conditional_edges(
    #     "medical_image_analyst",
    #     should_continue_image,
    #     {"image_analysis_tools": "image_analysis_tools", END: END}
    # )
    builder.add_conditional_edges(
        "doctor",
        should_continue_doctor,
        {"supervisor": "supervisor", "doctor_tools": "doctor_tools", END: END}
    )
    # builder.add_conditional_edges(
    #     "medical_tutor",
    #     should_continue_tutor,
    #     {"medical_tutor_tools": "medical_tutor_tools", END: END}
    # )

    builder.add_conditional_edges(
        "medical_tutor",
        should_continue_tutor,
        {
            "medical_tutor_tools": "medical_tutor_tools",
            "supervisor":"supervisor"
        }
    )
    # 编译返回
    return builder.compile(checkpointer=memory)