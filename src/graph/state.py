from typing import Annotated, List, Dict, Any, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
import operator


# ---------------------------------------------------------
# 1. 原有数据结构（完全保留，不动）
# ---------------------------------------------------------

class PatientInfo(TypedDict):
    """患者基础病历信息"""
    age: Optional[str]
    gender: Optional[str]
    symptoms: Optional[str]
    duration: Optional[str]
    severity: Optional[str]
    history: Optional[str]
    Supplement: Optional[str]


class LabOrder(TypedDict):
    """单个检查单的数据结构"""
    name: str
    reason: str
    status: str
    result: Optional[str]


class DebateTurn(TypedDict):
    """单轮辩论的记录"""
    role: str
    content: str
    diagnosis_proposal: Optional[str]


# ---------------------------------------------------------
# 2. 【新增】规划层必需的数据结构
# ---------------------------------------------------------

class Task(TypedDict):
    """动态规划任务：任务拆分 + 分配"""
    task_id: int
    description: str
    assigned_agent: str  # DOCTOR/TUTOR/PSYCHOLOGIST/MEDICAL_IMAGE_ANALYST
    status: str  # pending / completed


# ---------------------------------------------------------
# 3. 最终总 State（你原来的 + 规划层字段）
# ---------------------------------------------------------

class AgentState(TypedDict):
    # --------------------------
    # 原有字段（完全保留）
    # --------------------------
    messages: Annotated[List[Any], add_messages]
    patient_info: PatientInfo
    lab_orders: List[LabOrder]
    debate_history: List[DebateTurn]
    final_diagnosis: Optional[str]

    # --------------------------
    # 【规划层新增 3 个字段】
    # --------------------------
    # 1. 任务列表：动态拆出来的所有子任务
    task_list: List[Task]

    # 2. 下一个要执行的 Agent（Supervisor 分配）
    next_agent: str

    # 3. 当前意图（可选，方便调试）
    intent: Optional[str]
    # 4. 任务完成状态

    last_executed_agent: str
