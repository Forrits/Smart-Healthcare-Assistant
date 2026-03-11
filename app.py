# app.py

from typing import TypedDict, Annotated, List, Dict, Any, Optional, Literal
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import json
from datetime import datetime
import uuid
from medrag import init_medical_rag, retrieve_medical_knowledge
# 加载环境变量
load_dotenv()

# 初始化DeepSeek模型
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API"),
    base_url="https://api.deepseek.com/v1",
    temperature=0.2
)


# 1. 定义全局状态（修复所有未定义字段）
class MedicalDiagnosisState(TypedDict):
    """医疗诊断工作流的状态结构"""
    messages: Annotated[List, add_messages]
    patient_info: Dict[str, Any]
    symptoms: List[Dict[str, Any]]
    vital_signs: Dict[str, Any]
    lab_results: List[Dict[str, Any]]
    preliminary_diagnosis: List[str]
    recommended_tests: List[str]
    treatment_plan: Dict[str, Any]
    follow_up_plan: Dict[str, Any]
    current_stage: str
    urgency_level: str
    doctor_approval: Dict[str, bool]
    cycle_count: int
    max_cycles: int
    final_report: str
    error: Optional[str]
    session_id: str
    needs_consultation: bool  # 新增：解决InvalidUpdateError


# 2. 定义工作流节点 (所有节点函数保持逻辑不变，仅确保返回字段匹配状态类)
def initial_assessment_node(state: MedicalDiagnosisState) -> Dict[str, Any]:
    """步骤1：初步评估和分诊"""
    try:
        patient_info = state["patient_info"]
        symptoms = state["symptoms"]
        vital_signs = state["vital_signs"]



        assessment_prompt = f"""
        你是一个经验丰富的急诊科医生。请根据以下信息进行初步评估：

        患者信息：
        {json.dumps(patient_info, ensure_ascii=False, indent=2)}

        症状：
        {json.dumps(symptoms, ensure_ascii=False, indent=2)}

        生命体征：
        {json.dumps(vital_signs, ensure_ascii=False, indent=2)}

        请完成以下任务：
        1. 评估紧急程度（低/中/高/紧急）
        2. 识别可能的紧急情况
        3. 推荐立即需要的检查
        4. 给出初步印象

        请以JSON格式返回：
        {{
            "urgency_level": "紧急程度",
            "emergency_signs": ["紧急征象1", "紧急征象2"],
            "immediate_actions": ["立即行动1", "立即行动2"],
            "recommended_tests": ["推荐检查1", "推荐检查2"],
            "preliminary_impression": "初步印象"
        }}
        """

        response = llm.invoke([SystemMessage(content=assessment_prompt)])

        # 解析JSON响应
        try:
            response_text = response.content.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:-3]

            assessment_result = json.loads(response_text)
            urgency_level = assessment_result.get("urgency_level", "medium")
            recommended_tests = assessment_result.get("recommended_tests", [])
        except json.JSONDecodeError:
            urgency_level = "medium"
            recommended_tests = ["血常规", "体温检查"]

        # 翻译紧急程度为中文
        urgency_map = {
            "low": "低",
            "medium": "中",
            "high": "高",
            "emergency": "紧急"
        }
        urgency_cn = urgency_map.get(urgency_level, "中")

        return {
            "urgency_level": urgency_level,
            "recommended_tests": recommended_tests,
            "current_stage": "assessed",
            "messages": [AIMessage(content=f"🏥 初步评估完成，紧急程度：{urgency_cn}")]
        }

    except Exception as e:
        return {
            "error": f"初步评估失败: {str(e)}",
            "current_stage": "assessment_failed",
            "messages": [AIMessage(content=f"❌ 初步评估遇到问题: {str(e)}")]
        }


def decide_testing_strategy_node(state: MedicalDiagnosisState) -> Dict[str, Any]:
    """步骤2：决定检查策略"""
    try:
        urgency_level = state["urgency_level"]
        recommended_tests = state["recommended_tests"]

        # 根据紧急程度决定检查策略
        if urgency_level == "emergency":
            return {
                "current_stage": "emergency_testing",
                "messages": [AIMessage(content="🚨 启动紧急检查流程")]
            }
        elif urgency_level == "high":
            return {
                "current_stage": "priority_testing",
                "messages": [AIMessage(content="⚡ 启动优先检查流程")]
            }
        elif recommended_tests:
            return {
                "current_stage": "routine_testing",
                "messages": [AIMessage(content="📋 启动常规检查流程")]
            }
        else:
            return {
                "current_stage": "diagnosis",
                "messages": [AIMessage(content="🔍 直接进入诊断阶段")]
            }

    except Exception as e:
        return {
            "error": f"检查策略决策失败: {str(e)}",
            "current_stage": "testing_decision_failed",
            "messages": [AIMessage(content=f"❌ 检查策略决策遇到问题: {str(e)}")]
        }


def order_tests_node(state: MedicalDiagnosisState) -> Dict[str, Any]:
    """步骤3：开具检查单"""
    try:
        recommended_tests = state["recommended_tests"]

        # 模拟检查结果生成
        test_results = []
        for test in recommended_tests:
            result = {
                "test_name": test,
                "status": "completed",
                "result": f"{test}结果正常",
                "timestamp": datetime.now().strftime("%Y年%m月%d日 %H:%M:%S"),
                "normal_range": "正常范围"
            }
            test_results.append(result)

        return {
            "lab_results": test_results,
            "current_stage": "tested",
            "messages": [AIMessage(content=f"🧪 检查完成，共{len(test_results)}项")]
        }

    except Exception as e:
        return {
            "error": f"检查执行失败: {str(e)}",
            "current_stage": "testing_failed",
            "messages": [AIMessage(content=f"❌ 检查执行遇到问题: {str(e)}")]
        }


def make_diagnosis_node(state: MedicalDiagnosisState) -> Dict[str, Any]:
    """步骤4：诊断分析"""
    try:
        patient_info = state["patient_info"]
        symptoms = state["symptoms"]
        vital_signs = state["vital_signs"]
        lab_results = state["lab_results"]
################################################################################################

        # 构建更精准的检索查询
        test_results_text = ", ".join([f"{r['test_name']}: {r['result']}" for r in lab_results])
        rag_query = f"{patient_info.get('age', '')}岁{patient_info.get('gender', '')}，{test_results_text}，症状：{', '.join([s['symptom'] for s in symptoms])}，诊断"
        # 补充检索更多相关知识
        additional_knowledge = retrieve_medical_knowledge(rag_query, k=4)
        full_knowledge = additional_knowledge
      

        diagnosis_prompt = f"""
        你是一个专业的诊断医生。请根据以下信息进行诊断分析：

        患者信息：
        {json.dumps(patient_info, ensure_ascii=False, indent=2)}

        症状：
        {json.dumps(symptoms, ensure_ascii=False, indent=2)}

        生命体征：
        {json.dumps(vital_signs, ensure_ascii=False, indent=2)}

        检查结果：
        {json.dumps(lab_results, ensure_ascii=False, indent=2)}
       本地医疗医疗知识库：
        {full_knowledge}
        请提供：
        1. 主要诊断（可能多个）
        2. 鉴别诊断
        3. 诊断依据
        4. 严重程度评估（轻度/中度/重度）
        5. 是否需要会诊

        请以JSON格式返回：
        {{
            "main_diagnosis": ["诊断1", "诊断2"],
            "differential_diagnosis": ["鉴别诊断1", "鉴别诊断2"],
            "diagnostic_basis": "诊断依据",
            "severity": "mild/moderate/severe",
            "needs_consultation": true/false,
            "consultation_specialty": "会诊科室"
        }}
        """

        response = llm.invoke([SystemMessage(content=diagnosis_prompt)])

        # 解析JSON响应
        try:
            response_text = response.content.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:-3]

            diagnosis_result = json.loads(response_text)
            main_diagnosis = diagnosis_result.get("main_diagnosis", [])
            needs_consultation = diagnosis_result.get("needs_consultation", False)
        except json.JSONDecodeError:
            main_diagnosis = ["待查"]
            needs_consultation = False

        return {
            "preliminary_diagnosis": main_diagnosis,
            "current_stage": "diagnosed",
            "needs_consultation": needs_consultation,  # 已在状态类中定义
            "messages": [AIMessage(content=f"📋 诊断完成：{', '.join(main_diagnosis)}")]
        }

    except Exception as e:
        return {
            "error": f"诊断失败: {str(e)}",
            "current_stage": "diagnosis_failed",
            "messages": [AIMessage(content=f"❌ 诊断过程遇到问题: {str(e)}")]
        }


def create_treatment_plan_node(state: MedicalDiagnosisState) -> Dict[str, Any]:
    """步骤5：制定治疗方案"""
    try:
        preliminary_diagnosis = state["preliminary_diagnosis"]
        patient_info = state["patient_info"]

        current_date = datetime.now().strftime("%Y年%m月%d日")

        treatment_prompt = f"""
        请为以下患者制定治疗方案：

        诊断：
        {', '.join(preliminary_diagnosis)}

        患者信息：
        {json.dumps(patient_info, ensure_ascii=False, indent=2)}

        重要提示：今天是 {current_date}，所有复查日期请基于此日期计算。

        请提供：
        1. 药物治疗方案
        2. 非药物治疗
        3. 生活方式建议
        4. 注意事项
        5. 复查计划

        请以JSON格式返回：
        {{
            "medications": [
                {{"name": "药物名", "dosage": "剂量", "frequency": "频次", "duration": "疗程"，"detail":"适应症"，"dangerous"："禁忌"}}
            ],
            "non_pharmacological": ["非药物措施1", "非药物措施2"],
            "lifestyle_advice": ["生活建议1", "生活建议2"],
            "precautions": ["注意事项1", "注意事项2"],
            "follow_up_schedule": "复查计划"
        }}
        """

        response = llm.invoke([SystemMessage(content=treatment_prompt)])

        # 解析JSON响应
        try:
            response_text = response.content.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:-3]

            treatment_result = json.loads(response_text)
        except json.JSONDecodeError:
            treatment_result = {
                "medications": [],
                "non_pharmacological": [],
                "lifestyle_advice": [],
                "precautions": [],
                "follow_up_schedule": "1周后复查"
            }

        return {
            "treatment_plan": treatment_result,
            "current_stage": "treatment_planned",
            "messages": [AIMessage(content="💊 治疗方案制定完成")]
        }

    except Exception as e:
        return {
            "error": f"治疗方案制定失败: {str(e)}",
            "current_stage": "treatment_planning_failed",
            "messages": [AIMessage(content=f"❌ 治疗方案制定遇到问题: {str(e)}")]
        }


def doctor_approval_node(state: MedicalDiagnosisState) -> Dict[str, Any]:
    """步骤6：医生审批节点"""
    try:
        current_stage = state["current_stage"]
        preliminary_diagnosis = state["preliminary_diagnosis"]
        treatment_plan = state.get("treatment_plan", {})

        # 模拟医生审批过程
        approval_prompt = f"""
        模拟医生审批以下内容：

        诊断：{', '.join(preliminary_diagnosis)}
        治疗方案：{json.dumps(treatment_plan, ensure_ascii=False, indent=2)}

        请决定是否批准（返回true/false）：
        """

        response = llm.invoke([SystemMessage(content=approval_prompt)])
        approved = "true" in response.content.lower()

        approval_type = "diagnosis" if current_stage == "diagnosed" else "treatment"

        return {
            "doctor_approval": {approval_type: approved},
            "current_stage": "approved" if approved else "revision_needed",
            "messages": [AIMessage(content=f"👨‍⚕️ 医生审批结果：{'批准' if approved else '需要修改'}")]
        }

    except Exception as e:
        return {
            "error": f"医生审批失败: {str(e)}",
            "current_stage": "approval_failed",
            "messages": [AIMessage(content=f"❌ 医生审批遇到问题: {str(e)}")]
        }


def follow_up_planning_node(state: MedicalDiagnosisState) -> Dict[str, Any]:
    """步骤7：随访计划"""
    try:
        treatment_plan = state.get("treatment_plan", {})
        preliminary_diagnosis = state["preliminary_diagnosis"]
        patient_info = state["patient_info"]

        current_date = datetime.now().strftime("%Y年%m月%d日")

        follow_up_prompt = f"""
        请为患者制定随访计划：

        诊断：{', '.join(preliminary_diagnosis)}
        治疗方案：{json.dumps(treatment_plan, ensure_ascii=False, indent=2)}
        患者信息：{json.dumps(patient_info, ensure_ascii=False, indent=2)}

        重要提示：今天是 {current_date}，所有日期请基于此日期计算。例如：
        - 1周后应该是 {datetime.now().strftime('%Y年%m月%d日')} 的7天后
        - 1个月后应该是 {datetime.now().strftime('%Y年%m月%d日')} 的30天后

        请提供：
        1. 随访时间表
        2. 随访项目
        3. 需要观察的症状
        4. 紧急情况处理

        请以JSON格式返回：
        {{
            "follow_up_schedule": [
                {{"time": "时间点", "items": ["检查项目1", "检查项目2"]}}
            ],
            "monitoring_symptoms": ["症状1", "症状2"],
            "emergency_indicators": ["紧急指标1", "紧急指标2"],
            "next_appointment": "下次预约时间"
        }}
        """

        response = llm.invoke([SystemMessage(content=follow_up_prompt)])

        # 解析JSON响应
        try:
            response_text = response.content.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:-3]

            follow_up_result = json.loads(response_text)
        except json.JSONDecodeError:
            follow_up_result = {
                "follow_up_schedule": [],
                "monitoring_symptoms": [],
                "emergency_indicators": [],
                "next_appointment": "1周后"
            }

        return {
            "follow_up_plan": follow_up_result,
            "current_stage": "completed",
            "messages": [AIMessage(content="📅 随访计划制定完成")]
        }

    except Exception as e:
        return {
            "error": f"随访计划制定失败: {str(e)}",
            "current_stage": "follow_up_planning_failed",
            "messages": [AIMessage(content=f"❌ 随访计划制定遇到问题: {str(e)}")]
        }


def generate_final_report_node(state: MedicalDiagnosisState) -> Dict[str, Any]:
    """步骤8：生成最终报告"""
    try:
        patient_info = state["patient_info"]
        symptoms = state["symptoms"]
        preliminary_diagnosis = state["preliminary_diagnosis"]
        treatment_plan = state.get("treatment_plan", {})
        follow_up_plan = state.get("follow_up_plan", {})

        current_date = datetime.now().strftime("%Y年%m月%d日")

        report_prompt = f"""
        生成完整的医疗报告：

        患者信息：{json.dumps(patient_info, ensure_ascii=False, indent=2)}
        症状：{json.dumps(symptoms, ensure_ascii=False, indent=2)}
        诊断：{', '.join(preliminary_diagnosis)}
        治疗方案：{json.dumps(treatment_plan, ensure_ascii=False, indent=2)}
        随访计划：{json.dumps(follow_up_plan, ensure_ascii=False, indent=2)}

        重要提示：今天是 {current_date}，报告中的所有日期请基于此日期生成。

        请生成专业的医疗报告，包括：
        1. 患者基本信息
        2. 主诉和现病史
        3. 检查结果
        4. 诊断结论
        5. 治疗方案
        6. 随访计划
        7. 医生建议
        """

        response = llm.invoke([SystemMessage(content=report_prompt)])
        final_report = response.content

        return {
            "final_report": final_report,
            "current_stage": "report_completed",
            "messages": [AIMessage(content="📄 最终报告生成完成")]
        }

    except Exception as e:
        return {
            "error": f"报告生成失败: {str(e)}",
            "current_stage": "report_failed",
            "messages": [AIMessage(content=f"❌ 报告生成遇到问题: {str(e)}")]
        }


# 3. 构建图
def create_medical_diagnosis_assistant():
    """创建医疗诊断工作流"""
    workflow = StateGraph(MedicalDiagnosisState)

    # 添加节点（修复doctor_approval命名冲突）
    workflow.add_node("initial_assessment", initial_assessment_node)
    workflow.add_node("decide_testing", decide_testing_strategy_node)
    workflow.add_node("order_tests", order_tests_node)
    workflow.add_node("make_diagnosis", make_diagnosis_node)
    workflow.add_node("create_treatment", create_treatment_plan_node)
    workflow.add_node("doctor_approval_node", doctor_approval_node)
    workflow.add_node("follow_up_planning", follow_up_planning_node)
    workflow.add_node("generate_report", generate_final_report_node)

    # 定义条件路由函数
    def route_after_assessment(state: MedicalDiagnosisState) -> Literal[
        "decide_testing", "order_tests", "make_diagnosis", "generate_report"]:
        """初步评估后的路由决策"""
        if state.get("error") or "failed" in state.get("current_stage", ""):
            return "generate_report"
        urgency = state.get("urgency_level", "")
        if urgency == "emergency":
            return "order_tests"  # 紧急情况直接检查
        return "decide_testing"  # 其他情况先决定检查策略

    def route_after_testing_decision(state: MedicalDiagnosisState) -> Literal["order_tests", "make_diagnosis"]:
        """检查策略决策后的路由"""
        stage = state.get("current_stage", "")
        if "testing" in stage:
            return "order_tests"
        return "make_diagnosis"

    def route_after_diagnosis(state: MedicalDiagnosisState) -> Literal[
        "create_treatment", "doctor_approval_node", "follow_up_planning"]:
        """诊断后的路由（修复节点名引用）"""
        if state.get("needs_consultation", False):
            return "doctor_approval_node"
        return "create_treatment"

    # def route_after_treatment(state: MedicalDiagnosisState) -> Literal["doctor_approval_node", "follow_up_planning"]:
    #     """治疗方案制定后的路由（修复节点名引用）"""
    #     return "doctor_approval_node"

    def route_after_approval(state: MedicalDiagnosisState) -> Literal[
        "follow_up_planning", "make_diagnosis", "create_treatment"]:
        """医生审批后的路由"""
        stage = state.get("current_stage", "")
        approvals = state.get("doctor_approval", {})
        if stage == "revision_needed":
            if not approvals.get("diagnosis", True):
                return "make_diagnosis"
            elif not approvals.get("treatment", True):
                return "create_treatment"

        return "follow_up_planning"

    # 设置流程入口（修复START不存在的问题）
    workflow.set_entry_point("initial_assessment")

    # 添加条件边（确保所有节点引用正确）
    workflow.add_conditional_edges(
        "initial_assessment",
        route_after_assessment,
        {
            "decide_testing": "decide_testing",
            "order_tests": "order_tests",
            "make_diagnosis": "make_diagnosis",
            "generate_report": "generate_report"
        }
    )

    workflow.add_conditional_edges(
        "decide_testing",
        route_after_testing_decision,
        {
            "order_tests": "order_tests",
            "make_diagnosis": "make_diagnosis"
        }
    )

    workflow.add_edge("order_tests", "make_diagnosis")


    #这个边是诊断后
    workflow.add_conditional_edges(
        "make_diagnosis",
        route_after_diagnosis,
        {
            "create_treatment": "create_treatment",
            "doctor_approval_node": "doctor_approval_node",
            "follow_up_planning": "follow_up_planning"
        }
    )

    workflow.add_edge("create_treatment", "doctor_approval_node")

    workflow.add_conditional_edges(
        "doctor_approval_node",
        route_after_approval,
        {
            "follow_up_planning": "follow_up_planning",
            "make_diagnosis": "make_diagnosis",
            "create_treatment": "create_treatment"
        }
    )

    workflow.add_edge("follow_up_planning", "generate_report")
    workflow.add_edge("generate_report", END)

    # 编译图
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app


# 4. 使用示例
def medical_diagnosis(patient_info: Dict[str, Any], symptoms: List[Dict[str, Any]],
                      vital_signs: Dict[str, Any], stream: bool = False):
    """医疗诊断的便捷函数"""
    app = create_medical_diagnosis_assistant()

    # 生成唯一的会话ID
    session_id = str(uuid.uuid4())

    initial_state = {
        "messages": [HumanMessage(content="开始医疗诊断流程")],
        "patient_info": patient_info,
        "symptoms": symptoms,
        "vital_signs": vital_signs,
        "lab_results": [],
        "preliminary_diagnosis": [],
        "recommended_tests": [],
        "treatment_plan": {},
        "follow_up_plan": {},
        "current_stage": "initial",
        "urgency_level": "medium",
        "doctor_approval": {},
        "cycle_count": 0,
        "max_cycles": 3,
        "final_report": "",
        "error": None,
        "session_id": session_id,
        "needs_consultation": False  # 初始化新增字段
    }

    # 配置checkpointer所需的config
    config = {
        "configurable": {
            "thread_id": session_id,
            "checkpoint_ns": "medical_diagnosis"
        }
    }

    if stream:
        # 流式运行
        return app.stream(initial_state, config=config)
    else:
        # 一次性运行
        result = app.invoke(initial_state, config=config)
        return result


# --- Streamlit 界面部分 ---
import streamlit as st

# 设置页面配置
st.set_page_config(
    page_title="智能医疗诊断助手",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 优化的内置样式（不引入外部CSS文件）
st.markdown("""
<style>
    /* 主标题样式 */
    h1 {
        color: #165DFF;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* 二级标题样式 */
    h2 {
        color: #0E42D2;
        font-weight: 600;
        border-left: 4px solid #165DFF;
        padding-left: 10px;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }

    /* 三级标题样式 */
    h3 {
        color: #2E5BCC;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    /* 表单容器样式 */
    .stForm {
        background-color: #F5F7FA;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #E5E9F2;
    }

    /* 扩展面板样式 */
    .streamlit-expanderHeader {
        background-color: #EEF2FF;
        border-radius: 6px;
    }

    /* 按钮样式优化 */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }

    /* 主要按钮样式 */
    .stButton>button[type="primary"] {
        background-color: #165DFF;
        color: white;
    }

    /* 结果容器样式 */
    .result-box {
        background-color: #F9FBFF;
        border-radius: 8px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        border: 1px solid #E8F0FE;
    }

    /* 成功消息样式 */
    .success-text {
        color: #00B42A;
        font-weight: 500;
    }

    /* 错误消息样式 */
    .error-text {
        color: #F53F3F;
        font-weight: 500;
    }

    /* 警告消息样式 */
    .warning-text {
        color: #FF7D00;
        font-weight: 500;
    }

    /* 进度条样式 */
    .stProgress > div > div {
        background-color: #165DFF;
    }

    /* 侧边栏样式优化 */
    [data-testid="stSidebar"] {
        background-color: #F8FAFF;
    }

    /* 输入框样式优化 */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stTextArea>div>div>textarea {
        border-radius: 6px;
        border: 1px solid #D0D7E3;
    }

    /* 选择框样式优化 */
    .stSelectbox>div>div>select {
        border-radius: 6px;
        border: 1px solid #D0D7E3;
    }
</style>
""", unsafe_allow_html=True)

# 主标题

st.title("🤖  智能医疗诊断助手 🩺 ")
st.divider()

# 优化布局：将患者信息移到主界面顶部，而非侧边栏
with st.container():
    st.subheader("📋 患者基本信息")
    col1, col2, col3 = st.columns(3)
    with col1:
        name = st.text_input("姓名", value="", placeholder="请输入患者姓名")
    with col2:
        age = st.number_input("年龄", min_value=0, max_value=120, value=20, help="请输入患者实际年龄")
    with col3:
        gender = st.selectbox("性别", ["男", "女", "其他"], index=0)

    col4, col5 = st.columns([2, 1])
    with col4:
        medical_history = st.text_area("既往病史", value="", placeholder="例如：高血压, 糖尿病, 冠心病")
    with col5:
        allergies = st.text_area("过敏史", value="", placeholder="例如：青霉素, 海鲜, 花粉")

    medications = st.text_input("当前用药", value="", placeholder="例如：降压药, 二甲双胍, 阿司匹林")

# 主体部分：症状和生命体征（优化布局）
st.subheader("🩺 临床信息录入")
tab1, tab2 = st.tabs(["症状描述", "生命体征"])

with tab1:
    # 动态添加症状
    if 'symptom_count' not in st.session_state:
        st.session_state.symptom_count = 1

    symptoms_list = []

    # 症状录入区域
    symptom_container = st.container()

    with symptom_container:
        for i in range(st.session_state.symptom_count):
            with st.expander(f"症状 {i + 1}", expanded=(i == 0)):
                col_sym1, col_sym2 = st.columns(2)
                with col_sym1:
                    symptom_name = st.text_input(f"症状名称", key=f"symptom_name_{i}",
                                                 value="" if i == 0 else "",
                                                 placeholder="例如：头痛、咳嗽、胸闷")
                    duration = st.text_input(f"持续时间", key=f"duration_{i}",
                                             value="" if i == 0 else "",
                                             placeholder="例如：3天、2小时、1周")
                with col_sym2:
                    severity = st.selectbox(f"严重程度", key=f"severity_{i}",
                                            options=["轻度", "中度", "重度"],
                                            index=1 if i == 0 else 0)
                    details = st.text_input(f"详情/位置/诱因", key=f"details_{i}",
                                            value="" if i == 0 else "",
                                            placeholder="例如：胸骨后、左侧头部、运动后加重")

                if symptom_name:
                    symptoms_list.append({
                        "symptom": symptom_name,
                        "duration": duration,
                        "severity": severity,
                        "details": details
                    })

        # 按钮布局优化
        col_btn_sym1, col_btn_sym2, col_btn_sym3 = st.columns([1, 1, 8])
        with col_btn_sym1:
            add_symptom = st.button("➕ 添加症状", key="add_symptom")
        with col_btn_sym2:
            if st.session_state.symptom_count > 1:
                remove_symptom = st.button("➖ 移除症状", key="remove_symptom")

        if add_symptom:
            st.session_state.symptom_count += 1
            st.rerun()

        if st.session_state.symptom_count > 1 and 'remove_symptom' in locals() and remove_symptom:
            st.session_state.symptom_count -= 1
            st.rerun()

with tab2:
    # 生命体征录入（优化布局）
    col_vit1, col_vit2, col_vit3 = st.columns(3)
    with col_vit1:
        blood_pressure = st.text_input("血压 (mmHg)", value="140/90", help="格式：收缩压/舒张压")
        heart_rate = st.number_input("心率 (bpm)", min_value=0, max_value=300, value=95, help="次/分钟")
    with col_vit2:
        temperature = st.number_input("体温 (°C)", min_value=35.0, max_value=42.0, value=36.8, step=0.1)
        respiratory_rate = st.number_input("呼吸频率 (次/分)", min_value=0, max_value=60, value=20)
    with col_vit3:
        oxygen_saturation = st.number_input("血氧饱和度 (%)", min_value=70, max_value=100, value=98)
        bmi = st.number_input("BMI (可选)", min_value=10.0, max_value=50.0, value=24.0, step=0.1, help="身体质量指数")

# 诊断按钮（优化布局）
st.divider()
col_diag1, col_diag2, col_diag3 = st.columns([3, 2, 3])
with col_diag2:
    diagnose_button = st.button("🔍 开始智能诊断", type="primary", use_container_width=True)

# 显示诊断结果（优化UI）
if diagnose_button:
    if not name or not symptoms_list:
        st.error("⚠️ 请至少填写患者姓名和一个症状！")
    else:
        # 准备输入数据
        patient_info = {
            "name": name,
            "age": age,
            "gender": gender,
            "medical_history": [h.strip() for h in medical_history.split(",") if h.strip()],
            "allergies": [a.strip() for a in allergies.split(",") if a.strip()],
            "medications": [m.strip() for m in medications.split(",") if m.strip()]
        }

        vital_signs = {
            "blood_pressure": f"{blood_pressure} mmHg",
            "heart_rate": f"{heart_rate} bpm",
            "temperature": f"{temperature}°C",
            "respiratory_rate": f"{respiratory_rate} breaths/min",
            "oxygen_saturation": f"{oxygen_saturation}%",
            "bmi": f"{bmi}"
        }

        # 创建结果容器（优化样式）
        st.subheader("📊 诊断结果")
        result_container = st.container()

        with result_container:
            # 创建进度条和状态文本
            progress_col, status_col = st.columns([8, 2])
            with progress_col:
                progress_bar = st.progress(0)
            with status_col:
                status_text = st.empty()
                status_text.markdown("<div style='text-align: center;'><b>准备中...</b></div>", unsafe_allow_html=True)

            # 创建消息容器用于流式输出
            message_container = st.container()

            # 运行诊断流程
            with st.spinner("🤖 正在进行智能诊断，请稍候..."):
                try:
                    # 使用流式运行以显示进度
                    stream_results = medical_diagnosis(patient_info, symptoms_list, vital_signs, stream=True)

                    # 显示每个步骤的结果
                    step = 0
                    final_result = {}
                    accumulated_state = {}  # 用于累积状态
                    step_names = {
                        "initial_assessment": "初步评估",
                        "decide_testing": "决定检查策略",
                        "order_tests": "执行检查",
                        "make_diagnosis": "诊断分析",
                        "create_treatment": "制定治疗方案",
                        "doctor_approval_node": "医生审批",
                        "follow_up_planning": "制定随访计划",
                        "generate_report": "生成最终报告"
                    }

                    with message_container:
                        for event in stream_results:
                            step += 1
                            progress = min(step / 8.0, 1.0)  # 假设大约有8个步骤
                            progress_bar.progress(progress)

                            for node_name, node_data in event.items():
                                if node_name != "__end__":
                                    # 更新状态文本
                                    step_name_cn = step_names.get(node_name, node_name)
                                    status_text.markdown(
                                        f"<div style='text-align: center;'><b>{step_name_cn}</b></div>",
                                        unsafe_allow_html=True)

                                    # 累积状态数据
                                    accumulated_state.update(node_data)

                                    # 显示步骤结果（优化样式）
                                    with st.expander(f"📝 步骤 {step}: {step_name_cn}", expanded=(step <= 2)):
                                        for msg in node_data.get("messages", []):
                                            if isinstance(msg, AIMessage):
                                                if "❌" in msg.content:
                                                    st.markdown(f'<p class="error-text">{msg.content}</p>',
                                                                unsafe_allow_html=True)
                                                elif "🚨" in msg.content or "紧急" in msg.content:
                                                    st.markdown(f'<p class="warning-text">{msg.content}</p>',
                                                                unsafe_allow_html=True)
                                                else:
                                                    st.markdown(f'<p class="success-text">{msg.content}</p>',
                                                                unsafe_allow_html=True)

                                # 保存最终结果
                                if node_name == "__end__":
                                    final_result = node_data

                    # 如果没有获取到final_result，使用累积的状态
                    if not final_result and accumulated_state:
                        final_result = accumulated_state

                    # 更新进度为完成
                    progress_bar.progress(1.0)
                    status_text.markdown("<div style='text-align: center;'><b>✅ 完成</b></div>", unsafe_allow_html=True)

                    # 显示最终报告（优化布局和样式）
                    st.divider()

                    # 检查是否有错误
                    if final_result.get("error"):
                        st.markdown(
                            f'<div class="result-box"><p class="error-text">⚠️ 诊断过程中出现错误: {final_result.get("error", "未知错误")}</p></div>',
                            unsafe_allow_html=True)
                    else:
                        # 诊断摘要卡片
                        col_summary1, col_summary2, col_summary3 = st.columns(3)

                        with col_summary1:
                            st.markdown('<div class="result-box"><h3>📋 初步诊断</h3></div>', unsafe_allow_html=True)
                            diagnosis_list = final_result.get("preliminary_diagnosis", [])
                            if diagnosis_list:
                                for diag in diagnosis_list:
                                    st.markdown(f'<div class="result-box">• {diag}</div>', unsafe_allow_html=True)
                            else:
                                st.markdown('<div class="result-box">• 暂无诊断结果</div>', unsafe_allow_html=True)

                        with col_summary2:
                            st.markdown('<div class="result-box"><h3>🚨 紧急程度</h3></div>', unsafe_allow_html=True)
                            urgency_map = {"low": "低", "medium": "中", "high": "高", "emergency": "紧急"}
                            urgency_cn = urgency_map.get(final_result.get("urgency_level", "medium"), "中")
                            urgency_color = {"低": "#00B42A", "中": "#FF7D00", "高": "#F53F3F", "紧急": "#F53F3F"}
                            st.markdown(
                                f'<div class="result-box"><span style="color: {urgency_color.get(urgency_cn, "#FF7D00")}; font-size: 1.2rem; font-weight: bold;">{urgency_cn}</span></div>',
                                unsafe_allow_html=True)

                        with col_summary3:
                            st.markdown('<div class="result-box"><h3>🧪 推荐检查</h3></div>', unsafe_allow_html=True)
                            tests_list = final_result.get("recommended_tests", [])
                            if tests_list:
                                for test in tests_list[:5]:  # 只显示前5项
                                    st.markdown(f'<div class="result-box">• {test}</div>', unsafe_allow_html=True)
                                if len(tests_list) > 5:
                                    st.markdown(f'<div class="result-box">• 还有 {len(tests_list) - 5} 项检查...</div>',
                                                unsafe_allow_html=True)
                            else:
                                st.markdown('<div class="result-box">• 暂无推荐检查</div>', unsafe_allow_html=True)

                        # 详细治疗方案
                        st.markdown('<h3>💊 治疗方案</h3>', unsafe_allow_html=True)
                        treatment = final_result.get("treatment_plan", {})

                        if treatment:
                            tab_treat1, tab_treat2, tab_treat3 = st.tabs(["药物治疗", "非药物治疗", "生活建议"])

                            with tab_treat1:
                                medications = treatment.get("medications", [])
                                if medications:
                                    for med in medications:
                                        st.markdown(f"""
                                        <div class="result-box">
                                            <b>💊 {med.get('name', '未知药物')}</b><br>
                                            剂量：{med.get('dosage', '未指定')}<br>
                                            频次：{med.get('frequency', '未指定')}<br>
                                            疗程：{med.get('duration', '未指定')}<br>
                                            适应症：{med.get('detail', '未指定')}<br>
                                            ⚠️禁忌：{med.get('dangerous', '未指定')}<br>
                                        </div>
                                        """, unsafe_allow_html=True)
                                else:
                                    st.markdown('<div class="result-box">暂无药物治疗建议</div>',
                                                unsafe_allow_html=True)

                            with tab_treat2:
                                non_med = treatment.get("non_pharmacological", [])
                                if non_med:
                                    for item in non_med:
                                        st.markdown(f'<div class="result-box">• {item}</div>', unsafe_allow_html=True)
                                else:
                                    st.markdown('<div class="result-box">暂无非药物治疗建议</div>',
                                                unsafe_allow_html=True)

                            with tab_treat3:
                                lifestyle = treatment.get("lifestyle_advice", [])
                                if lifestyle:
                                    for advice in lifestyle:
                                        st.markdown(f'<div class="result-box">• {advice}</div>', unsafe_allow_html=True)
                                else:
                                    st.markdown('<div class="result-box">暂无生活方式建议</div>',
                                                unsafe_allow_html=True)

                        # 随访计划
                        st.markdown('<h3>📅 随访计划</h3>', unsafe_allow_html=True)
                        follow_up = final_result.get("follow_up_plan", {})
                        if follow_up:
                            col_fu1, col_fu2 = st.columns(2)
                            with col_fu1:
                                if follow_up.get("next_appointment"):
                                    st.markdown(f"""
                                    <div class="result-box">
                                        <b>下次预约时间：</b>{follow_up.get("next_appointment", "未指定")}
                                    </div>
                                    """, unsafe_allow_html=True)

                                if follow_up.get("monitoring_symptoms"):
                                    st.markdown('<b>需要观察的症状：</b>', unsafe_allow_html=True)
                                    for symptom in follow_up.get("monitoring_symptoms", []):
                                        st.markdown(f'<div class="result-box">• {symptom}</div>',
                                                    unsafe_allow_html=True)

                            with col_fu2:
                                if follow_up.get("emergency_indicators"):
                                    st.markdown('<b>⚠️ 紧急情况指标：</b>', unsafe_allow_html=True)
                                    for indicator in follow_up.get("emergency_indicators", []):
                                        st.markdown(
                                            f'<div class="result-box" style="border-color: #FF7D00;">• {indicator}</div>',
                                            unsafe_allow_html=True)

                        # 完整报告
                        if final_result.get("final_report"):
                            with st.expander("📄 查看完整医疗报告", expanded=False):
                                st.markdown(f'<div class="result-box">{final_result.get("final_report", "")}</div>',
                                            unsafe_allow_html=True)

                        # 成功提示
                except Exception as e:
                    st.markdown(
                        f'<div class="result-box"><p class="error-text">❌ 诊断过程中出现异常: {str(e)}</p></div>',
                        unsafe_allow_html=True)
                    # 打印详细的错误信息用于调试
                    st.markdown(
                        f'<div class="result-box"><p class="error-text">错误详情: {type(e).__name__}: {str(e)}</p></div>',
                        unsafe_allow_html=True)
