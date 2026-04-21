from langchain_core.tools import tool
from typing import Optional


# 1. 记笔记工具
@tool
def update_patient_record(
        age: str = None,
        gender: str = None,
        symptoms: str = None,
        duration: str = None,
        severity: str = None,
        history: str = None,
        Supplement: Optional[str] = None
):
    """更新患者的基本信息和补充信息。只更新传入的字段，未传入的字段保持不变。"""

    print("使用了工具去更新患者信息")
    updates = {}
    if age: updates['age'] = age
    if gender: updates['gender'] = gender
    if symptoms: updates['symptoms'] = symptoms
    if duration: updates['duration'] = duration
    if severity: updates['severity'] = severity
    if history: updates['history'] = history
    if Supplement: updates['Supplement'] = Supplement
    print(f"工具里要更新的字段: {updates}")

    return {"patient_info_updates": updates, "_append_fields": ["Supplement"]}


@tool
def order_lab_test(
        test_name: str,
        reason: str
):
    """
    开具检验/检查申请单。
    当仅凭问诊无法确诊，需要辅助检查时调用。
    参数:
        test_name: 检查项目名称（如：血常规、胸部CT）。
        reason: 开单理由。
    """

    # 1. 获取当前的检查列表（如果不存在则初始化）

    # 2. 添加新订单
    new_order = {
        "name": test_name,
        "reason": reason,
        "status": "pending"  # 状态：进行中/待出结果
    }


    return {
        "new_lab_order": new_order,
        "waiting_for_result": True
    }






from langchain_openai import ChatOpenAI

import  os
@tool
def make_diagnosis(
        age: str = None,
        gender: str = None,
        symptoms: str = None,
        duration: str = None,
        severity: str = None,
        history: str = None,
) -> str:
    """
    给出最终诊断建议。
    只有在收集完所有必要信息后，才能调用此工具。
    """

    # 1. 数据清洗
    def clean(val):
        return val if val else "未知"

    # 2. 构造 Prompt
    # 重点：在 System Prompt 中强制要求 Markdown 格式
    system_prompt = """
    你是一位经验丰富、严谨且富有同情心的全科医生。
    请根据提供的患者信息，生成一份**Markdown 格式**的诊断报告。

    **输出格式要求**：
    1. 使用 Markdown 语法（如标题、加粗、列表）。
    2. 不要包含任何开场白（如“好的，这是诊断...”），直接输出报告内容。
    3. 结构必须包含以下部分：
       - ### 🩺 初步诊断
       - ### 💊 治疗建议 (包含用药、生活护理)
       - ### ⚠️ 风险提示 (何时需要去医院)
    """

    user_content = f"""
    ### 患者档案
    - **年龄**：{clean(age)}
    - **性别**：{clean(gender)}
    - **症状**：{clean(symptoms)}
    - **时长**：{clean(duration)}
    - **程度**：{clean(severity)}
    - **病史**：{clean(history)}
    """

    # 3. 调用大模型
    # 注意：这里需要你根据实际使用的 LLM 库来调整代码
    # 以下是 LangChain 的标准调用方式示例：
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
        temperature=0.3
    )
    try:
       # 假设你有一个全局的 llm 对象
        messages = [
            ("system", system_prompt),
            ("human", user_content)
        ]
        response = llm.invoke(messages)
        return response.content

        # --- 下面是模拟返回，请替换为上面的真实调用 ---
        mock_markdown = f"""
### 🩺 初步诊断

根据您描述的症状（{clean(symptoms)}）以及病程（{clean(duration)}），结合{clean(age)}{clean(gender)}的生理特点，初步判断为 **上呼吸道感染（普通感冒）** 的可能性较大。

### 💊 治疗建议

1.  **生活护理**：
    -   多喝温水，保持充足睡眠。
    -   饮食清淡，避免辛辣刺激。
2.  **用药建议**（仅供参考，请阅读说明书）：
    -   如发热超过38.5℃，可服用对乙酰氨基酚。
    -   针对鼻塞流涕，可使用生理盐水洗鼻。

### ⚠️ 风险提示

如果出现以下情况，请**立即前往医院**：
-   持续高热不退超过3天。
-   出现呼吸困难或胸痛。
-   精神状态明显变差。
"""
        return mock_markdown

    except Exception as e:
        return f"诊断生成失败：{str(e)}"


@tool
def update_allergy_history(
    drug_allergy: Optional[str] = None,
    food_allergy: Optional[str] = None,
    other_allergy: Optional[str] = None
):
    """
    记录患者过敏史，包括药物过敏、食物过敏、其他过敏。
    用于安全用药、避免医疗风险。
    """
    print("使用工具：更新过敏史")
    updates = {}
    if drug_allergy: updates["drug_allergy"] = drug_allergy
    if food_allergy: updates["food_allergy"] = food_allergy
    if other_allergy: updates["other_allergy"] = other_allergy

    return {
        "patient_info_updates": updates,
        "_append_fields": ["drug_allergy", "food_allergy", "other_allergy"]
    }


@tool
def update_medical_history(
    chronic_disease: Optional[str] = None,
    surgery_history: Optional[str] = None,
    infectious_history: Optional[str] = None
):
    """
    更新既往病史：
    - 慢性病（高血压、糖尿病等）
    - 手术史
    - 传染病史
    """
    print("使用工具：更新既往病史")
    updates = {}
    if chronic_disease: updates["chronic_disease"] = chronic_disease
    if surgery_history: updates["surgery_history"] = surgery_history
    if infectious_history: updates["infectious_history"] = infectious_history

    return {
        "patient_info_updates": updates,
        "_append_fields": ["chronic_disease", "surgery_history", "infectious_history"]
    }


@tool
def add_medical_advice(
    advice_title: str,
    advice_content: str
):
    """
    添加医嘱，如饮食禁忌、复查提醒、注意事项、护理方式等。
    用于治疗方案补充。
    """
    print("使用工具：添加医嘱")
    return {
        "advice": {
            "title": advice_title,
            "content": advice_content
        }
    }
@tool
def prescribe_medication(
    drug_name: str,
    usage: str,
    dosage: str,
    frequency: str,
    duration: str,
    notes: Optional[str] = None
):
    """
    开具正式用药处方，包含药名、用法、剂量、频次、疗程、注意事项。
    """
    print("使用工具：开具用药处方")
    prescription = {
        "drug_name": drug_name,
        "usage": usage,
        "dosage": dosage,
        "frequency": frequency,
        "duration": duration,
        "notes": notes or "无特殊说明"
    }
    return {
        "prescription": prescription
    }


# 把它们打包
doctor_tools = [update_patient_record, make_diagnosis, order_lab_test,add_medical_advice,prescribe_medication]
