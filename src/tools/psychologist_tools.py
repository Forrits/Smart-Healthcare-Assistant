# src/tools/psychologist_tools.py

from langchain_core.tools import tool
from typing import Optional


@tool
def perform_phq9_assessment(answers: list[int]) -> dict:
    """
    执行 PHQ-9 (病人健康问卷-9) 抑郁筛查。

    参数说明：
    - answers: 一个包含 9 个整数的列表，代表用户对 9 个问题的评分。
      评分标准：0=完全不会，1=好几天，2=一半以上天数，3=几乎每天。

    返回结果：
    - 包含总分、严重程度分级（如“中度抑郁”）及就医建议。
    """
    if len(answers) != 9:
        return {"error": "PHQ-9 需要 9 个问题的答案"}

    score = sum(answers)
    severity = ""
    advice = ""

    if score < 5:
        severity = "无明显抑郁症状"
        advice = "保持当前良好的心理状态，注意劳逸结合。"
    elif score < 10:
        severity = "轻度抑郁"
        advice = "建议尝试运动、冥想或与人倾诉。如果持续两周以上无改善，请咨询医生。"
    elif score < 15:
        severity = "中度抑郁"
        advice = "建议寻求专业心理咨询师的帮助，或前往医院心理科就诊。"
    elif score < 20:
        severity = "中重度抑郁"
        advice = "强烈建议前往医院精神科或心理科进行专业诊断和治疗。"
    else:
        severity = "重度抑郁"
        advice = "请务必尽快前往医院就诊，不要独自承受，必要时请拨打心理援助热线。"

    return {
        "score": score,
        "severity": severity,
        "advice": advice
    }


@tool
def provide_emotional_support(emotion_type: str, user_statement: str) -> str:
    """
    提供即时的情绪安抚和支持性心理治疗。

    参数说明：
    - emotion_type: 用户当前的主要情绪（如：焦虑、悲伤、愤怒、孤独）。
    - user_statement: 用户的具体描述。

    功能：
    - 调用此工具会触发共情回应，使用积极倾听技术，让用户感到被理解。
    """
    # 这里只是模拟返回，实际逻辑由 LLM 在 Prompt 指导下生成，
    # 工具主要用于标记“正在进行心理干预”这一动作。
    return f"正在针对 {emotion_type} 情绪提供心理支持..."
psychologist_tools=[
    perform_phq9_assessment,
    provide_emotional_support
]