from langchain_core.tools import tool
from typing import Optional, List, Dict
import datetime


# ==============================================
# 1. PHQ-9 抑郁评估（你原来的，我优化了返回）
# ==============================================
@tool
def perform_phq9_assessment(answers: list[int]) -> dict:
    """
    执行 PHQ-9 抑郁筛查。
    参数：answers = [0-3, 0-3, ...] 共9项
    0=完全不会 1=好几天 2=一半以上天数 3=几乎每天
    """
    if len(answers) != 9:
        return {"error": "需要提供9个问题的答案"}

    score = sum(answers)
    severity = ""
    advice = ""

    if score < 5:
        severity = "无明显抑郁症状"
        advice = "心理状态良好，继续保持规律作息与社交活动。"
    elif score < 10:
        severity = "轻度抑郁"
        advice = "可通过运动、倾诉、兴趣爱好调节；持续两周建议咨询专业人士。"
    elif score < 15:
        severity = "中度抑郁"
        advice = "建议尽快寻求心理咨询师或医院心理科评估。"
    elif score < 20:
        severity = "中重度抑郁"
        advice = "强烈建议前往精神科就诊，需要专业干预。"
    else:
        severity = "重度抑郁"
        advice = "请立即就医，必要时联系心理危机干预热线。"

    return {
        "phq9_score": score,
        "depression_severity": severity,
        "professional_advice": advice,
        "assessment_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    }


# ==============================================
# 2. 情绪支持（你原来的）
# ==============================================
@tool
def provide_emotional_support(emotion_type: str, user_content: str) -> dict:
    """
    对焦虑、悲伤、愤怒、孤独、压力等情绪提供共情支持
    """
    return {
        "action": "情绪安抚",
        "emotion": emotion_type,
        "suggestion": "我理解你的感受，我会在这里陪伴你慢慢梳理情绪。"
    }


# ==============================================
# 3. GAD-7 焦虑量表（新增！超级常用）
# ==============================================
@tool
def perform_gad7_assessment(answers: list[int]) -> dict:
    """
    GAD-7 广泛性焦虑量表评估
    输入7个分数：0=完全不会 1=几天 2=一半以上 3=几乎每天
    """
    if len(answers) != 7:
        return {"error": "需要7个答案"}
    score = sum(answers)

    if score < 5:
        severity = "无明显焦虑"
    elif score < 10:
        severity = "轻度焦虑"
    elif score < 15:
        severity = "中度焦虑"
    else:
        severity = "重度焦虑"

    return {
        "gad7_score": score,
        "anxiety_severity": severity,
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    }


# ==============================================
# 4. 情绪记录（心理咨询必备）
# ==============================================
@tool
def record_user_mood(
    emotion: str,
    intensity: int,
    trigger_event: str,
    note: Optional[str] = ""
) -> dict:
    """
    记录用户当下情绪、强度、触发事件
    intensity: 1-10分
    """
    return {
        "record_type": "情绪日记",
        "emotion": emotion,
        "intensity": intensity,
        "trigger": trigger_event,
        "note": note,
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    }


# ==============================================
# 5. 压力评分（专业）
# ==============================================
@tool
def assess_stress_level(stress_score: int) -> dict:
    """
    压力水平评估 0-10分
    """
    if stress_score <= 3:
        level = "低压力"
    elif stress_score <= 6:
        level = "中等压力"
    else:
        level = "高压力"

    return {
        "stress_score": stress_score,
        "stress_level": level,
        "suggestion": "可通过呼吸训练、冥想、运动缓解压力"
    }


# ==============================================
# 6. 睡眠质量评估
# ==============================================
@tool
def assess_sleep_quality(
    sleep_hours: float,
    difficulty_falling_asleep: bool,
    wake_up_early: bool
) -> dict:
    """
    评估睡眠问题：失眠、入睡困难、早醒
    """
    issues = []
    if sleep_hours < 6:
        issues.append("睡眠时间不足")
    if difficulty_falling_asleep:
        issues.append("入睡困难")
    if wake_up_early:
        issues.append("早醒")

    return {
        "sleep_hours": sleep_hours,
        "sleep_issues": issues if issues else ["睡眠基本正常"],
        "advice": "保持规律作息，睡前减少电子产品使用"
    }


# ==============================================
# 7. 正念/呼吸引导（干预工具）
# ==============================================
@tool
def guided_mindfulness_practice(duration_min: int) -> dict:
    """
    提供正念呼吸引导
    """
    return {
        "guide": f"请安静坐下，用鼻子缓慢呼吸，专注感受气息进出，持续{duration_min}分钟。",
        "effect": "缓解焦虑、平复情绪、集中注意力"
    }


# ==============================================
# 8. 认知调整（CBT 技术）
# ==============================================
@tool
def cognitive_restructuring(negative_thought: str) -> dict:
    """
    CBT 认知重构：识别负面思维 → 引导理性思考
    """
    return {
        "analysis": "正在识别不合理认知...",
        "suggestion": "我们一起看看这个想法是否符合事实，有没有更温和的解释方式。"
    }


# ==============================================
# 9. 社会支持评估
# ==============================================
@tool
def assess_social_support(
    has_family_support: bool,
    has_friend_support: bool,
    feel_lonely: bool
) -> dict:
    """
    评估用户社会支持系统：家人、朋友、孤独感
    """
    support_score = 0
    if has_family_support: support_score +=1
    if has_friend_support: support_score +=1

    return {
        "support_score": support_score,
        "feel_lonely": feel_lonely,
        "advice": "适当向信任的人表达感受，能有效改善心理状态"
    }


# ==============================================
# 10. 危机筛查（安全必备）
# ==============================================
@tool
def suicide_risk_screening(
    has_suicidal_thought: bool,
    has_plan: bool,
    has_intent: bool
) -> dict:
    """
    心理危机筛查：自伤/自杀风险
    """
    risk = "低风险"
    if has_suicidal_thought:
        risk = "中风险"
    if has_plan or has_intent:
        risk = "高风险"

    return {
        "risk_level": risk,
        "warning": "如风险较高，请立即联系心理危机干预热线或前往急诊",
        "hotline": "全国心理危机干预热线：400-161-9995 或 010-82951332"
    }


# ==============================================
# 最终工具列表（10个专业工具）
# ==============================================
psychologist_tools = [
    perform_phq9_assessment,
    perform_gad7_assessment,
    provide_emotional_support,
    record_user_mood,
    assess_stress_level,
    assess_sleep_quality,
    guided_mindfulness_practice,
    cognitive_restructuring,
    assess_social_support,
    suicide_risk_screening
]
