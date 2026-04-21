# reflection_config.py
AGENT_REFLECTION_CONFIG = {
    "DOCTOR": {
        "check_items": [
            "症状收集是否完整",
            "病史信息是否缺失",
            "逻辑是否自洽",
            "是否需要进一步问诊",
            "是否需要补充检查"
        ],
        "standard": "临床问诊规范，严谨、完整、安全"
    },
    "MEDICAL_IMAGE_ANALYST": {
        "check_items": [
            "影像描述是否客观",
            "是否遗漏关键异常",
            "结论是否与图片匹配",
            "是否建议进一步检查"
        ],
        "standard": "影像报告规范，准确、客观、不漏诊"
    },
    "PSYCHOLOGIST": {
        "check_items": [
            "情绪状态评估是否完整",
            "是否识别高风险点",
            "建议是否合理"
        ],
        "standard": "心理评估规范，温和、严谨、识别风险"
    },
    "TUTOR": {
        "check_items": [
            "知识是否科学准确",
            "无虚假、无绝对化词语",
            "通俗易懂、无误导"
        ],
        "standard": "医学科普规范，准确、通俗、安全"
    },
    "CHAT": {
        "check_items": ["对话是否友好合规"],
        "standard": "友好、礼貌、安全"
    }
}
