import logging
import sys
from pathlib import Path

# 日志存放目录（自动创建）
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(exist_ok=True)

def get_logger(name: str) -> logging.Logger:
    """
    获取统一格式的日志对象
    输出：控制台 + 文件
    格式：时间 | 日志级别 | 模块名 | 信息
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    # 日志格式
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-6s | %(name)-12s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 1. 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. 文件输出（按天滚动）
    file_handler = logging.FileHandler(
        LOG_DIR / f"{name}.log",
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
