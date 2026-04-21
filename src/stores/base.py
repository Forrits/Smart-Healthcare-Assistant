from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List

class BaseStore(ABC):
    """所有存储类的抽象基类"""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """根据 key 获取数据"""
        pass

    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """设置 key-value 数据"""
        pass

    @abstractmethod
    def update(self, key: str, updates: Dict[str, Any]) -> None:
        """更新嵌套字典数据"""
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """删除 key 数据"""
        pass