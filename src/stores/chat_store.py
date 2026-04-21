import json
import os
from typing import List, Dict, Any
from .base import BaseStore

class ChatStore(BaseStore):
    """对话历史存储（JSON文件持久化）"""

    def __init__(self, persist_dir: str = "./stores/data"):
        self.persist_dir = persist_dir
        self.persist_path = os.path.join(persist_dir, "chats.json")
        os.makedirs(persist_dir, exist_ok=True)
        self._load()

    def _load(self):
        if os.path.exists(self.persist_path):
            with open(self.persist_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        else:
            self.data = {}

    def _save(self):
        with open(self.persist_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def get(self, session_id: str) -> Optional[List[Dict[str, Any]]]:
        return self.data.get(session_id)

    def set(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        self.data[session_id] = messages
        self._save()

    def append(self, session_id: str, message: Dict[str, Any]) -> None:
        """追加单条消息"""
        if session_id not in self.data:
            self.data[session_id] = []
        self.data[session_id].append(message)
        self._save()

    def update(self, session_id: str, updates: Dict[str, Any]) -> None:
        pass  # 对话历史不支持修改，用append即可

    def delete(self, session_id: str) -> None:
        if session_id in self.data:
            del self.data[session_id]
            self._save()