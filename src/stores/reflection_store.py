import json
import os
from typing import Optional, Dict, Any, List
from .base import BaseStore

class ReflectionStore(BaseStore):
    """反思与评审记录存储"""

    def __init__(self, persist_dir: str = "./stores/data"):
        self.persist_dir = persist_dir
        self.persist_path = os.path.join(persist_dir, "reflections.json")
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

    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        return self.data.get(task_id)

    def set(self, task_id: str, reflection: Dict[str, Any]) -> None:
        self.data[task_id] = reflection
        self._save()

    def add_review(self, task_id: str, critic_opinion: str, is_pass: bool) -> None:
        """添加挑刺评审记录"""
        if task_id not in self.data:
            self.data[task_id] = {"reflections": [], "reviews": []}
        self.data[task_id]["reviews"].append({
            "opinion": critic_opinion,
            "is_pass": is_pass
        })
        self._save()

    def update(self, key: str, updates: Dict[str, Any]) -> None: pass
    def delete(self, key: str) -> None: pass