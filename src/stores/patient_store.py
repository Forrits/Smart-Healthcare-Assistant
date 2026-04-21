import json
import os
from typing import Optional, Dict, Any
from .base import BaseStore

class PatientStore(BaseStore):
    """患者长期信息存储（JSON文件持久化）"""

    def __init__(self, persist_dir: str = "./stores/data"):
        self.persist_dir = persist_dir
        self.persist_path = os.path.join(persist_dir, "patients.json")
        os.makedirs(persist_dir, exist_ok=True)
        self._load()

    def _load(self):
        """从文件加载数据"""
        if os.path.exists(self.persist_path):
            with open(self.persist_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        else:
            self.data = {}

    def _save(self):
        """保存数据到文件"""
        with open(self.persist_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def get(self, patient_id: str) -> Optional[Dict[str, Any]]:
        return self.data.get(patient_id)

    def set(self, patient_id: str, patient_info: Dict[str, Any]) -> None:
        self.data[patient_id] = patient_info
        self._save()

    def update(self, patient_id: str, updates: Dict[str, Any], append_fields: list = None) -> None:
        """更新患者信息，支持追加字段（如病史）"""
        append_fields = append_fields or []
        patient = self.data.get(patient_id, {})
        for k, v in updates.items():
            if k in append_fields and patient.get(k):
                patient[k] = f"{patient[k]}；{v}"
            else:
                patient[k] = v
        self.data[patient_id] = patient
        self._save()

    def delete(self, patient_id: str) -> None:
        if patient_id in self.data:
            del self.data[patient_id]
            self._save()