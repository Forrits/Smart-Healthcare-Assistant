
from src.Rag.setup import RETRIEVER
from langchain_core.tools import tool
import json
@tool
def local_knowledge_search(query: str) -> str:
    """
    使用本地知识库进行搜索。
    参数：
        query: 查询内容。
    返回：
        JSON 格式的搜索结果。
    """
    try:
        docs = RETRIEVER.invoke(query)
        if not docs:
            return json.dumps({
                "success": True,
                "query": query,
                "content": None,
                "error": None
            }, ensure_ascii=False)

        content = "\n".join([d.page_content for d in docs])
        return json.dumps({
            "success": True,
            "query": query,
            "content": content,
            "error": None
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "success": False,
            "query": query,
            "content": None,
            "error": str(e)
        }, ensure_ascii=False)
# 导出工具列表
medical_tutor_tools = [local_knowledge_search]