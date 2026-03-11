import os
import json
import warnings
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    JSONLoader,
    PyMuPDFLoader  # 更稳定的PDF加载器
)
from langchain_chroma import Chroma

# 加载环境变量
load_dotenv()

# 忽略PDF解析警告（可选，清理控制台）
warnings.filterwarnings("ignore", message="Multiple definitions in dictionary")


class MedicalRAG:
    """医疗领域RAG工具类：负责本地医疗文档向量化和检索"""

    def __init__(
            self,
            persist_directory: str = "./chroma_db",
            collection_name: str = "medical_knowledge",
    ):

        # 初始化嵌入模型
        self.embeddings = OpenAIEmbeddings(
            model="BAAI/bge-large-zh-v1.5",
            api_key=API-KEY,
            base_url="https://api.siliconflow.cn/v1/"
        )

        # 核心配置
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # 创建目录（如果不存在）
        os.makedirs(persist_directory, exist_ok=True)

        # 初始化Langchain Chroma（统一入口）
        self.chroma_db = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50,
            separators=["\n\n"],
            length_function=len
        )

    def load_document(self, file_path: str) -> List[Any]:
        """
        加载本地文档并分割（返回Langchain Document对象列表）

        Args:
            file_path: 文件路径（支持txt, pdf, docx, json）

        Returns:
            分割后的文档块列表
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 根据文件扩展名选择加载器
        ext = os.path.splitext(file_path)[1].lower()

        try:
            if ext == ".txt":
                loader = TextLoader(file_path, encoding="utf-8")
            elif ext == ".pdf":
                # 替换为更稳定的PyMuPDFLoader，解决PDF解析报错问题
                loader = PyMuPDFLoader(file_path)
            elif ext == ".docx":
                loader = Docx2txtLoader(file_path)
            elif ext == ".json":
                loader = JSONLoader(
                    file_path=file_path,
                    jq_schema=".[]",
                    text_content=False
                )
            else:
                raise ValueError(f"不支持的文件格式: {ext}")

            # 加载并分割文档
            documents = loader.load()
            split_docs = self.text_splitter.split_documents(documents)

            # 为每个文档块添加元数据
            for i, doc in enumerate(split_docs):
                doc.metadata.update({
                    "file_name": os.path.basename(file_path),
                    "chunk_id": i,
                    "total_chunks": len(split_docs)
                })

            return split_docs

        except Exception as e:
            raise Exception(f"加载文档失败: {str(e)}")

    def add_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        向量化并添加文档到向量数据库（基于langchain_chroma）

        Args:
            file_paths: 文件路径列表

        Returns:
            处理结果统计
        """
        stats = {
            "total_files": len(file_paths),
            "processed_files": 0,
            "total_chunks": 0,
            "failed_files": []
        }

        for file_path in file_paths:
            try:
                # 加载并分割文档
                split_docs = self.load_document(file_path)
                if not split_docs:
                    raise Exception("文档分割后为空")

                # 核心：通过langchain_chroma添加文档（统一接口）
                self.chroma_db.add_documents(documents=split_docs)



                stats["processed_files"] += 1
                stats["total_chunks"] += len(split_docs)

            except Exception as e:
                stats["failed_files"].append({
                    "file": file_path,
                    "error": str(e)
                })

        return stats

    # def similarity_search(
    #         self,
    #         query: str,
    #         k: int = 5,
    #         score_threshold: float = 0.1,  # 调低阈值，解决检索不到结果的问题
    #         filter_metadata: Optional[Dict] = None
    # ) -> List[Dict[str, Any]]:
    #     """
    #     相似性检索（完全基于langchain_chroma）
    #
    #     Args:
    #         query: 查询文本
    #         k: 返回结果数量
    #         score_threshold: 相似度阈值（0-1，越大越相似）
    #         filter_metadata: 元数据过滤条件
    #
    #     Returns:
    #         格式化的检索结果列表
    #     """
    #     # 执行相似性检索（带分数）
    #     results = self.chroma_db.similarity_search_with_score(
    #         query=query,
    #         k=k,
    #         filter=filter_metadata
    #     )
    #
    #     # 过滤低相似度结果并格式化
    #     formatted_results = []
    #     for doc, score in results:
    #         # Chroma距离分数越小越相似，转换为0-1的相似度
    #         similarity = 1 - score
    #         if similarity >= score_threshold:
    #             formatted_results.append({
    #                 "content": doc.page_content,
    #                 "metadata": doc.metadata,
    #                 "similarity_score": round(similarity, 4)
    #             })
    #
    #     return formatted_results
    def similarity_search(
            self,
            query: str,
            k: int = 5,
            score_threshold: float = 0.2,  # 调低阈值，适配医疗文档
            filter_metadata: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        results = self.chroma_db.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter_metadata
        )
        formatted_results = []
        for doc, score in results:
            similarity = score  # ✅ 直接使用score作为余弦相似度
            if similarity >= score_threshold:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": round(similarity, 4)
                })
        return formatted_results
    def get_relevant_knowledge(self, query: str, k: int = 3) -> str:

        results = self.similarity_search(query, k=k)

        if not results:
            return "未找到相关医疗知识。"

        # 格式化结果
        formatted_text = "### 相关医疗知识参考\n"
        for i, result in enumerate(results, 1):
            formatted_text += f"\n{i}. {result['content']}\n"
            if "file_name" in result["metadata"]:
                formatted_text += f"   (来源: {result['metadata']['file_name']}, 相似度: {result['similarity_score']:.2%})\n"

        return formatted_text

    def clear_collection(self) -> bool:

        try:
            # 获取所有文档ID并删除
            all_ids = self.chroma_db.get()["ids"]
            if all_ids:
                self.chroma_db.delete(ids=all_ids)
            return True
        except Exception as e:
            print(f"清空集合失败: {str(e)}")
            return False

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息（基于langchain_chroma）

        Returns:
            统计信息
        """
        # 通过langchain_chroma获取集合数据
        collection_data = self.chroma_db.get()
        return {
            "collection_name": self.collection_name,
            "total_documents": len(collection_data["ids"]),  # 文档数量=ID数量
            "persist_directory": self.persist_directory
        }

    def get_all_documents(self, include_embeddings: bool = False) -> Dict[str, Any]:
        """
        获取向量数据库中的所有文档内容
        """
        try:
            all_data = self.chroma_db.get(
                include=["documents", "metadatas"] + (["embeddings"] if include_embeddings else [])
            )
            formatted_docs = []
            for idx, doc_id in enumerate(all_data["ids"]):
                doc_info = {
                    "document_id": doc_id,
                    "content": all_data["documents"][idx],
                    "metadata": all_data["metadatas"][idx] if all_data["metadatas"] else {}
                }
                if include_embeddings and all_data["embeddings"]:
                    doc_info["embedding"] = all_data["embeddings"][idx][:5] + ["..."]
                formatted_docs.append(doc_info)
            return {
                "total_documents": len(formatted_docs),
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory,
                "documents": formatted_docs
            }
        except Exception as e:
            raise Exception(f"获取所有文档失败: {str(e)}")

    def print_all_documents(self, include_embeddings: bool = False) -> None:
        """
        友好打印向量数据库所有内容
        """
        try:
            all_docs = self.get_all_documents(include_embeddings=include_embeddings)
            print(f"\n=== 向量数据库内容概览 ===")
            print(f"集合名称: {all_docs['collection_name']}")
            print(f"存储路径: {all_docs['persist_directory']}")
            print(f"总文档块数量: {all_docs['total_documents']}\n")
            if all_docs["total_documents"] == 0:
                print("⚠️  向量数据库为空")
                return
            for i, doc in enumerate(all_docs["documents"], 1):
                print(f"--- 文档块 {i} ---")
                print(f"ID: {doc['document_id']}")
                print(f"来源文件: {doc['metadata'].get('file_name', '未知')}")
                print(f"块ID: {doc['metadata'].get('chunk_id', '未知')}")
                print(f"内容: {doc['content'][:200]}..." if len(doc['content']) > 200 else f"内容: {doc['content']}")
                print()
        except Exception as e:
            print(f"打印所有文档失败: {str(e)}")


# 便捷函数
def init_medical_rag() -> MedicalRAG:
    """初始化医疗RAG实例"""
    return MedicalRAG()


def load_medical_documents(file_paths: List[str]) -> Dict[str, Any]:
    """
    加载医疗文档到向量数据库

    Args:
        file_paths: 文件路径列表

    Returns:
        处理结果
    """
    rag = init_medical_rag()
    return rag.add_documents(file_paths)


def retrieve_medical_knowledge(query: str, k: int = 3) -> str:
    """
    检索医疗知识

    Args:
        query: 查询文本
        k: 返回结果数量

    Returns:
        格式化的知识文本
    """
    rag = init_medical_rag()
    return rag.get_relevant_knowledge(query, k=k)




# 测试代码
if __name__ == "__main__":
    # 初始化RAG
    rag = MedicalRAG()
    #示例：添加文档（取消注释并替换为你的实际文件路径）
    # file_paths = [
    #     r"D:\python_project\medical-diagnosis-assistant-main\medical-diagnosis-assistant-main\Local_file\头痛.txt"]
    #
    # # 仅当文件存在时才执行添加操作
    # valid_files = [f for f in file_paths if os.path.exists(f)]
    # if valid_files:
    #     stats = rag.add_documents(valid_files)
    #     print(f"文档处理统计: {json.dumps(stats, ensure_ascii=False, indent=2)}")
    # else:
    #     print("指定的文件不存在，请检查文件路径")
    #
    # # 示例：检索知识
    # query = "头痛预防"
    # knowledge = rag.get_relevant_knowledge(query)
    # print(knowledge)
    #
    # # 查看集合统计
    # stats = rag.get_collection_stats()
    # print(f"\n集合统计: {json.dumps(stats, ensure_ascii=False, indent=2)}")
    a=rag.print_all_documents()
    b=rag.get_relevant_knowledge("头痛",3)
    print(a)
    print(b)
