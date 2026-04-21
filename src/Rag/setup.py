import os
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever

# ==========================
# 全局单例（只初始化一次）
# ==========================
VECTORSTORE: Chroma | None = None
FINAL_RETRIEVER = None

# 配置
PERSIST_DIRECTORY = os.path.abspath("./chroma_medical_db")
KNOWLEDGE_FOLDER = r"D:\agent-project\new_assistant\src\Rag\Local_file"

# ==========================
# 初始化 RAG（自动加载 TXT + PDF）
# ==========================
def init_medical_rag():
    global VECTORSTORE, FINAL_RETRIEVER

    if FINAL_RETRIEVER is not None:
        print("✅ RAG 已初始化（关键词+向量+重排序）")
        return True

    try:
        print("🔧 初始化医学 RAG：TXT+PDF + 关键词 + 向量 + 重排序...")

        # 1. 嵌入模型
        embeddings = OpenAIEmbeddings(
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            base_url="https://api.siliconflow.cn/v1",
            model="BAAI/bge-large-zh-v1.5",
        )

        all_docs = []

        # 2. 加载或重建向量库
        if os.path.exists(PERSIST_DIRECTORY):
            print("📂 加载已存在的向量库")
            VECTORSTORE = Chroma(
                persist_directory=PERSIST_DIRECTORY,
                embedding_function=embeddings,
            )
            all_docs = VECTORSTORE.get()["documents"]
        else:
            print("📝 未找到向量库，开始从【整个文件夹】重建（自动加载）...")

            if not os.path.exists(KNOWLEDGE_FOLDER):
                print(f"❌ 知识库文件夹不存在：{KNOWLEDGE_FOLDER}")
                return False

            # ------------------------------
            # 🔥 核心：自动加载所有文件
            # ------------------------------
            docs = []

    
            txt_loader = DirectoryLoader(
                KNOWLEDGE_FOLDER, glob="*.txt",
                loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}
            )
            docs.extend(txt_loader.load())

       
            pdf_loader = DirectoryLoader(
                KNOWLEDGE_FOLDER, glob="*.pdf",
                loader_cls=PyPDFLoader
            )
            docs.extend(pdf_loader.load())

            print(f"✅ 共加载 {len(docs)} 个文件（TXT + PDF）")

            # 文本切分
            splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
            splits = splitter.split_documents(docs)

            # 构建向量库
            VECTORSTORE = Chroma.from_documents(
                documents=splits,
                embedding_function=embeddings,
                persist_directory=PERSIST_DIRECTORY,
            )
            all_docs = [d.page_content for d in splits]

        # ------------------------------
        # 3. 关键词 + 向量 混合检索
        # ------------------------------
        vector_ret = VECTORSTORE.as_retriever(search_kwargs={"k": 10})
        bm25_ret = BM25Retriever.from_texts(all_docs)
        bm25_ret.k = 10

        ensemble_ret = EnsembleRetriever(
            retrievers=[bm25_ret, vector_ret],
            weights=[0.3, 0.7]
        )

        # ------------------------------
        # 4. 重排序（最强 Reranker）
        # ------------------------------
        reranker = CrossEncoderReranker(
            model=HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-large"),
            top_n=3
        )

        FINAL_RETRIEVER = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=ensemble_ret
        )

        print("✅ RAG 初始化完成!")
        return True

    except Exception as e:
        print(f"❌ RAG 初始化失败：{str(e)}")
        return False


# ==========================
# 对外调用接口（不变）
# ==========================
def medical_rag_search(query: str):
    if not FINAL_RETRIEVER:
        return []
    docs = FINAL_RETRIEVER.invoke(query)
    return [doc.page_content for doc in docs]


if __name__ == "__main__":
    init_medical_rag()
