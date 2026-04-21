import os
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# ==========================
# 全局单例（只初始化一次）
# ==========================
VECTORSTORE: Chroma | None = None
RETRIEVER = None

# 配置（你自己改）
PERSIST_DIRECTORY = os.path.abspath("./chroma_medical_db")
KNOWLEDGE_FILE = os.path.abspath(r"D:\agent-project\new_assistant\src\Rag\Local_file\a.txt")


def init_medical_rag():
    """
    初始化医学科普 RAG 向量库
    【全局只调用一次】
    """
    global VECTORSTORE, RETRIEVER

    # 已经初始化过，直接返回
    if VECTORSTORE is not None and RETRIEVER is not None:
        print("✅ RAG 已初始化，跳过")
        # 加载完成后打印向量库信息
        docs = VECTORSTORE.get()
        print("==== 向量库总条数 ====", len(docs["ids"]))
        print("==== 第一条内容 ====")
        if docs["documents"]:
            print(docs["documents"][0])
        return True

    try:
        print("🔧 开始初始化医学科普 RAG...")

        # 1. 嵌入模型
        embeddings = OpenAIEmbeddings(
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            base_url="https://api.siliconflow.cn/v1",
            model="BAAI/bge-large-zh-v1.5",
        )

        # 2. 如果已有持久化向量库 → 直接加载
        if os.path.exists(PERSIST_DIRECTORY):
            print("📂 加载已存在的向量库...")
            VECTORSTORE = Chroma(
                persist_directory=PERSIST_DIRECTORY,
                embedding_function=embeddings,
            )
            RETRIEVER = VECTORSTORE.as_retriever(search_kwargs={"k": 3})
            print("✅ 向量库加载完成")

            # 🔥 加载完成后打印向量库信息
            docs = VECTORSTORE.get()
            print("==== 向量库总条数 ====", len(docs["ids"]))
            # print("==== 第一条内容 ====")
            # if docs["documents"]:
            #     print(docs["documents"][0])
            return True

        # 3. 不存在 → 从文档重建
        print("📝 未找到向量库，开始从文档重建...")
        if not os.path.exists(KNOWLEDGE_FILE):
            print(f"❌ 知识库文件不存在：{KNOWLEDGE_FILE}")
            return False

        # 加载 + 切分
        loader = TextLoader(KNOWLEDGE_FILE, encoding="utf-8")
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
        splits = splitter.split_documents(docs)

        # 构建向量库（Chroma 0.5.x+ 自动持久化，无需手动persist）
        VECTORSTORE = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY,
        )
        # 👇 彻底移除废弃的 persist() 方法
        # VECTORSTORE.persist()
        RETRIEVER = VECTORSTORE.as_retriever(search_kwargs={"k": 3})

        print("✅ 医学科普 RAG 初始化完成！")

        # 🔥 重建完成后打印向量库信息
        # docs = VECTORSTORE.get()
        # print("==== 向量库总条数 ====", len(docs["ids"]))
        # print("==== 第一条内容 ====")
        # if docs["documents"]:
        #     print(docs["documents"][0])
        # return True

    except Exception as e:
        print(f"❌ RAG 初始化失败：{str(e)}")
        return False


# ==========================
# 【启动入口】直接运行文件即可测试
# ==========================
if __name__ == "__main__":
    init_medical_rag()
