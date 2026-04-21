# 🏥 模块化医疗问诊多智能体系统（带任务规划与 RAG）

基于 LangGraph + Streamlit 构建的模块化医疗问诊助手，通过“Supervisor 规划 + 多角色 Agent + 工具调用 + 本地医学科普 RAG”的方式，实现从闲聊、科普、临床问诊到影像分析的完整流程。

> ⚠️ 本项目仅用于学习与演示，不构成任何医疗建议或诊断依据。如有健康问题，请及时线下就医。

---

## ✨ 核心特性

- **多智能体协作**：包含医生、医学科普讲师、心理咨询师、医学影像分析师、闲聊助手等角色。
- **Supervisor 任务规划**：根据用户意图动态拆分任务队列，并在多个 Agent 之间调度执行。
- **工具增强能力**：
  - 医生端：更新患者信息、开具检查单、生成结构化诊断报告。
  - 科普端：基于本地知识库的 RAG 检索与通俗解释。
  - 心理端：PHQ-9 抑郁筛查与情绪支持。
  - 影像端：X 光/CT 与皮肤病变的视觉分析。
- **记忆与持久化**：
  - 使用 LangGraph `MemorySaver` 管理对话状态。
  - 自定义 `ChatStore` / `PatientStore` / `ReflectionStore`，以 JSON 文件持久化聊天历史、患者信息与反思记录。
- **RAG 医学科普库**：基于 Chroma + OpenAIEmbeddings（SiliconFlow）构建本地医学知识向量库。
- **Streamlit Web UI**：支持文本对话与图片上传（病历/患处照片），实时展示多轮对话与中间过程。

---

## 🗂 项目结构概览

```text
├── main.py                        # Streamlit 入口 & 与 LangGraph 交互
├── .env                           # API Key 配置
├── chroma_medical_db/             # Chroma 向量库持久化目录
└── src/
    ├── Rag/
    │   ├── setup.py               # RAG 初始化（Chroma + Embedding + Retriever）
    │   └── Local_file/            # 本地医学科普知识源（存放本地文件）
    ├── graph/
    │   ├── state.py               # LangGraph State 定义（AgentState 等）
    │   ├── builder.py             # LangGraph 构图与编译（get_compiled_graph）
    │   ├── supervisor.py          # Supervisor 节点：意图判断 + 任务拆分 + 调度
    │   ├── edges.py               # 路由边逻辑（doctor/tutor/image 等）
    │   └── reflectors/           # 任务结果反思相关（core/config）
    ├── nodes/
    │   ├── agents/
    │   │   ├── doctor.py          # 医生节点：多轮问诊 + 工具调用
    │   │   ├── medical_tutor.py   # 医学科普节点：RAG 检索 + 通俗解释
    │   │   ├── psychologist_node.py # 心理咨询节点：PHQ-9 + 情绪支持
    │   │   ├── image_analyst.py   # 影像分析节点：多模态视觉分析
    │   │   ├── challenger_node.py # 质疑者节点：对诊断进行挑刺与追问
    │   │   └── joker_chat.py      # 闲聊节点：通用聊天
    │   └── memory/
    │       ├── context_filter_node.py # 上下文过滤节点
    │       ├── context_trim_node.py   # 上下文裁剪节点
    │       └── memory_load_node.py    # 记忆加载节点（患者历史信息）
    ├── tools/
    │   ├── doctor_tools.py        # 医生工具：更新病历、开检查单、生成诊断
    │   ├── medical_tutor_tools.py # 科普工具：local_knowledge_search(RAG)
    │   ├── psychologist_tools.py  # 心理工具：PHQ-9 评分、情绪支持
    │   └── image_analyst_tools.py # 影像工具：X光/CT、皮肤病变分析
    ├── stores/
    │   ├── base.py                # Store 基类
    │   ├── chat_store.py          # 聊天历史持久化
    │   ├── patient_store.py       # 患者长期信息持久化
    │   └── reflection_store.py    # 反思与评审记录持久化
    ├── prompts/
    │   ├── supervisor.txt         # Supervisor 路由提示词
    │   ├── plan.txt               # 任务规划提示词
    │   ├── doctor.txt             # 医生问诊提示词
    │   └── joker.txt              # 闲聊提示词
    └── utils/                     # 粗放初始化的LLm
        └── init_llm.py            # 全局 LLM 初始化（DeepSeek）
---

## 🧠 架构说明

### 1. LangGraph State 设计

核心状态定义在 `src/graph/state.py` 中的 `AgentState`：

- **messages**: 对话消息列表（Human/AI/Tool）。
- **patient_info**: 患者基本信息与病史（年龄、性别、症状、时长、严重程度、既往史、补充信息等）。
- **lab_orders**: 检查/检验单列表。
- **debate_history**: 辩论/质疑记录（用于 challenger 等场景）。
- **final_diagnosis**: 最终诊断结论。
- **task_list**: Supervisor 拆分的任务队列，每个任务包含：
  - `task_id`
  - `description`
  - `assigned_agent`（DOCTOR/TUTOR/PSYCHOLOGIST/MEDICAL_IMAGE_ANALYST/CHAT）
  - `status`（pending/completed）
- **next_agent**: 下一个要执行的 Agent 名称。
- **intent**: 当前意图标签（便于调试）。
- **last_executed_agent**: 最近一次执行的 Agent。

### 2. Supervisor 规划与调度

`src/graph/supervisor.py` 中的 `supervisor_node` 负责：

- 读取最近若干轮对话，判断用户新输入是否与上一轮任务相关（`is_intent_related`）。
- 调用 LLM + `plan.txt` 提示词，将用户需求拆分为 `task_list`，并决定 `next_agent`。
- 如果所有任务已完成：
  - 检测到新的用户输入 → 清空任务队列，重新规划。
  - 无新输入 → 结束本轮流程。

### 3. 各 Agent 节点职责

- **doctor_node**（`src/nodes/agents/doctor.py`）：
  - 遵循 `doctor.txt` 提示词进行多轮问诊。
  - 使用 `update_patient_record` 记录患者信息。
  - 必要时调用 `order_lab_test` 开具检查。
  - 信息充分后调用 `make_diagnosis` 生成 Markdown 诊断报告。
  - 支持“退出问诊”指令，提前终止当前医生任务。

- **medical_tutor_node**（`src/nodes/agents/medical_tutor.py`）：
  - 针对分配给 `TUTOR` 的任务，调用 `local_knowledge_search` 从 RAG 中检索医学科普内容。
  - 基于检索结果生成简短、通俗的中文解释。

- **psychologist_node**（`src/nodes/agents/psychologist_node.py`）：
  - 扮演心理咨询师，提供共情式对话。
  - 可使用 PHQ-9 工具进行抑郁筛查，并根据分数给出建议。

- **medical_image_analyst_node**（`src/nodes/agents/image_analyst.py`）：
  - 从最新用户消息中提取文本与 Base64 图片。
  - 结合任务描述，调用多模态 LLM 进行影像分析（X 光/CT/皮肤等）。

- **challenger_node**（`src/nodes/agents/challenger_node.py`）：
  - 阅读主治医生的诊断与患者信息，提出质疑与潜在风险点。
  - 可用于内部“辩论”或审查流程。

- **chat_node**（`src/nodes/agents/joker_chat.py`）：
  - 处理闲聊类任务。
  - 可根据反思结果（problem/suggestion）调整回答风格。

### 4. 工具层（Tools）

- **医生工具**（`src/tools/doctor_tools.py`）：
  - `update_patient_record`: 更新患者档案，支持追加字段（如补充病史）。
  - `order_lab_test`: 创建待完成的检查单。
  - `make_diagnosis`: 汇总患者信息，调用 LLM 生成结构化诊断报告。

- **科普工具**（`src/tools/medical_tutor_tools.py`）：
  - `local_knowledge_search`: 使用 `RETRIEVER` 从 Chroma 中检索相关医学科普片段，返回 JSON。

- **心理工具**（`src/tools/psychologist_tools.py`）：
  - `perform_phq9_assessment`: 计算 PHQ-9 总分、分级与建议。
  - `provide_emotional_support`: 标记正在进行情绪支持干预。

- **影像工具**（`src/tools/image_analyst_tools.py`）：
  - `analyze_x_ray_tool`: 分析 X 光/CT 等影像。
  - `analyze_skin_tool`: 分析皮肤病变照片。

### 5. RAG 医学科普库

- 实现在 `src/Rag/setup.py`：
  - 使用 `langchain_chroma.Chroma` 作为向量库。
  - 使用 SiliconFlow 提供的 OpenAI 兼容 Embedding 接口（模型：`BAAI/bge-large-zh-v1.5`）。
  - 从 `src/Rag/Local_file/a.txt` 加载文本，切分后构建向量索引，持久化到 `chroma_medical_db`。
  - 暴露全局 `RETRIEVER` 供 `local_knowledge_search` 使用。

### 6. 记忆与持久化 Stores

- `ChatStore`：按 session_id 存储聊天历史。
- `PatientStore`：按 patient_id 存储患者长期信息，支持字段追加（如多次就诊的病史累积）。
- `ReflectionStore`：存储任务反思与评审意见，用于后续优化回答质量。

---

## 🚀 快速开始

### 1. 环境准备

- Python 3.9+（推荐 3.10/3.11）
- 安装依赖（示例，具体以你实际使用的包为准）：

