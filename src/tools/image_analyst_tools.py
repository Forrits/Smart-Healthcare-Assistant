from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="Qwen/Qwen3.5-122B-A10B",
    api_key="sk-nnbtrcocdpjbbksvonxsgorqgigqsckefstlztcmboafdvwx",
    base_url="https://api.siliconflow.cn/v1",
    temperature=0.3
)

@tool
def analyze_x_ray_tool(image_base64: str) -> str:
    """专门用于分析 X 光片、CT 或骨骼影像。"""

    print("图片分析工具使用啦！！！！！！")
    response = llm.invoke([
        {"role": "user", "content": [
            {"type": "text", "text": "请专业分析这张医学 X 光片，描述影像表现、是否异常、骨折、炎症或结节。"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
        ]}
    ])
    return f"【X 光影像分析结果】\n{response.content}"

@tool
def analyze_skin_tool(image_base64: str) -> str:
    """专门用于分析皮肤病变、外伤照片。"""
    # 🔥 修改：移除硬编码返回，改为真实调用视觉模型
    response = llm.invoke([
        {"role": "user", "content": [
            {"type": "text", "text": "请专业分析这张皮肤病变照片，描述颜色、形状、边缘、大小及可能的症状。"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
        ]}
    ])
    return f"【皮肤影像分析结果】\n{response.content}"

image_analysis_tools = [analyze_x_ray_tool, analyze_skin_tool]
