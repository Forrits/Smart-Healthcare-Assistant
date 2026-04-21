from src.graph.state import AgentState

def get_current_task(state: AgentState, agent_name: str):
    """
    统一任务过滤器：获取分配给指定Agent的第一个待执行（pending）任务
    """
    task_list = state.get("task_list", [])
    for task in task_list:
        if task["assigned_agent"] == agent_name and task["status"] == "pending":
            return task
    return None