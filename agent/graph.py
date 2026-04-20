from langgraph.graph import StateGraph, START,END
from langgraph.checkpoint.memory import MemorySaver
from agent.state import AgentState
from agent.nodes.input_collector import input_collector_node
from agent.nodes.profile_structurer import profile_structurer_node
from agent.nodes.metrics_calculator import metrics_calculator_node
from agent.nodes.query_builder import query_builder_node
from agent.nodes.diet_generator import diet_generator_node
from agent.nodes.output_formatter import output_formatter_node

def route_after_profile(state: AgentState) -> str:
    if state.get("is_profile_complete"):
        return "proceed"
    return "collect"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("profile_structurer", profile_structurer_node)
    graph.add_node("input_collector", input_collector_node)
    graph.add_node("metrics_calculator", metrics_calculator_node)
    graph.add_node("query_builder", query_builder_node)
    graph.add_node("diet_generator", diet_generator_node)
    graph.add_node("output_formatter", output_formatter_node)


    graph.add_edge(START, "profile_structurer")
    graph.add_conditional_edges("profile_structurer", route_after_profile,{
        "collect": "input_collector",
        "proceed": "metrics_calculator"
    }
    )
    graph.add_edge("input_collector", END)
    graph.add_edge("metrics_calculator", "query_builder")
    graph.add_edge("query_builder", "diet_generator")
    graph.add_edge("diet_generator", "output_formatter")
    graph.add_edge("output_formatter", END)


    memory = MemorySaver()
    return graph.compile(checkpointer=memory)

agent_graph = build_graph()
