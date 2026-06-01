from langgraph.graph import StateGraph, END

from src.schemas.state import EmailState
from src.agents.technical.analyse import analyse_technical
from src.agents.technical.classify import classify_technical

workflow = StateGraph(EmailState)

workflow.add_node("analyse_technical", analyse_technical)
workflow.add_node("classify_technical", classify_technical)

workflow.set_entry_point("analyse_technical")

workflow.add_edge("analyse_technical", "classify_technical")
workflow.add_edge("classify_technical", END)

app = workflow.compile()
