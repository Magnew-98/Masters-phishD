from langgraph.graph import StateGraph, END

from src.schemas.state import EmailState
from src.agents.rag.retrieve import rag_retrieve
from src.agents.binary.analyse import analyse_email
from src.agents.technical.analyse import analyse_technical
from src.agents.coordinator.classify import coordinate

workflow = StateGraph(EmailState)

workflow.add_node("rag_retrieve", rag_retrieve)
workflow.add_node("analyse", analyse_email)
workflow.add_node("analyse_technical", analyse_technical)
workflow.add_node("coordinate", coordinate)

workflow.set_entry_point("rag_retrieve")

workflow.add_edge("rag_retrieve", "analyse")
workflow.add_edge("analyse", "analyse_technical")
workflow.add_edge("analyse_technical", "coordinate")
workflow.add_edge("coordinate", END)

app = workflow.compile()
