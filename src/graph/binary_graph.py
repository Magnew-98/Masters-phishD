from langgraph.graph import StateGraph, END

from src.schemas.state import EmailState
from src.agents.binary.analyse import analyse_email
from src.agents.binary.classify import classify_email

workflow = StateGraph(EmailState)

workflow.add_node("analyse", analyse_email)
workflow.add_node("classify", classify_email)

workflow.set_entry_point("analyse")

workflow.add_edge("analyse", "classify")
workflow.add_edge("classify", END)

app = workflow.compile()