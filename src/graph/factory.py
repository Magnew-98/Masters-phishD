from langgraph.graph import StateGraph, END, START
from src.schemas.state import EmailState

_agents = {
    "binary":    ("analyse",            "src.agents.binary.analyse",     "analyse_email"),
    "technical": ("analyse_technical",  "src.agents.technical.analyse",  "analyse_technical"),
    "sentiment": ("analyse_sentiment",  "src.agents.sentiment.analyse",  "analyse_sentiment"),
    "linguistic": ("analyse_linguistic", "src.agents.linguistic.analyse", "analyse_linguistic"),
}

_classifiers = {
    "binary":    ("classify",           "src.agents.binary.classify",    "classify_email"),
    "technical": ("classify_technical", "src.agents.technical.classify", "classify_technical"),
    "sentiment": ("classify_sentiment", "src.agents.sentiment.classify", "classify_sentiment"),
    "linguistic": ("classify_linguistic", "src.agents.linguistic.classify", "classify_linguistic"),
}


def _load(module_path: str, fn_name: str):
    import importlib
    return getattr(importlib.import_module(module_path), fn_name)


def build_graph(components: list[str], use_rag: bool = False, parallel: bool = False,
                few_shot: bool = False, no_filter: bool = False, unrestricted: bool = False):
    for c in components:
        if c not in _agents:
            raise ValueError(f"Unknown component '{c}'. Options: {list(_agents)}")

    use_context = use_rag or few_shot
    if few_shot:
        rag_fn = "rag_retrieve_fixed"
    elif unrestricted:
        rag_fn = "rag_retrieve_unrestricted"
    elif no_filter:
        rag_fn = "rag_retrieve_nofilter"
    else:
        rag_fn = "rag_retrieve"

    workflow = StateGraph(EmailState)

    if len(components) == 1:
        component = components[0]
        a_name, agent_mod, agent_fn = _agents[component]
        classifier_name, classifier_mod, classifier_fn = _classifiers[component]

        workflow.add_node(a_name, _load(agent_mod, agent_fn))
        workflow.add_node(classifier_name, _load(classifier_mod, classifier_fn))

        if use_context:
            workflow.add_node("rag_retrieve", _load("src.agents.rag.retrieve", rag_fn))
            workflow.add_edge(START, "rag_retrieve")
            workflow.add_edge("rag_retrieve", a_name)
        else:
            workflow.add_edge(START, a_name)

        workflow.add_edge(a_name, classifier_name)
        workflow.add_edge(classifier_name, END)

    elif parallel:
        for component in components:
            a_name, agent_mod, agent_fn = _agents[component]
            workflow.add_node(a_name, _load(agent_mod, agent_fn))

        workflow.add_node("coordinate", _load("src.agents.coordinator.classify", "coordinate"))

        if use_context:
            workflow.add_node("rag_retrieve", _load("src.agents.rag.retrieve", rag_fn))
            workflow.add_edge(START, "rag_retrieve")
            for component in components:
                workflow.add_edge("rag_retrieve", _agents[component][0])
        else:
            for component in components:
                workflow.add_edge(START, _agents[component][0])

        for component in components:
            workflow.add_edge(_agents[component][0], "coordinate")

        workflow.add_edge("coordinate", END)

    else:
        chain = []

        if use_context:
            workflow.add_node("rag_retrieve", _load("src.agents.rag.retrieve", rag_fn))
            chain.append("rag_retrieve")

        for component in components:
            a_name, a_mod, a_fn = _agents[component]
            workflow.add_node(a_name, _load(a_mod, a_fn))
            chain.append(a_name)

        workflow.add_node("coordinate", _load("src.agents.coordinator.classify", "coordinate"))
        chain.append("coordinate")

        workflow.add_edge(START, chain[0])
        for i in range(len(chain) - 1):
            workflow.add_edge(chain[i], chain[i + 1])
        workflow.add_edge(chain[-1], END)

    return workflow.compile()


def agent_name(components: list[str], use_rag: bool = False, few_shot: bool = False,
               no_filter: bool = False, unrestricted: bool = False) -> str:
    name = "_".join(components)
    if few_shot:
        name += "_fewshot"
    elif use_rag:
        name += "_rag"
        if unrestricted:
            name += "_unrestricted"
        elif no_filter:
            name += "_nofilter"
    return name
