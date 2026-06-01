from langgraph.graph import StateGraph, END, START
from src.schemas.state import EmailState

_ANALYSIS_NODES = {
    "binary":    ("analyse",            "src.agents.binary.analyse",     "analyse_email"),
    "technical": ("analyse_technical",  "src.agents.technical.analyse",  "analyse_technical"),
    "sentiment": ("analyse_sentiment",  "src.agents.sentiment.analyse",  "analyse_sentiment"),
    "linguistic": ("analyse_linguistic", "src.agents.linguistic.analyse", "analyse_linguistic"),
}

_SINGLE_CLASSIFY_NODES = {
    "binary":    ("classify",           "src.agents.binary.classify",    "classify_email"),
    "technical": ("classify_technical", "src.agents.technical.classify", "classify_technical"),
    "sentiment": ("classify_sentiment", "src.agents.sentiment.classify", "classify_sentiment"),
    "linguistic": ("classify_linguistic", "src.agents.linguistic.classify", "classify_linguistic"),
}


def _import(module_path: str, fn_name: str):
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, fn_name)


def build_graph(components: list[str], use_rag: bool = False, parallel: bool = False):
    """
    Build a LangGraph app from a list of component names and optional flags.

    Single component:
        [rag →] analyse → classify → END

    Multiple components, sequential (default):
        [rag →] analyse1 → analyse2 → ... → coordinate → END

    Multiple components, parallel:
        [rag →] analyse1 ↘
                analyse2 → coordinate → END
                analyse3 ↗
        All analysis nodes run concurrently; coordinator waits for all.
        Requires Ollama to support concurrent requests (OLLAMA_NUM_PARALLEL env var).

    Supported components: binary, technical, sentiment, linguistic
    """
    for c in components:
        if c not in _ANALYSIS_NODES:
            raise ValueError(f"Unknown component '{c}'. Choose from: {list(_ANALYSIS_NODES)}")

    workflow = StateGraph(EmailState)

    if len(components) == 1:
        component = components[0]
        analyse_name, analyse_mod, analyse_fn = _ANALYSIS_NODES[component]
        classify_name, classify_mod, classify_fn = _SINGLE_CLASSIFY_NODES[component]

        workflow.add_node(analyse_name, _import(analyse_mod, analyse_fn))
        workflow.add_node(classify_name, _import(classify_mod, classify_fn))

        if use_rag:
            workflow.add_node("rag_retrieve", _import("src.agents.rag.retrieve", "rag_retrieve"))
            workflow.add_edge(START, "rag_retrieve")
            workflow.add_edge("rag_retrieve", analyse_name)
        else:
            workflow.add_edge(START, analyse_name)

        workflow.add_edge(analyse_name, classify_name)
        workflow.add_edge(classify_name, END)

    elif parallel:
        # Fan-out: all specialists run concurrently, coordinator fans in
        for component in components:
            analyse_name, analyse_mod, analyse_fn = _ANALYSIS_NODES[component]
            workflow.add_node(analyse_name, _import(analyse_mod, analyse_fn))

        coordinate_fn = _import("src.agents.coordinator.classify", "coordinate")
        workflow.add_node("coordinate", coordinate_fn)

        if use_rag:
            workflow.add_node("rag_retrieve", _import("src.agents.rag.retrieve", "rag_retrieve"))
            workflow.add_edge(START, "rag_retrieve")
            for component in components:
                workflow.add_edge("rag_retrieve", _ANALYSIS_NODES[component][0])
        else:
            for component in components:
                workflow.add_edge(START, _ANALYSIS_NODES[component][0])

        for component in components:
            workflow.add_edge(_ANALYSIS_NODES[component][0], "coordinate")

        workflow.add_edge("coordinate", END)

    else:
        # Sequential (default): analysts run one after another
        chain = []

        if use_rag:
            workflow.add_node("rag_retrieve", _import("src.agents.rag.retrieve", "rag_retrieve"))
            chain.append("rag_retrieve")

        for component in components:
            analyse_name, analyse_mod, analyse_fn = _ANALYSIS_NODES[component]
            workflow.add_node(analyse_name, _import(analyse_mod, analyse_fn))
            chain.append(analyse_name)

        coordinate_fn = _import("src.agents.coordinator.classify", "coordinate")
        workflow.add_node("coordinate", coordinate_fn)
        chain.append("coordinate")

        workflow.add_edge(START, chain[0])
        for i in range(len(chain) - 1):
            workflow.add_edge(chain[i], chain[i + 1])
        workflow.add_edge(chain[-1], END)

    return workflow.compile()


def agent_name(components: list[str], use_rag: bool = False) -> str:
    name = "_".join(components)
    if use_rag:
        name += "_rag"
    return name
