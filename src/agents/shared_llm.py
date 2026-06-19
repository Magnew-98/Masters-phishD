import uuid
from langchain_ollama import ChatOllama


def get_llm():
    return ChatOllama(
        model="llama3.1",
        temperature=0.2,
        seed=98,
        num_ctx=4096,
        client_kwargs={"timeout": 120},
    )


def fresh_prompt(prompt: str) -> str:
    """Prepend a unique call ID to prevent Ollama reusing KV cache across sequential agent calls.
    Ollama caches token computations by prompt prefix similarity; emails containing repeated
    patterns (e.g. forwarded-header dashes) can cause the second specialist's call to inherit
    cached context from the first, corrupting its output. A unique prefix defeats the cache hit."""
    return f"[{uuid.uuid4().hex[:8]}]\n{prompt}"
