from langchain_ollama import ChatOllama

def get_llm():
    return ChatOllama(
        model="llama3.1",
        temperature=0.2,
        seed=98,
        num_ctx=4096,
        num_keep=0,
        client_kwargs={"timeout": 120},
    )
