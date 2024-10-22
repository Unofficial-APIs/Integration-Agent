from langchain_openai import ChatOpenAI

class LLMSingleton:
    _instance = None

    @classmethod
    def get_instance(cls, model: str = None):
        if cls._instance is None and model is not None:
            cls._instance = ChatOpenAI(model=model, temperature=1)
        return cls._instance

    @classmethod
    def set_model(cls, model: str):
        cls._instance = ChatOpenAI(model=model, temperature=1)

llm = LLMSingleton()