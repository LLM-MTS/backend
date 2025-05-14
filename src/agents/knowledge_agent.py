import os
from typing import List
from crewai import Agent, Task, LLM
from qdrant_client import QdrantClient
from src.embedding.emb_to_db import get_remote_embedding

# Настройки Qdrant
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "knowledge_collection")
QDRANT_VECTOR_NAME = os.getenv("QDRANT_VECTOR_NAME", "embedding")
QDRANT_TOP_K = int(os.getenv("QDRANT_TOP_K", "3"))

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    prefer_grpc=os.getenv("QDRANT_USE_GRPC", "False").lower() in ("true", "1"),
)

# Прямой поиск в Qdrant — можно вызывать вручную
def search_qdrant(query: str) -> str:
    embedding: List[float] = get_remote_embedding(query)
    if not embedding:
        return "Нет embedding для запроса"
    try:
        hits = qdrant_client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=embedding,
            limit=QDRANT_TOP_K,
            with_payload=True,
            with_vectors=False,
            vector_name=QDRANT_VECTOR_NAME,
        )
        results = [hit.payload.get("answer", "") for hit in hits if hit.payload]
        return "\n".join(results) if results else "Ничего не найдено"
    except Exception as e:
        return f"Ошибка Qdrant: {e}"

# Простая LLM модель через Groq (или OpenAI)
llm = LLM(
    model="llama3-70b-8192",
    api_key=os.getenv("GROQ_TOKEN"),
    api_base="https://api.groq.com/openai/v1",
)

# Агент CrewAI
knowledge_agent = Agent(
    role="Агент знаний",
    goal="Находить точные ответы на вопросы клиента из базы знаний",
    backstory="У тебя есть доступ к базе знаний и способность использовать её для помощи клиентам.",
    llm=llm,
    verbose=True,
)
from pydantic import BaseModel

class AnswerModel(BaseModel):
    answer: str
    
# Задача для агента
knowledge_task = Task(
    description="Ответь на вопрос клиента, используя свою базу знаний.",
    expected_output="JSON с ключом 'answer': {\"answer\": \"...\"}",
    agent=knowledge_agent,
    output_json=AnswerModel,
    async_execution=False,
)
