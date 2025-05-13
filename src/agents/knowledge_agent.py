import os
import json
from dotenv import load_dotenv
from crewai import LLM, Agent, Task, Crew
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# load_dotenv()

# embedding_model = SentenceTransformer("BAAI/bge-m3")


# qdrant_client = QdrantClient(
#     url="http://localhost:6333",  # Явно указываем HTTP
#     # api_key="QDRANT_API_KEY",
#     prefer_grpc=False,
# )

# COLLECTION_NAME = "knowledge_collection"
# EMBEDDING_SIZE = 1024

# try:
#     collection_info = qdrant_client.get_collection(COLLECTION_NAME)
#     if collection_info.vectors.size != EMBEDDING_SIZE:
#         raise ValueError("Несовпадение размерности векторов")
# except Exception:
#     qdrant_client.recreate_collection(
#         collection_name=COLLECTION_NAME,
#         vectors_config={"text": {"size": EMBEDDING_SIZE, "distance": "Cosine"}},
#     )

# llm = LLM(
#     model="openai/mws-gpt-alpha",
#     api_key=os.getenv("MWS_API_KEY"),
#     base_url=os.getenv("MWS_BASE_URL"),
#     temperature=0.2,
# )


# class KnowledgeResponse(BaseModel):
#     answer: str


# def search_qdrant(query: str, top_k: int = 3) -> list:
#     query_embedding = embedding_model.encode(query).tolist()

#     hits = qdrant_client.search(
#         collection_name=COLLECTION_NAME,
#         query_vector=("text", query_embedding),
#         limit=top_k,
#         with_payload=True,
#     )

#     return [hit.payload["answer"] for hit in hits if hit.payload]


# knowledge_agent = Agent(
#     role="Knowledge Agent",
#     goal="Находить ответ на вопрос клиента в векторной базе знаний Qdrant",
#     backstory="Эксперт по поиску в технической документации с использованием RAG",
#     llm=llm,
# )

# knowledge_task = Task(
#     description=(
#         "Ты — интеллектуальный ассистент поиска. Анализируй найденные данные и формируй структурированный ответ.\n"
#         "Алгоритм работы:\n"
#         "1. Найди до 3 релевантных ответов в Qdrant\n"
#         "2. Объедини информацию в четкий пошаговый ответ\n"
#         "3. Сохрани оригинальную структуру из базы данных\n"
#         "4. Если информации нет - честно признайся\n"
#         'Формат вывода: {"answer": "структурированный_ответ"}'
#     ),
#     expected_output='JSON с ключом "answer" в формате Markdown',
#     agent=knowledge_agent,
#     output_json=KnowledgeResponse,
#     async_execution=False,
#     execution_handler=lambda task_input: (
#         {"answer": format_answers(search_qdrant(task_input["query"]))}
#         if task_input
#         else {"answer": "Информация не найдена"}
#     ),
# )


# def format_answers(answers: list) -> str:
#     if not answers:
#         return "Информация не найдена"

#     unique_answers = []
#     seen = set()
#     for answer in answers:
#         if answer not in seen:
#             seen.add(answer)
#             unique_answers.append(answer)

#     return "\n\n".join(unique_answers)


# if __name__ == "__main__":
#     test_query = {"query": "Как изменить код входа в приложение МТС Банка?"}
#     result = knowledge_task.execute(test_query)
#     print(result.answer)
import os
import json
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

load_dotenv()

# Инициализация модели для эмбеддингов
embedding_model = SentenceTransformer("BAAI/bge-m3")

# Конфигурация Qdrant
qdrant_client = QdrantClient(
    url="http://localhost:6333",
    prefer_grpc=False,
)

COLLECTION_NAME = "knowledge_collection"
VECTOR_NAME = "embedding"  # Должно совпадать с именем в вашей коллекции


class KnowledgeResponse(BaseModel):
    answer: str


def search_qdrant(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Поиск в Qdrant с правильным именем вектора"""
    query_embedding = embedding_model.encode(query).tolist()

    hits = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=(
            VECTOR_NAME,
            query_embedding,
        ),  # Используем правильное имя вектора
        limit=top_k,
        with_payload=True,
    )

    return [
        {
            "answer": hit.payload.get("answer", ""),
            "question": hit.payload.get("question", ""),
            "source": hit.payload.get("url", ""),
        }
        for hit in hits
    ]


def format_answers(results: List[Dict[str, Any]]) -> str:
    """Форматирование ответов в Markdown"""
    if not results:
        return "Информация не найдена"

    formatted = []
    for i, result in enumerate(results, 1):
        formatted.append(
            f"### Результат {i}\n"
            f"**Вопрос:** {result['question']}\n\n"
            f"**Ответ:** {result['answer']}\n\n"
            f"*Источник:* {result['source']}"
        )

    return "\n\n".join(formatted)


# Инициализация LLM
llm = LLM(
    model="openai/mws-gpt-alpha",
    api_key=os.getenv("MWS_API_KEY"),
    base_url=os.getenv("MWS_BASE_URL"),
    temperature=0.2,
)

knowledge_agent = Agent(
    role="Knowledge Agent",
    goal="Находить ответ на вопрос клиента в векторной базе знаний Qdrant",
    backstory="Эксперт по поиску в технической документации с использованием RAG",
    llm=llm,
    allow_delegation=False,
)


def knowledge_task_executor(task_input: Dict) -> Dict:
    """Обработчик задачи, возвращающий словарь для Pydantic"""
    query = task_input.get("query", "")
    search_results = search_qdrant(query)
    formatted_answer = format_answers(search_results)
    return {"answer": formatted_answer}


knowledge_task = Task(
    description=(
        "Ты — интеллектуальный ассистент поиска. Анализируй найденные данные и формируй структурированный ответ.\n"
        "Алгоритм работы:\n"
        "1. Найди до 3 релевантных ответов в Qdrant\n"
        "2. Объедини информацию в четкий пошаговый ответ\n"
        "3. Сохрани оригинальную структуру из базы данных\n"
        "4. Если информации нет - честно признайся\n"
    ),
    expected_output="Ответ в формате Markdown с источниками информации",
    agent=knowledge_agent,
    output_json=KnowledgeResponse,
    async_execution=False,
    execution_handler=knowledge_task_executor,
)

if __name__ == "__main__":
    # Тестирование
    test_query = {"query": "Как изменить код входа в приложение МТС Банка?"}
    result = knowledge_task.execute(test_query)

    if isinstance(result, KnowledgeResponse):
        print(result.answer)
    else:
        print("Ошибка: результат не соответствует ожидаемому формату")
