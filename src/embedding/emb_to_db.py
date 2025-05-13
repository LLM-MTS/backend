import os
import json
import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
import re


def clean_text(text):
    text = re.sub(r"<[^>]+>", "", text)  # удаляем HTML
    text = text.replace("\xa0", " ")
    text = text.strip()
    return text


load_dotenv()

API_KEY = os.getenv("MWS_API_KEY")
BASE_URL = os.getenv("MWS_BASE_URL")
INPUT_FILE = "back/src/embedding/data.json"
COLLECTION = "knowledge_collection"
VECTOR_NAME = "embedding"  # имя вектора в схеме

# Инициализация клиента
qdrant_client = QdrantClient(
    url="http://localhost:6333",
    prefer_grpc=False,
)


def get_embedding(text: str) -> list[float]:
    resp = requests.post(
        f"{BASE_URL}/embeddings",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"model": "bge-m3", "input": text},
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def create_collection():
    # Получаем примерный размер вектора
    sample = get_embedding("sample")
    vec_size = len(sample)

    # Рекреируем коллекцию, указав имя вектора "embedding" через простой dict
    qdrant_client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config={
            VECTOR_NAME: models.VectorParams(
                size=vec_size, distance=models.Distance.COSINE
            )
        },
    )


def create_collection_if_not_exists():
    # Попробуем получить метаданные; если коллекции нет — создадим новую
    try:
        qdrant_client.get_collection(COLLECTION)
        print(f"Коллекция «{COLLECTION}» уже существует — пропускаем создание")
    except Exception:
        # Ниже тот же код: определяем размер embedding и создаём коллекцию
        sample = get_embedding("sample")
        vec_size = len(sample)

        qdrant_client.create_collection(
            collection_name=COLLECTION,
            vectors_config={
                VECTOR_NAME: models.VectorParams(
                    size=vec_size, distance=models.Distance.COSINE
                )
            },
        )
        print(f"Коллекция «{COLLECTION}» создана")


def upload_to_qdrant(records: list[dict]):
    points = []
    n = len(records)
    for idx, rec in enumerate(records):
        # print(idx, "\\", n)
        points.append(
            models.PointStruct(
                id=idx,
                # Передаём словарь с тем же именем, что и в vectors_config
                vector={VECTOR_NAME: rec["embedding"]},
                payload={
                    "url": rec.get("url"),
                    "service_name": rec.get("service_name"),
                    "question": rec.get("question"),
                    "answer": rec.get("answer"),
                },
            )
        )
    qdrant_client.upload_points(
        collection_name=COLLECTION,
        points=points,
        wait=True,  # дождаться индексации
    )


def add_to_db():
    print("start add_to_db")
    create_collection()
    print("created _collection")
    with open(INPUT_FILE, encoding="utf-8") as f:
        records = json.load(f)
    print("records get")
    # Генерация embedding’ов
    # processed = []
    # i = 0
    # n = len(records)
    # for rec in records:
    #     i +=1
    #     print(i, f"/{n}")
    #     text = f"{rec['question']}\n\n{rec['answer']}"
    #     text = clean_text(text)

    #     rec["embedding"] = get_embedding(text)
    #     processed.append(rec)
    #     if i% 1000 == 0:
    #         with open("processed.json", "w", encoding="utf-8") as f:
    #             json.dump(processed, f, ensure_ascii=False, indent=2)
    print("created_embedding")
    with open("back/src/embedding/processed.json", encoding="utf-8") as f:
        processed = json.load(f)

    upload_to_qdrant(processed)

    cnt = qdrant_client.count(collection_name=COLLECTION, exact=True).count
    print(f"Успешно загружено {len(processed)} точек, всего в коллекции: {cnt}")


if __name__ == "__main__":
    add_to_db()
