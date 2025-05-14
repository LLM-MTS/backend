import os
import json
import re
from typing import List
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from qdrant_client import QdrantClient
from qdrant_client.http import models

# === Загрузка переменных окружения ===
load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

INPUT_FILE = "src/embedding/data.json"
COLLECTION = "knowledge_collection"
VECTOR_NAME = "embedding"

# === Qdrant клиент ===
qdrant_client = QdrantClient(
    url="http://qdrant:6333",
    prefer_grpc=False,
)

# === HuggingFace Inference клиент ===
client = InferenceClient(
    model=HF_MODEL,
    token=HF_TOKEN
)

# === Очистка текста ===
def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\xa0", " ")
    return text.strip()

# === Эмбеддинг через HuggingFace Inference API ===
def get_remote_embedding(text: str) -> List[float]:
    return client.feature_extraction(text)

# === Создание коллекции в Qdrant ===
def create_collection():
    sample = get_remote_embedding("Пример")
    vec_size = len(sample)
    qdrant_client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config={
            VECTOR_NAME: models.VectorParams(
                size=vec_size,
                distance=models.Distance.COSINE
            )
        },
    )

# === Загрузка точек в Qdrant ===
def upload_to_qdrant(records: List[dict]):
    points = []
    for idx, rec in enumerate(records):
        points.append(
            models.PointStruct(
                id=idx,
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
        wait=True,
    )

# === Основная функция ===
def add_to_db():
    print("🚀 Старт загрузки в Qdrant...")
    
    try:
        create_collection()
        print("✅ Коллекция создана")
    except Exception as e:
        print(f"⚠️ Ошибка при создании коллекции: {e}")
        return

    with open(INPUT_FILE, encoding="utf-8") as f:
        records = json.load(f)

    processed = []
    for i, rec in enumerate(records, start=1):
        print(f"{i}/{len(records)}")
        text = f"{rec['question']}\n\n{rec['answer']}"
        text = clean_text(text)
        try:
            rec["embedding"] = get_remote_embedding(text).tolist()
        except Exception as e:
            print(f"⚠️ Ошибка при эмбеддинге: {e}")
            continue
        processed.append(rec)

        if i % 1000 == 0:
            with open("processed.json", "w", encoding="utf-8") as f:
                json.dump(processed, f, ensure_ascii=False, indent=2)

    print("✅ Эмбеддинги готовы, загружаем в Qdrant...")
    upload_to_qdrant(processed)

    cnt = qdrant_client.count(collection_name=COLLECTION, exact=True).count
    print(f"🎉 Загружено {len(processed)} точек, всего в коллекции: {cnt}")

# === Точка входа ===
if __name__ == "__main__":
    add_to_db()
