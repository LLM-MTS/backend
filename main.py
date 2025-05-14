from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from src.crew import get_response
from src.embedding.emb_to_db import add_to_db
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

@app.on_event("startup")
async def startup_event():
    try:
        # add_to_db()
        pass
    except Exception as e:
        # Логируем ошибку, но не даём серверу упасть
        print("⚠️  Ошибка в add_to_db, пропускаем стартовую инициализацию:", e)


class MessagePayload(BaseModel):
    message: str


@app.post("/message")
async def send_message(payload: MessagePayload):
    result = get_response(payload.message)
    return JSONResponse(content=result)



if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
