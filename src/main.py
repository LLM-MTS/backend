from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from crew import get_response
from embedding.emb_to_db import add_to_db

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    add_to_db()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Или указать конкретные URL для безопасности
    allow_credentials=True,
    allow_methods=["*"],  # Разрешить все методы (или указать только POST)
    allow_headers=["*"],  # Разрешить все заголовки
)


class MessagePayload(BaseModel):
    message: str


@app.post("/message")
async def send_message(payload: MessagePayload):

    result = get_response(payload.message)
    return JSONResponse(content=result)


@app.get("/")
async def root():
    return JSONResponse(content={"message": "Добро пожаловать!"})


@app.middleware("http")
async def redirect_to_root(request: Request, call_next):
    allowed_paths = {"/", "/message"}
    if request.url.path not in allowed_paths:
        return RedirectResponse(url="/")
    return await call_next(request)


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
