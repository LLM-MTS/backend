import os
from dotenv import load_dotenv
from crewai import LLM, Agent, Task, Crew
from pydantic import BaseModel, Field

load_dotenv()
llm = LLM(
    model="groq/llama3-70b-8192",
    api_key=os.getenv("GROQ_TOKEN"),
    api_base="https://api.groq.com/openai/v1",
    temperature=0.7,
)
suggestions = {
    "anger": [
        "Проявите сочувствие и предложите решение проблемы.",
        "Извинитесь и уточните детали для помощи.",
    ],
    "happy": [
        "Поблагодарите клиента и пожелайте хорошего дня.",
        "Ответьте в позитивном ключе и уточните, нужна ли ещё помощь.",
    ],
    "confusion": [
        "Повторите информацию простыми словами.",
        "Дайте пошаговую инструкцию или ссылку на помощь.",
    ],
}


suggest_agent = Agent(
    role="Action Suggestion Agent",
    goal="Предложить оператору реакцию на основе intent и emotion",
    backstory="Ассистент по стратегии общения",
    llm=llm,
)


class SuggestionResponse(BaseModel):
    answer: str  # Можно добавить валидацию если нужно

    # Опционально: автоматическое преобразование строки в словарь
    @classmethod
    def validate_output(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                pass
        return v


suggest_task = Task(
    description=(
        "Ты — агент рекомендаций. Отвечай от первого лица. Получив JSON с intent и emotion, предложи оператору подходящее действие.\n"
        "Примеры:\n"
        "- intent: complaint, emotion: anger → 'Прояви сочувствие и предложи решение проблемы.'\n"
        "- intent: thank_you, emotion: happy → 'Поблагодарите клиента и пожелайте хорошего дня.'\n"
        "- intent: request_info, emotion: confusion → 'Повторите информацию простыми словами и дайте ссылку на инструкцию.'"
        + "\n".join(
            [
                f"- {emotion} → " + " / ".join(examples)
                for emotion, examples in suggestions.items()
            ]
        )
    ),
    expected_output='JSON с ключом "suggestion". Строго в формате JSON без пояснений!',
    agent=suggest_agent,
    output_json=SuggestionResponse,
    parse_output=True,
)
