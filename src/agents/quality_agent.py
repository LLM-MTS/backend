import os
from dotenv import load_dotenv
from crewai import LLM, Agent, Task, Crew
from pydantic import BaseModel, Field

load_dotenv()

llm = LLM(
    model="openai/mws-gpt-alpha",
    api_key=os.getenv("MWS_API_KEY"),
    base_url=os.getenv("MWS_BASE_URL"),
    temperature=0.2,
)
qa_feedback = {
    "good": [
        "Оператор был вежлив и доброжелателен.",
        "Ответил корректно и без резких выражений.",
    ],
    "neutral": [
        "Оператор был формален, но не груб.",
        "Ответ был краткий и без эмоциональной окраски.",
    ],
    "bad": [
        "Оператор перебил клиента и не извинился.",
        "Было использовано неуместное выражение.",
    ],
    "script_match_true": [
        "Следовал стандартному скрипту.",
        "Инструкция соответствовала внутренним правилам.",
    ],
    "script_match_false": [
        "Не представился в начале общения.",
        "Пропустил этап подтверждения данных.",
    ],
    "correctness_correct": [
        "Ответ точный, информация верная.",
        "Оператор справился без ошибок.",
    ],
    "correctness_mistake": [
        "Была допущена неточность в сроках.",
        "Оператор случайно ввёл клиента в заблуждение по поводу адреса.",
    ],
    "correctness_critical": [
        "Предоставлена ложная информация о возврате.",
        "Оператор отказал без права на это.",
    ],
}


quality_agent = Agent(
    role="Quality Assurance Agent",
    goal="Оценить общение по параметрам качества",
    backstory="Контролёр качества, следит за стандартами общения",
    llm=llm,
)


class QAResponse(BaseModel):
    politeness: str = Field(..., pattern="^(good|neutral|bad)$")
    script_match: bool
    correctness: str = Field(..., pattern="^(correct|mistake|critical_mistake)$")
    comment: str | None = None


quality_task = Task(
    description=(
        "Ты — агент контроля качества. Получи диалог и оцени его. Верни JSON:\n"
        '{\n  "politeness": "good|neutral|bad",\n  "script_match": true|false,\n  "correctness": "correct|mistake|critical_mistake",\n  "comment": "(опционально)"\n}\n'
        "Примеры:\n"
        "- politeness: good → 'Оператор был вежлив и доброжелателен.'\n"
        "- politeness: neutral → 'Оператор был формален, но не груб.'\n"
        "- politeness: bad → 'Оператор перебил клиента и не извинился.'\n"
        "- script_match: true → 'Следовал стандартному скрипту.'\n"
        "- script_match: false → 'Пропустил этап подтверждения данных.'\n"
        "- correctness: correct → 'Оператор справился без ошибок.'\n"
        "- correctness: mistake → 'Была допущена неточность в сроках.'\n"
        "- correctness: critical_mistake → 'Предоставлена ложная информация о возврате.'"
        + "\n".join(
            [
                f"- {emotion} → " + " / ".join(examples)
                for emotion, examples in qa_feedback.items()
            ]
        )
    ),
    expected_output="JSON с параметрами качества. Строго в формате JSON без пояснений!",
    agent=quality_agent,
    output_json=QAResponse,
    parse_output=True,
)
