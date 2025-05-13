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
intent_examples = {
    "complaint": [
        "Мне не доставили товар вовремя.",
        "Хочу пожаловаться на работу курьера.",
    ],
    "request_info": [
        "Расскажите подробнее про тарифы.",
        "Какие у вас условия доставки?",
    ],
    "cancel_service": ["Отмените мой заказ.", "Больше не хочу пользоваться услугой."],
    "update_data": ["Измените мой адрес.", "Обновите, пожалуйста, номер телефона."],
    "thank_you": ["Спасибо большое за помощь!", "Благодарю, всё отлично!"],
    "greeting": ["Здравствуйте!", "Добрый день!"],
    "schedule_change": [
        "Можно перенести доставку на завтра?",
        "Мне нужно изменить дату получения.",
    ],
    "make_purchase": ["Хочу купить ваш продукт.", "Где можно оформить заказ?"],
    "check_status": ["Где мой заказ?", "Какой статус у моей заявки?"],
    "other": ["А вы видели прогноз погоды?", "Просто проверяю, работает ли чат."],
}


class IntentResponse(BaseModel):
    intent: str = Field(..., pattern=f"^({'|'.join(intent_examples.keys())})$")


intent_agent = Agent(
    role="Intent Agent",
    goal="Определять намерение клиента по сообщению",
    backstory="Специалист по выявлению целей взаимодействия клиента",
    llm=llm,
)

intent_task = Task(
    description=(
        'Ты — агент по анализу намерений. Проанализируй сообщение клиента и верни JSON: {"intent": "<твоя_метка>"}.\n'
        "Возможные метки:\n"
        "- complaint — жалоба\n"
        "- request_info — запрос информации\n"
        "- cancel_service — отказ от услуги\n"
        "- update_data — изменение персональных данных\n"
        "- thank_you — благодарность\n"
        "- greeting — приветствие\n"
        "- schedule_change — изменение графика/доставки\n"
        "- make_purchase — желание купить\n"
        "- check_status — проверка статуса заказа/услуги\n"
        "- other — другое или нераспознанное намерение\n\n"
        "Примеры:\n"
        "- 'Мне не доставили товар вовремя.' → complaint\n"
        "- 'Расскажите подробнее про тарифы.' → request_info\n"
        "- 'Отмените мой заказ.' → cancel_service"
        + "\n".join(
            [
                f"- {emotion} → " + " / ".join(examples)
                for emotion, examples in intent_examples.items()
            ]
        )
    ),
    expected_output='JSON с ключом "intent". ',
    agent=intent_agent,
    output_json=IntentResponse,
    parse_output=True,
)
