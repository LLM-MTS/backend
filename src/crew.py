from crewai import Crew
from agents import (
    emotion_task,
    intent_task,
    knowledge_task,
    quality_task,
    suggest_task,
    summary_task,
)
from agents import (
    emotion_agent,
    intent_agent,
    knowledge_agent,
    quality_agent,
    suggest_agent,
    summary_agent,
)

crew = Crew(
    agents=[
        intent_agent,
        emotion_agent,
        knowledge_agent,
        suggest_agent,
        quality_agent,
        summary_agent,
    ],
    tasks=[
        intent_task,
        emotion_task,
        knowledge_task,
        suggest_task,
        quality_task,
        summary_task,
    ],
    verbose=True,
)


def get_response(user_msg: str) -> dict:
    crew_output = crew.kickoff({"input": user_msg})
    tasks = crew_output.tasks_output
    d = {
        "intent": tasks[0].json_dict.get("intent", "unknown"),
        "emotion": tasks[1].json_dict.get("emotion", "neutral"),
        "knowledge": tasks[2].json_dict.get("answer", ""),
        "suggestion": tasks[3].json_dict.get(
            "answer", ""
        ),  # Обратите внимание на "answer" вместо "suggestion"
        "quality": {
            "politeness": tasks[4].json_dict.get("politeness"),
            "script_match": tasks[4].json_dict.get("script_match"),
            "correctness": tasks[4].json_dict.get("correctness"),
            "comment": tasks[4].json_dict.get("comment", ""),
        },
        "summary": tasks[5].json_dict.get("summary", ""),
        "crm_template": {
            "issue_type": tasks[0].json_dict.get("intent", "other"),
            "client_sentiment": tasks[1].json_dict.get("emotion", "neutral"),
            "resolution": (
                "compensation"
                if "компенсац" in str(tasks[2].json_dict.get("answer", "")).lower()
                else (
                    "escalation"
                    if "передан" in str(tasks[2].json_dict.get("answer", "")).lower()
                    else "info_provided"
                )
            ),
        },
    }
    print(d)
    return d
