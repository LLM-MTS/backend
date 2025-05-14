from crewai import Crew
from src.agents import (
    emotion_task,
    intent_task,
    knowledge_task,
    quality_task,
    suggest_task,
    summary_task,
)
from src.agents import (
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


def safe_get(task_output, key, default=None):
    return (
        task_output.json_dict.get(key, default)
        if task_output and task_output.json_dict
        else default
    )


def get_response(user_msg: str) -> dict:
    crew_output = crew.kickoff({"input": user_msg})
    tasks = crew_output.tasks_output
    d = {
        "intent": safe_get(tasks[0], "intent", "unknown"),
        "emotion": safe_get(tasks[1], "emotion", "neutral"),
        "knowledge": safe_get(tasks[2], "answer", ""),
        "suggestion": safe_get(tasks[3], "answer", ""),
        "quality": {
            "politeness": safe_get(tasks[4], "politeness"),
            "script_match": safe_get(tasks[4], "script_match"),
            "correctness": safe_get(tasks[4], "correctness"),
            "comment": safe_get(tasks[4], "comment", ""),
        },
        "summary": safe_get(tasks[5], "summary", ""),
        "crm_template": {
            "issue_type": safe_get(tasks[0], "intent", "other"),
            "client_sentiment": safe_get(tasks[1], "emotion", "neutral"),
            "resolution": (
                "compensation"
                if "компенсац" in str(safe_get(tasks[2], "answer", "")).lower()
                else (
                    "escalation"
                    if "передан" in str(safe_get(tasks[2], "answer", "")).lower()
                    else "info_provided"
                )
            ),
        },
    }

    print(d)
    return d
