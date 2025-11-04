"""
Simplified CAMEL experiment with:
- Two-agent role-playing dialogue
- Optional judge
- JSONL logging and basic metrics collection
"""

import os
import sys
import json
import uuid
import re
from datetime import datetime
from typing import Optional, Any, Dict

try:
    from camel.societies import RolePlaying
    from camel.messages import BaseMessage
    from camel.agents import ChatAgent
except ImportError:
    print("CAMEL is not installed. Run: pip install camel-ai", file=sys.stderr)
    sys.exit(1)



# Utility helpers

def getenv(name: str, default: Any, cast=str) -> Any:
    val = os.getenv(name, str(default)).strip()
    try:
        return cast(val)
    except Exception:
        return default


def to_base_message(obj: Any) -> Optional[BaseMessage]:
    """Safely converts various CAMEL response objects to BaseMessage."""
    if obj is None:
        return None
    if isinstance(obj, BaseMessage):
        return obj
    msg = getattr(obj, "msg", None)
    if isinstance(msg, BaseMessage):
        return msg
    msgs = getattr(obj, "msgs", None)
    if isinstance(msgs, (list, tuple)) and msgs:
        last = msgs[-1]
        return getattr(last, "msg", last) if isinstance(last, (BaseMessage, object)) else None
    return None


def print_msg(tag: str, msg: Any) -> None:
    # Печатает сообщение в консоль, стараясь не упасть на неожиданных типах
    try:
        content = getattr(msg, "content", str(msg))
        print(f"[{tag}]\n{content}\n")
    except Exception:
        print(f"[{tag}] <unprintable message>\n")


def should_print_dialog() -> bool:
    # Флаг: выводить ли диалог по ходам (управляется переменной окружения)
    return getenv("CAMEL_PRINT_DIALOG", "1").lower() not in ("0", "false", "no")


# Metrics & Logging

def extract_basic_metrics(dialog: list[BaseMessage]) -> Dict[str, Any]:
    """Считает простые метрики по диалогу CAMEL (кол-во ходов, «водность», среднюю длину)."""
    if not dialog:
        return {}
    text_concat = "\n".join(str(getattr(m, "content", "")) for m in dialog)
    n_turns = len(dialog)
    flake_count = len(re.findall(r"\b(I will|I can|I should|let me|I am going to)\b", text_concat, re.I))
    flake_ratio = flake_count / max(1, n_turns)
    avg_len = sum(len(str(getattr(m, "content", ""))) for m in dialog) / n_turns
    return {
        "turns": n_turns,
        "flake_ratio": round(flake_ratio, 3),
        "avg_msg_len": round(avg_len, 1),
    }


def log_dialog(run_info: dict, dialog: list[BaseMessage], final_msg: BaseMessage, info: dict):
    """Записывает запись диалога в файл ``logs/dialogs.jsonl`` (по строке на запись)."""
    os.makedirs("logs", exist_ok=True)
    data = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "assistant_model": run_info["model_a"],
        "user_model": run_info["model_b"],
        "task": run_info["task"],
        "rounds": run_info["rounds"],
        "stop_reason": info.get("stop_reason") if info else None,
        "usage": info.get("usage") if info else None,
        "metrics": extract_basic_metrics(dialog),
        "final": getattr(final_msg, "content", ""),
        "dialog": [
            {
                "role": getattr(m, "role_name", ""),
                "content": getattr(m, "content", ""),
            }
            for m in dialog
        ],
    }
    with open("logs/dialogs.jsonl", "a", encoding="utf8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


# Core experiment logic

def run_role_playing() -> BaseMessage:
    """Запускает сессию ролевого взаимодействия двух агентов CAMEL.

    - Настройки берутся из переменных окружения с дефолтами
    - Агентам модель пробрасывается через ``assistant_agent_kwargs``/``user_agent_kwargs``
    - Поддерживаются как современный API (``run``), так и легаси (``init_chat``/``step``)
    """
    run_info = {
        "assistant_role": getenv("CAMEL_ASSISTANT_ROLE", "Coder"),
        "user_role": getenv("CAMEL_USER_ROLE", "Reviewer"),
        "task": getenv("CAMEL_TASK", "Спроектировать простой CLI калькулятор и написать тесты."),
        "rounds": getenv("CAMEL_ROUNDS", 10, int),
        "temperature": getenv("CAMEL_TEMPERATURE", 0.3, float),
        "max_tokens": getenv("CAMEL_MAX_TOKENS", 384, int),
        "model_a": getenv("AGENT_MODEL_A", "gpt-4o-mini"),
        "model_b": getenv("AGENT_MODEL_B", "gpt-4o-mini"),
    }

    # Инициализируем сценарий ролевого взаимодействия
    rp = RolePlaying(
        assistant_role_name=run_info["assistant_role"],
        user_role_name=run_info["user_role"],
        task_prompt=run_info["task"],
        with_task_specify=True,
        assistant_agent_kwargs=dict(
            model=run_info["model_a"],
        ),
        user_agent_kwargs=dict(
            model=run_info["model_b"],
        ),
    )

    dialog, info, final_msg = [], {}, None

    # Современный API (предпочтительный путь)
    if hasattr(rp, "run"):
        result = rp.run(n_rounds=run_info["rounds"]) if "n_rounds" in rp.run.__code__.co_varnames else rp.run()
        if isinstance(result, (list, tuple)):
            final_msg = result[0]
            if len(result) > 1 and isinstance(result[1], (list, dict)):
                if isinstance(result[1], list):
                    dialog = result[1]
                elif isinstance(result[1], dict):
                    info = result[1]
        else:
            final_msg = result

        if should_print_dialog() and dialog:
            for i, m in enumerate(dialog, 1):
                role = getattr(m, "role_name", "message")
                print_msg(f"TURN {i} | {role}", m)

        log_dialog(run_info, dialog, final_msg, info)
        return to_base_message(final_msg) or BaseMessage(role_name="assistant", content="(no output)")

    # Легаси-ветка на случай старых версий CAMEL
    if hasattr(rp, "init_chat") and hasattr(rp, "step"):
        init_out = rp.init_chat() or (None, None)
        assistant_msg = to_base_message(init_out[0]) if isinstance(init_out, (list, tuple)) else to_base_message(init_out)
        user_msg = to_base_message(init_out[1]) if isinstance(init_out, (list, tuple)) and len(init_out) > 1 else None
        if assistant_msg:
            dialog.append(assistant_msg)
        if user_msg:
            dialog.append(user_msg)

        for t in range(1, run_info["rounds"] + 1):
            if not isinstance(assistant_msg, BaseMessage):
                break
            step_out = rp.step(assistant_msg)
            if not step_out:
                break
            if isinstance(step_out, (list, tuple)):
                assistant_msg = to_base_message(step_out[0])
                user_msg = to_base_message(step_out[1]) if len(step_out) > 1 else None
            else:
                assistant_msg = to_base_message(step_out)
            if assistant_msg:
                dialog.append(assistant_msg)
            if user_msg:
                dialog.append(user_msg)
            if should_print_dialog():
                if assistant_msg: print_msg(f"TURN {t} | assistant", assistant_msg)
                if user_msg: print_msg(f"TURN {t} | user", user_msg)

        final_msg = assistant_msg or BaseMessage(role_name="assistant", content="(no output)")
        log_dialog(run_info, dialog, final_msg, info)
        return final_msg

    # No valid API path
    return BaseMessage(role_name="assistant", content="(no output)")

# Optional judge 

def maybe_judge(final_msg: BaseMessage) -> Optional[str]:
    """Опционально прогоняет итог через «судью», если задан ``JUDGE_MODEL``.

    Возвращает текст короткой оценки или ``None``.
    """
    model = os.getenv("JUDGE_MODEL")
    if not model:
        return None
    prompt = (
        "Оцени итоговое решение агента. Дай краткий вердикт (OK/ISSUES) и 1–2 рекомендации.\n\n"
        f"Ответ агента:\n{getattr(final_msg, 'content', '')}"
    )
    judge = ChatAgent(model=model)
    resp = judge.step(BaseMessage(role_name="user", role_type="user", content=prompt))
    return getattr(resp.msg, "content", str(resp.msg))



# Entry point

def main():
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_BASE")):
        print("Warning: OPENAI_API_KEY не задан. Установите перед запуском.", file=sys.stderr)

    final_msg = run_role_playing()
    print("\n=== FINAL MESSAGE ===\n")
    print(getattr(final_msg, "content", str(final_msg)))

    if judge_text := maybe_judge(final_msg):
        print("\n=== JUDGE ===\n")
        print(judge_text)


if __name__ == "__main__":
    main()
