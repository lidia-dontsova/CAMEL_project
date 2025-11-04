"""
CAMEL Experiment with:
- Two-agent role-playing dialogue
- Always-on judge (LLM-based evaluation)
- JSONL logging and full metric tracking
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
    try:
        content = getattr(msg, "content", str(msg))
        print(f"[{tag}]\n{content}\n")
    except Exception:
        print(f"[{tag}] <unprintable message>\n")


def should_print_dialog() -> bool:
    return getenv("CAMEL_PRINT_DIALOG", "1").lower() not in ("0", "false", "no")


# Metrics & Logging 

def extract_basic_metrics(dialog: list[BaseMessage]) -> Dict[str, Any]:
    """Считает простые метрики по диалогу."""
    if not dialog:
        return {}
    texts = [str(getattr(m, "content", "")) for m in dialog]
    text_concat = "\n".join(texts)
    n_turns = len(dialog)
    flake_count = len(re.findall(r"\b(I will|I can|I should|let me|I am going to)\b", text_concat, re.I))
    flake_ratio = flake_count / max(1, n_turns)
    avg_len = sum(len(t) for t in texts) / n_turns
    return {
        "turns": n_turns,
        "flake_ratio": round(flake_ratio, 3),
        "avg_msg_len": round(avg_len, 1),
    }


def log_dialog(run_info: dict, dialog: list[BaseMessage], final_msg: BaseMessage, info: dict, judge_verdict: str):
    """Сохраняет весь эксперимент (включая вердикт судьи) в logs/dialogs.jsonl"""
    os.makedirs("logs", exist_ok=True)
    metrics = extract_basic_metrics(dialog)
    metrics["judge_verdict"] = judge_verdict
    metrics["success_flag"] = "OK" in judge_verdict.upper() or "SUCCESS" in judge_verdict.upper()

    data = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "assistant_model": run_info["model_a"],
        "user_model": run_info["model_b"],
        "task": run_info["task"],
        "rounds": run_info["rounds"],
        "metrics": metrics,
        "stop_reason": info.get("stop_reason") if info else None,
        "usage": info.get("usage") if info else None,
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


#  Core logic 

def run_role_playing() -> tuple[BaseMessage, list[BaseMessage], dict]:
    """Запускает ролевой диалог CAMEL и возвращает (final_msg, dialog, info)."""
    run_info = {
        "assistant_role": getenv("CAMEL_ASSISTANT_ROLE", "Coder"),
        "user_role": getenv("CAMEL_USER_ROLE", "Reviewer"),
        "task": getenv("CAMEL_TASK", "Спроектировать CLI калькулятор и написать тесты."),
        "rounds": getenv("CAMEL_ROUNDS", 8, int),
        "temperature": getenv("CAMEL_TEMPERATURE", 0.3, float),
        "max_tokens": getenv("CAMEL_MAX_TOKENS", 384, int),
        "model_a": getenv("AGENT_MODEL_A", "gpt-4o-mini"),
        "model_b": getenv("AGENT_MODEL_B", "gpt-4o-mini"),
    }

    rp = RolePlaying(
        assistant_role_name=run_info["assistant_role"],
        user_role_name=run_info["user_role"],
        task_prompt=run_info["task"],
        with_task_specify=True,
        assistant_agent_kwargs=dict(model=run_info["model_a"]),
        user_agent_kwargs=dict(model=run_info["model_b"]),
    )

    dialog, info, final_msg = [], {}, None

    # Попытка использовать современный API (run)
    if hasattr(rp, "run"):
        try:
            result = rp.run(n_rounds=run_info["rounds"]) if "n_rounds" in rp.run.__code__.co_varnames else rp.run()
            if isinstance(result, (list, tuple)):
                final_msg = result[0]
                if len(result) > 1:
                    if isinstance(result[1], list):
                        dialog = result[1]
                    elif isinstance(result[1], dict):
                        info = result[1]
            else:
                final_msg = result
        except Exception as e:
            # Если run() не сработал, используем легаси-API
            print(f"Warning: run() failed ({e}), falling back to init_chat/step", file=sys.stderr)
            final_msg, dialog = None, []

    # Легаси-API через init_chat и step
    if final_msg is None and hasattr(rp, "init_chat") and hasattr(rp, "step"):
        init_msg = rp.init_chat()
        assistant_msg = to_base_message(init_msg)
        if assistant_msg:
            dialog.append(assistant_msg)

        for t in range(1, run_info["rounds"] + 1):
            if not isinstance(assistant_msg, BaseMessage):
                break
            step_out = rp.step(assistant_msg)
            if not step_out or not isinstance(step_out, (list, tuple)) or len(step_out) < 2:
                break
            
            # step() возвращает (ChatAgentResponse, ChatAgentResponse)
            assistant_response, user_response = step_out[0], step_out[1]
            
            # Собираем info из ответов
            if assistant_response.info:
                info.update(assistant_response.info)
            if user_response.info:
                info.update(user_response.info)
            
            # Проверяем, не завершился ли диалог
            if assistant_response.terminated or user_response.terminated:
                if assistant_response.msgs:
                    assistant_msg = to_base_message(assistant_response.msgs[0])
                if user_response.msgs:
                    user_msg = to_base_message(user_response.msgs[0])
                    if user_msg:
                        dialog.append(user_msg)
                break
            
            # Извлекаем сообщения из ChatAgentResponse
            assistant_msg = to_base_message(assistant_response.msgs[0]) if assistant_response.msgs else None
            user_msg = to_base_message(user_response.msgs[0]) if user_response.msgs else None
            
            if assistant_msg:
                dialog.append(assistant_msg)
            if user_msg:
                dialog.append(user_msg)
            if should_print_dialog():
                if assistant_msg: print_msg(f"TURN {t} | assistant", assistant_msg)
                if user_msg: print_msg(f"TURN {t} | user", user_msg)

        final_msg = assistant_msg or BaseMessage.make_assistant_message(role_name="assistant", content="(no output)")

    # Если ничего не сработало
    if final_msg is None:
        final_msg = BaseMessage.make_assistant_message(role_name="assistant", content="(no output)")

    if should_print_dialog() and dialog:
        for i, m in enumerate(dialog, 1):
            role = getattr(m, "role_name", "message")
            print_msg(f"TURN {i} | {role}", m)

    return to_base_message(final_msg) or BaseMessage.make_assistant_message(role_name="assistant", content="(no output)"), dialog, info


# Always-on Judge 

def run_judge(final_msg: BaseMessage) -> str:
    """Прогоняет финальный ответ через судью LLM."""
    model = getenv("JUDGE_MODEL", "gpt-4o-mini")
    prompt = (
        "Classify the agent’s final answer as one of the following:\n"
        "SUCCESS — The task is clearly solved.\n"
        "PARTIAL — Some progress but incomplete.\n"
        "FAIL — The task is not solved or irrelevant.\n\n"
        f"Task answer:\n{getattr(final_msg, 'content', '')}\n\n"
        "Also provide 1–2 short recommendations for improvement."
    )

    judge = ChatAgent(model=model)
    resp = judge.step(BaseMessage.make_user_message(role_name="user", content=prompt))
    return getattr(resp.msg, "content", "").strip() or "(no verdict)"


#  Entry point 

def main():
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_BASE")):
        print("Warning: OPENAI_API_KEY not set.", file=sys.stderr)

    final_msg, dialog, info = run_role_playing()
    print("\n=== FINAL MESSAGE ===\n")
    print(getattr(final_msg, "content", "(no output)"))

    print("\n=== JUDGE ===\n")
    judge_text = run_judge(final_msg)
    print(judge_text)

    # Save logs with metrics + verdict
    run_info = {
        "model_a": getenv("AGENT_MODEL_A", "gpt-4o-mini"),
        "model_b": getenv("AGENT_MODEL_B", "gpt-4o-mini"),
        "task": getenv("CAMEL_TASK", "CLI calculator"),
        "rounds": getenv("CAMEL_ROUNDS", 8, int),
    }
    log_dialog(run_info, dialog, final_msg, info, judge_text)


if __name__ == "__main__":
    main()
