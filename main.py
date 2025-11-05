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
from typing import Optional, Any, Dict, Tuple, List

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


def has_task_done_marker(msg: Any) -> bool:
    """Проверяет, содержит ли сообщение маркер завершения задачи."""
    if msg is None:
        return False
    content = getattr(msg, "content", str(msg)).lower()
    task_done_markers = ["<CAMEL_TASK_DONE>", "camel_task_done", "task done", "task completed"]
    return any(marker in content for marker in task_done_markers)


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

def run_role_playing(model_a: str, model_b: str, task: str, rounds: int = 8) -> Tuple[BaseMessage, List[BaseMessage], dict]:
    """Запускает ролевой диалог CAMEL и возвращает (final_msg, dialog, info)."""
    # Пробуем сначала с task_specify, если не получается - без него
    rp = None
    try:
        rp = RolePlaying(
            assistant_role_name="Coder",
            user_role_name="Reviewer",
            task_prompt=task,
            with_task_specify=True,
            assistant_agent_kwargs=dict(model=model_a),
            user_agent_kwargs=dict(model=model_b),
        )
    except Exception as e:
        error_str = str(e).lower()
        # Если ошибка доступа (403, региональная), пробуем без task_specify
        if "403" in str(e) or "unsupported_country" in error_str or "permission denied" in error_str:
            print(f"Warning: TaskSpecify failed ({e}), trying without it", file=sys.stderr)
            try:
                # Добавляем инструкции о завершении задачи в task_prompt, если task_specify отключен
                task_with_completion = task
                if "<CAMEL_TASK_DONE>" not in task:
                    task_with_completion = (
                        f"{task}\n\n"
                        "IMPORTANT: When the task is completed, the Reviewer should send "
                        "<CAMEL_TASK_DONE> to indicate task completion. The Coder should acknowledge completion."
                    )
                rp = RolePlaying(
                    assistant_role_name="Coder",
                    user_role_name="Reviewer",
                    task_prompt=task_with_completion,
                    with_task_specify=False,  # Отключаем task_specify
                    assistant_agent_kwargs=dict(model=model_a),
                    user_agent_kwargs=dict(model=model_b),
                )
            except Exception as e2:
                raise RuntimeError(f"Failed to initialize RolePlaying even without task_specify: {e2}") from e2
        else:
            raise RuntimeError(f"Failed to initialize RolePlaying: {e}") from e
    
    if rp is None:
        raise RuntimeError("Failed to create RolePlaying instance")

    dialog, info, final_msg = [], {}, None

    
    if hasattr(rp, "run"):
        try:
            result = rp.run(n_rounds=rounds) if "n_rounds" in rp.run.__code__.co_varnames else rp.run()
            # Возможные формы результата:
            # - BaseMessage
            # - ChatAgentResponse
            # - (assistant_response, user_response)
            # - (final_msg, dialog) или (final_msg, info)
            if isinstance(result, (list, tuple)):
                first = result[0]
                # Если первый элемент — ChatAgentResponse
                if hasattr(first, "msgs"):
                    msgs = getattr(first, "msgs", None)
                    if msgs:
                        final_msg = to_base_message(msgs[0])
                else:
                    final_msg = to_base_message(first) or first

                # Второй элемент может быть списком сообщений (диалог) или словарём info
                if len(result) > 1:
                    second = result[1]
                    if isinstance(second, list):
                        # Нормализуем элементы во второй позиции к BaseMessage
                        dialog = [to_base_message(m) or m for m in second if m is not None]
                        # Обрезаем диалог до маркера завершения, если он есть
                        for i, msg in enumerate(dialog):
                            if has_task_done_marker(msg):
                                dialog = dialog[:i]
                                break
                    elif isinstance(second, dict):
                        info = second
            else:
                # Единичный результат
                if hasattr(result, "msgs"):
                    msgs = getattr(result, "msgs", None)
                    if msgs:
                        final_msg = to_base_message(msgs[0])
                else:
                    final_msg = to_base_message(result) or result
        except Exception as e:
            print(f"Warning: run() failed ({e}), falling back to init_chat/step", file=sys.stderr)
            final_msg, dialog = None, []

    # Если современный API не дал результата (нет сообщения или пустой контент) — используем легаси-цикл
    need_legacy = (
        final_msg is None
        or not getattr(to_base_message(final_msg), "content", "").strip()
    )

    if need_legacy and hasattr(rp, "init_chat") and hasattr(rp, "step"):
        dialog = []
        info = {}
        try:
            init_msg = rp.init_chat()
            assistant_msg = to_base_message(init_msg)
            if assistant_msg:
                dialog.append(assistant_msg)
        except Exception as e:
            print(f"Warning: init_chat() failed: {e}", file=sys.stderr)
            assistant_msg = None

        for t in range(1, rounds + 1):
            if not isinstance(assistant_msg, BaseMessage):
                break
            try:
                step_out = rp.step(assistant_msg)
            except Exception as e:
                print(f"Warning: step() failed at turn {t}: {e}", file=sys.stderr)
                break
            if not step_out or not isinstance(step_out, (list, tuple)) or len(step_out) < 2:
                break

            assistant_response, user_response = step_out[0], step_out[1]

            if getattr(assistant_response, "info", None):
                info.update(assistant_response.info)
            if getattr(user_response, "info", None):
                info.update(user_response.info)

            # Достаём сообщения текущего шага
            next_assistant = to_base_message(assistant_response.msgs[0]) if getattr(assistant_response, "msgs", None) else None
            next_user = to_base_message(user_response.msgs[0]) if getattr(user_response, "msgs", None) else None

            # Проверяем наличие маркера завершения задачи ПЕРЕД добавлением в диалог
            if has_task_done_marker(next_user) or has_task_done_marker(next_assistant):
                # Маркер найден - не добавляем сообщения с маркером, останавливаемся
                break

            # Добавляем сообщения в диалог только если маркера нет
            if next_user:
                dialog.append(next_user)
            if next_assistant:
                dialog.append(next_assistant)

            if should_print_dialog():
                if next_user: print_msg(f"TURN {t} | user", next_user)
                if next_assistant: print_msg(f"TURN {t} | assistant", next_assistant)

            assistant_msg = next_assistant

            # Проверяем завершение
            if getattr(assistant_response, "terminated", False) or getattr(user_response, "terminated", False):
                break

        final_msg = assistant_msg or BaseMessage.make_assistant_message(role_name="assistant", content="(no output)")

    # Если final_msg не установлен или пустой, извлекаем последнее значимое сообщение ассистента из диалога
    if final_msg is None or not getattr(to_base_message(final_msg), "content", "").strip():
        # Ищем последнее значимое сообщение ассистента в диалоге (исключая прощальные фразы)
        farewell_phrases = [
            "have a great day", "have a wonderful day", "free to ask", "need further assistance",
            "more requests", "more tasks", "any more", "future tasks"
        ]
        
        # Ключевые слова, указывающие на решение задачи
        solution_keywords = ["solution:", "code:", "def ", "import ", "class ", "```"]
        
        # Сначала ищем сообщение с решением (код, решение задачи)
        for msg in reversed(dialog):
            normalized = to_base_message(msg)
            if normalized and getattr(normalized, "role_name", "").lower() in ("assistant", "coder"):
                content = getattr(normalized, "content", "").strip()
                content_lower = content.lower()
                # Пропускаем прощальные сообщения
                if content and not any(phrase in content_lower for phrase in farewell_phrases):
                    # Если есть признаки решения (код, ключевые слова), берём это сообщение
                    if any(keyword in content_lower for keyword in solution_keywords):
                        final_msg = normalized
                        break
        
        # Если не нашли решение, берём последнее не прощальное сообщение ассистента
        if final_msg is None or not getattr(to_base_message(final_msg), "content", "").strip():
            for msg in reversed(dialog):
                normalized = to_base_message(msg)
                if normalized and getattr(normalized, "role_name", "").lower() in ("assistant", "coder"):
                    content = getattr(normalized, "content", "").strip()
                    content_lower = content.lower()
                    if content and not any(phrase in content_lower for phrase in farewell_phrases):
                        final_msg = normalized
                        break

    # Если всё ещё нет финального сообщения, создаём заглушку
    if final_msg is None or not getattr(to_base_message(final_msg), "content", "").strip():
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
        "Classify the agent's final answer as one of the following:\n"
        "SUCCESS — The task is clearly solved.\n"
        "PARTIAL — Some progress but incomplete.\n"
        "FAIL — The task is not solved or irrelevant.\n\n"
        f"Task answer:\n{getattr(final_msg, 'content', '')}\n\n"
        "Also provide 1–2 short recommendations for improvement."
    )

    try:
        judge = ChatAgent(model=model)
        resp = judge.step(BaseMessage.make_user_message(role_name="user", content=prompt))
        return getattr(resp.msg, "content", "").strip() or "(no verdict)"
    except Exception as e:
        print(f"Warning: Judge failed ({e}), returning default verdict", file=sys.stderr)
        return "FAIL — Judge unavailable. Error: " + str(e)


#  Entry point 

def main():
    try:
        if not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_BASE")):
            print("Warning: OPENAI_API_KEY not set.", file=sys.stderr)

        task = getenv("CAMEL_TASK", "Design a simple CLI calculator with tests.")
        rounds = getenv("CAMEL_ROUNDS", 8, int)

        models_to_test = [
            "gpt-4o-mini",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0125",
        ]

        successful = 0
        failed = 0

        for model in models_to_test:
            print(f"\n==============================")
            print(f"Running experiment for model: {model}")
            print(f"==============================")

            try:
                final_msg, dialog, info = run_role_playing(model, model, task, rounds)
            except Exception as e:
                failed += 1
                error_msg = str(e)
                # Если это региональная ошибка или ошибка доступа, пропускаем модель
                if ("unsupported_country" in error_msg.lower() or 
                    "permission denied" in error_msg.lower() or 
                    "insufficient permissions" in error_msg.lower() or
                    "403" in error_msg or 
                    "401" in error_msg):
                    print(f"\nSkipping {model}: Access denied (region/API issue or insufficient permissions)", file=sys.stderr)
                    print(f"   Error details: {error_msg[:200]}", file=sys.stderr)
                    print(f"   This model is not available in your region or API access is restricted.\n")
                else:
                    print(f"\nError running experiment for {model}: {e}", file=sys.stderr)
                    import traceback
                    print("   Traceback:", file=sys.stderr)
                    traceback.print_exc()
                continue
            
            try:
                print("\n=== FINAL MESSAGE ===\n")
                print(getattr(final_msg, "content", "(no output)"))

                print("\n=== JUDGE ===\n")
                judge_text = run_judge(final_msg)
                print(judge_text)

                run_info = {
                    "model_a": model,
                    "model_b": model,
                    "task": task,
                    "rounds": rounds,
                }
                log_dialog(run_info, dialog, final_msg, info, judge_text)

                print(f"\nExperiment for {model} logged to logs/dialogs.jsonl")
                successful += 1
            except Exception as e:
                failed += 1
                print(f"Error processing results for {model}: {e}", file=sys.stderr)
                continue
        
        print(f"\n{'='*60}")
        print(f"Summary: {successful} successful, {failed} failed")
        print(f"{'='*60}")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error in main(): {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
