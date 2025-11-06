"""
CAMEL Experiment (Extended Version)
-----------------------------------
- Multi-domain task (5 domains × 3 tasks)
- Always-on LLM Judge (classification + 1–10 relevance score)
- Enhanced metrics: flake_ratio, completion_rate, time_per_round
- JSONL logging with domain tagging
"""

import os
import sys
import json
import uuid
import re
import time
from datetime import datetime
from typing import Optional, Any, Dict, Tuple, List

try:
    from camel.societies import RolePlaying
    from camel.messages import BaseMessage
    from camel.agents import ChatAgent
except ImportError:
    print("CAMEL is not installed. Run: pip install camel-ai", file=sys.stderr)
    sys.exit(1)


# ========== Utility helpers ==========

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


def has_task_done_marker(msg: Any) -> bool:
    """Checks if message contains task completion marker."""
    if msg is None:
        return False
    content = getattr(msg, "content", str(msg))
    # Check for exact marker (case-insensitive)
    markers = ["<CAMEL_TASK_DONE>", "<camel_task_done>", "task done", "task completed"]
    content_lower = content.lower()
    return any(marker.lower() in content_lower for marker in markers)


# ========== Metrics ==========

def extract_basic_metrics(dialog: list[BaseMessage], rounds: int, elapsed_time: float, task_done: bool) -> Dict[str, Any]:
    """Compute basic metrics for a dialog."""
    if not dialog:
        return {}
    texts = [str(getattr(m, "content", "")) for m in dialog]
    text_concat = "\n".join(texts)
    n_turns = len(dialog)

    # Improved flake_ratio: count vague or filler phrases
    flake_patterns = re.findall(
        r"\b(I will|I can|I should|let me|I am going to|maybe|perhaps|I think|I guess)\b",
        text_concat,
        re.I,
    )
    flake_ratio = len(flake_patterns) / max(1, n_turns)

    avg_len = sum(len(t) for t in texts) / n_turns
    time_per_round = round(elapsed_time / max(1, rounds), 3)

    return {
        "turns": n_turns,
        "flake_ratio": round(flake_ratio, 3),
        "avg_msg_len": round(avg_len, 1),
        "completion_rate": 1.0 if task_done else 0.0,
        "time_per_round": time_per_round,
    }


# ========== Logging ==========

def log_dialog(run_info: dict, dialog: list[BaseMessage], final_msg: BaseMessage, info: dict, metrics: dict):
    """Saves full experiment results into logs/dialogs.jsonl."""
    os.makedirs("logs", exist_ok=True)

    data = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "assistant_model": run_info["model_a"],
        "user_model": run_info["model_b"],
        "task": run_info["task"],
        "rounds": run_info["rounds"],
        "domain": run_info.get("domain", "Unknown"),
        "metrics": metrics,
        "stop_reason": info.get("stop_reason") if info else None,
        "usage": info.get("usage") if info else None,
        "final": getattr(final_msg, "content", ""),
        "dialog": [
            {"role": getattr(m, "role_name", ""), "content": getattr(m, "content", "")}
            for m in dialog
        ],
    }

    with open("logs/dialogs.jsonl", "a", encoding="utf8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


# ========== Core logic ==========

def run_role_playing(model_a: str, model_b: str, task: str, rounds: int = 8) -> Tuple[BaseMessage, List[BaseMessage], dict, bool, float]:
    """Run CAMEL two-agent role-playing."""
    start_time = time.time()
    task_done = False
    dialog, info, final_msg = [], {}, None

    # Add completion instructions to task
    task_with_completion = (
        f"{task}\n\n"
        "IMPORTANT: When the task is fully completed and the solution is ready, "
        "the Reviewer should write '<CAMEL_TASK_DONE>' to signal task completion. "
        "The Coder should acknowledge completion. After '<CAMEL_TASK_DONE>' is written, "
        "the conversation will stop and the final solution will be extracted."
    )

    # Initialize RolePlaying
    rp = RolePlaying(
        assistant_role_name="Coder",
        user_role_name="Reviewer",
        task_prompt=task_with_completion,
        with_task_specify=True,
        assistant_agent_kwargs=dict(model=model_a),
        user_agent_kwargs=dict(model=model_b),
    )

    # Run conversation (legacy-safe)
    if hasattr(rp, "run"):
        try:
            result = rp.run(n_rounds=rounds) if "n_rounds" in rp.run.__code__.co_varnames else rp.run()
            # Handle different result formats
            if isinstance(result, (list, tuple)):
                if len(result) >= 2:
                    # Try to extract final_msg from first element
                    first = result[0]
                    if hasattr(first, "msgs") and first.msgs:
                        final_msg = to_base_message(first.msgs[0])
                    else:
                        final_msg = to_base_message(first)
                    
                    # Extract dialog from second element
                    second = result[1]
                    if isinstance(second, list):
                        dialog = [to_base_message(m) for m in second if m is not None]
                        # Trim dialog at completion marker (don't include message with marker)
                        for i, msg in enumerate(dialog):
                            if has_task_done_marker(msg):
                                dialog = dialog[:i]
                                task_done = True
                                print(f"\n[Task completion marker detected in dialog at message {i+1} - trimming dialog]", file=sys.stderr)
                                break
                    elif isinstance(second, dict):
                        info = second
            else:
                # Single result - try to extract message
                if hasattr(result, "msgs") and result.msgs:
                    final_msg = to_base_message(result.msgs[0])
                else:
                    final_msg = to_base_message(result)
        except Exception as e:
            print(f"Warning: run() failed ({e}), falling back to init_chat/step", file=sys.stderr)
            final_msg, dialog = None, []

    # Fallback to legacy API if needed
    if (final_msg is None or not getattr(to_base_message(final_msg), "content", "").strip()) and hasattr(rp, "init_chat") and hasattr(rp, "step"):
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

            # Extract messages from current step
            next_assistant = to_base_message(assistant_response.msgs[0]) if getattr(assistant_response, "msgs", None) else None
            next_user = to_base_message(user_response.msgs[0]) if getattr(user_response, "msgs", None) else None

            # Check for completion marker - stop dialog if found
            if has_task_done_marker(next_user) or has_task_done_marker(next_assistant):
                task_done = True
                print(f"\n[Task completion marker detected at turn {t} - stopping dialog]", file=sys.stderr)
                break

            # Add messages to dialog
            if next_user:
                dialog.append(next_user)
            if next_assistant:
                dialog.append(next_assistant)

            assistant_msg = next_assistant

            # Check termination
            if getattr(assistant_response, "terminated", False) or getattr(user_response, "terminated", False):
                break

        final_msg = assistant_msg or BaseMessage.make_assistant_message(role_name="assistant", content="(no output)")

    # Extract final message from dialog if still missing
    if final_msg is None or not getattr(to_base_message(final_msg), "content", "").strip():
        # Look for last meaningful assistant message
        farewell_phrases = [
            "have a great day", "have a wonderful day", "free to ask", "need further assistance",
            "more requests", "more tasks", "any more", "future tasks"
        ]
        solution_keywords = ["solution:", "code:", "def ", "import ", "class ", "```"]
        
        # First, look for solution message
        for msg in reversed(dialog):
            normalized = to_base_message(msg)
            if normalized and getattr(normalized, "role_name", "").lower() in ("assistant", "coder"):
                content = getattr(normalized, "content", "").strip()
                content_lower = content.lower()
                if content and not any(phrase in content_lower for phrase in farewell_phrases):
                    if any(keyword in content_lower for keyword in solution_keywords):
                        final_msg = normalized
                        break
        
        # If no solution found, get last non-farewell assistant message
        if final_msg is None or not getattr(to_base_message(final_msg), "content", "").strip():
            for msg in reversed(dialog):
                normalized = to_base_message(msg)
                if normalized and getattr(normalized, "role_name", "").lower() in ("assistant", "coder"):
                    content = getattr(normalized, "content", "").strip()
                    content_lower = content.lower()
                    if content and not any(phrase in content_lower for phrase in farewell_phrases):
                        final_msg = normalized
                        break

    # Detect completion marker
    if not task_done:
        for msg in dialog:
            if has_task_done_marker(msg):
                task_done = True
                break

    elapsed = time.time() - start_time
    final_msg = final_msg or BaseMessage.make_assistant_message(role_name="assistant", content="(no output)")
    return to_base_message(final_msg) or BaseMessage.make_assistant_message(role_name="assistant", content="(no output)"), dialog, info, task_done, elapsed


# ========== Solution Summarization ==========

def summarize_dialog_solution(dialog: List[BaseMessage], task: str) -> BaseMessage:
    """Summarize the entire dialog and extract the final solution."""
    if not dialog:
        return BaseMessage.make_assistant_message(role_name="assistant", content="(no dialog)")
    
    # Collect all assistant (Coder) messages
    assistant_messages = []
    for msg in dialog:
        normalized = to_base_message(msg)
        if normalized:
            role = getattr(normalized, "role_name", "").lower()
            if role in ("assistant", "coder"):
                content = getattr(normalized, "content", "").strip()
                if content:
                    assistant_messages.append(content)
    
    if not assistant_messages:
        return BaseMessage.make_assistant_message(role_name="assistant", content="(no assistant messages)")
    
    # Combine all assistant messages
    full_dialog_text = "\n\n---\n\n".join(assistant_messages)
    
    # Use LLM to summarize and extract final solution
    model = getenv("JUDGE_MODEL", "gpt-4o-mini")
    prompt = (
        "Analyze the following conversation between a Coder and Reviewer working on a task.\n"
        "Extract and summarize the FINAL COMPLETE SOLUTION to the task.\n\n"
        f"Task: {task}\n\n"
        f"Full conversation:\n{full_dialog_text}\n\n"
        "Instructions:\n"
        "1. Identify the final, complete solution (code, answer, or result)\n"
        "2. Include all necessary code blocks, explanations, and final answers\n"
        "3. Remove intermediate steps, questions, and discussion - only keep the final solution\n"
        "4. If the solution includes code, preserve the complete, runnable code\n"
        "5. If the solution is an explanation or answer, provide the complete final answer\n"
        "6. Format clearly with proper structure\n\n"
        "Provide ONLY the final solution, without meta-commentary about the conversation."
    )
    
    try:
        summarizer = ChatAgent(model=model)
        resp = summarizer.step(BaseMessage.make_user_message(role_name="user", content=prompt))
        summary = getattr(resp.msg, "content", "").strip()
        
        if summary:
            return BaseMessage.make_assistant_message(role_name="assistant", content=summary)
    except Exception as e:
        print(f"Warning: Solution summarization failed ({e}), using last assistant message", file=sys.stderr)
    
    # Fallback: extract code blocks and key solution parts from last messages
    solution_parts = []
    code_blocks = re.findall(r"```(?:python|javascript|java|c\+\+|go|rust|sql)?\n(.*?)```", full_dialog_text, re.DOTALL)
    if code_blocks:
        solution_parts.append("```python\n" + code_blocks[-1] + "\n```")
    
    # Look for solution keywords in last few messages
    for msg_text in reversed(assistant_messages[-3:]):
        if any(keyword in msg_text.lower() for keyword in ["solution:", "final:", "answer:", "result:"]):
            # Extract text after solution marker
            match = re.search(r"(?:solution|final|answer|result)[:\s]*(.*)", msg_text, re.I | re.DOTALL)
            if match:
                solution_parts.append(match.group(1).strip())
                break
    
    if solution_parts:
        return BaseMessage.make_assistant_message(role_name="assistant", content="\n\n".join(solution_parts))
    
    # Last resort: return last assistant message
    return BaseMessage.make_assistant_message(
        role_name="assistant", 
        content=assistant_messages[-1] if assistant_messages else "(no solution found)"
    )


# ========== Judge ==========

def run_judge(final_msg: BaseMessage, task: str) -> Tuple[str, float]:
    """Judge model's final answer: classification + 1–10 relevance score."""
    model = getenv("JUDGE_MODEL", "gpt-4o-mini")
    answer = getattr(final_msg, "content", "")

    prompt = (
        "Evaluate the assistant's final answer for task relevance.\n"
        "Return two parts:\n"
        "1️. Classification: SUCCESS / PARTIAL / FAIL.\n"
        "2️. Numerical score (1–10) for how relevant and correct the answer is.\n\n"
        f"Task:\n{task}\n\n"
        f"Answer:\n{answer}\n\n"
        "Respond in format:\n"
        "Classification: <label>\n"
        "Score: <number>\n"
        "Comments: <1–2 brief improvement tips>."
    )

    try:
        judge = ChatAgent(model=model)
        resp = judge.step(BaseMessage.make_user_message(role_name="user", content=prompt))
        text = getattr(resp.msg, "content", "").strip()

        verdict_match = re.search(r"Classification:\s*(\w+)", text, re.I)
        score_match = re.search(r"Score:\s*(\d+(?:\.\d+)?)", text)
        verdict = verdict_match.group(1).upper() if verdict_match else "UNKNOWN"
        score = float(score_match.group(1)) if score_match else 0.0

        return verdict, score
    except Exception as e:
        print(f"Warning: Judge failed ({e}), returning default verdict", file=sys.stderr)
        return "FAIL", 0.0


# ========== Tasks & Main loop ==========

task_groups = {
    "Programming": [
        "Implement a Python function that returns the n-th Fibonacci number.",
        "Write a program that removes duplicate lines from a text file.",
        "Design a simple REST API for user registration and authentication.",
    ],
    "Language": [
        "Explain the meaning and usage of the English idiom 'break the ice'.",
        "Translate the Russian phrase 'Он шел по улице и думал о будущем' into English, preserving style.",
        "Explain what presupposition means in semantics and give an example.",
    ],
    "Math_Logic": [
        "Solve the system of equations: 2x + y = 10, x - y = 4.",
        "If all mammals are warm-blooded and whales are mammals, what can we infer about whales?",
        "Find the derivative of f(x) = 3x^2 + 2x - 5 and explain each step.",
    ],
    "STEM_Explanation": [
        "Explain how a neural network learns from data in simple terms.",
        "Describe how solar panels convert sunlight into electricity.",
        "Outline the basic architecture of an autonomous driving system.",
    ],
    "Role_Playing": [
        "Simulate a doctor explaining a diagnosis to a patient with mild anxiety.",
        "Act as an HR recruiter interviewing a candidate for a data analyst position.",
        "Role-play a conversation between a buyer and seller negotiating the price of a used laptop.",
    ],
}


def main():
    try:
        if not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_BASE")):
            print("Warning: OPENAI_API_KEY not set.", file=sys.stderr)

        rounds = getenv("CAMEL_ROUNDS", 8, int)

        models_to_test = [
            "gpt-4o-mini",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0125",
        ]

        successful, failed = 0, 0

        for domain, tasks in task_groups.items():
            print(f"\n{'='*80}")
            print(f"DOMAIN: {domain}")
            print(f"{'='*80}")

            for task in tasks:
                for model in models_to_test:
                    print(f"\n==============================")
                    print(f"Running {model} on task: {task[:60]}...")
                    print(f"==============================")

                    try:
                        _, dialog, info, task_done, elapsed = run_role_playing(model, model, task, rounds)
                    except Exception as e:
                        failed += 1
                        print(f"Error running {model}: {e}")
                        continue

                    # Summarize dialog to extract final solution
                    final_msg = summarize_dialog_solution(dialog, task)
                    verdict, relevance = run_judge(final_msg, task)

                    metrics = extract_basic_metrics(dialog, rounds, elapsed, task_done)
                    metrics["judge_verdict"] = verdict
                    metrics["relevance_score"] = relevance
                    metrics["success_flag"] = verdict == "SUCCESS"
                    metrics["domain"] = domain

                    run_info = {
                        "model_a": model,
                        "model_b": model,
                        "task": task,
                        "rounds": rounds,
                        "domain": domain,
                    }

                    log_dialog(run_info, dialog, final_msg, info, metrics)

                    # Print full dialog
                    if dialog:
                        print(f"\n=== FULL DIALOG ({len(dialog)} messages) ===")
                        for i, msg in enumerate(dialog, 1):
                            role = getattr(msg, "role_name", "unknown")
                            print_msg(f"TURN {i} | {role}", msg)
                    
                    print(f"\n=== FINAL MESSAGE ===\n{getattr(final_msg, 'content', '(no output)')}")
                    print(f"\n=== JUDGE ===\nVerdict: {verdict} | Score: {relevance}/10")
                    print(f"Metrics: {json.dumps(metrics, ensure_ascii=False, indent=2)}")

                    successful += 1

        print(f"\n{'='*80}")
        print(f"Summary: {successful} successful, {failed} failed")
        print(f"{'='*80}")

    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error in main(): {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
