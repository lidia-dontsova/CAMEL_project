# Скрипт запуска упрощённого эксперимента CAMEL: двухагентный ролевой диалог + опциональный «судья»
import os
import sys
from typing import Optional, Any

try:
	# Базовые сущности CAMEL, используемые в эксперименте
	from camel.societies import RolePlaying
	from camel.messages import BaseMessage
	from camel.agents import ChatAgent
except Exception as e:
	# Если пакет не установлен в текущем интерпретаторе
	print("CAMEL is not installed. Please run: pip install camel-ai", file=sys.stderr)
	raise


def _to_base_message(obj: Any) -> Optional[BaseMessage]:
	"""Приводит ответ/объект к BaseMessage, если это ChatAgentResponse и т.п."""
	if obj is None:
		return None
	# Уже BaseMessage
	if isinstance(obj, BaseMessage):
		return obj
	# Ответ с полем .msg
	msg = getattr(obj, "msg", None)
	if isinstance(msg, BaseMessage):
		return msg
	# Ответ с коллекцией .msgs — берём последний элемент
	msgs = getattr(obj, "msgs", None)
	if isinstance(msgs, (list, tuple)) and msgs:
		last = msgs[-1]
		m = getattr(last, "msg", None)
		if isinstance(m, BaseMessage):
			return m
		if isinstance(last, BaseMessage):
			return last
	return None


def getenv_str(name: str, default: str) -> str:
	# Чтение строки из окружения с дефолтом
	val = os.getenv(name)
	return val if val is not None and val.strip() != "" else default


def getenv_int(name: str, default: int) -> int:
	# Чтение целого из окружения с безопасным парсингом
	try:
		return int(getenv_str(name, str(default)))
	except Exception:
		return default


def getenv_float(name: str, default: float) -> float:
	# Чтение числа с плавающей точкой из окружения с безопасным парсингом
	try:
		return float(getenv_str(name, str(default)))
	except Exception:
		return default


def run_role_playing() -> BaseMessage:
	# Роли, задача и лимиты берём из окружения (с дефолтами)
	assistant_role = getenv_str("CAMEL_ASSISTANT_ROLE", "Coder")
	user_role = getenv_str("CAMEL_USER_ROLE", "Reviewer")
	task = getenv_str("CAMEL_TASK", "Спроектировать простой CLI калькулятор и набросать тесты.")

	rounds = getenv_int("CAMEL_ROUNDS", 8)
	max_tokens = getenv_int("CAMEL_MAX_TOKENS", 192)
	temperature = getenv_float("CAMEL_TEMPERATURE", 0.3)

	model_a = getenv_str("AGENT_MODEL_A", "gpt-4o-mini")
	model_b = getenv_str("AGENT_MODEL_B", "gpt-4o-mini")

	# Создаём сценарий ролевого взаимодействия 
	rp = RolePlaying(
		assistant_role_name=assistant_role,
		user_role_name=user_role,
		task_prompt=task,
		with_task_specify=True,
	)

	# API: единым вызовом прогоняет весь диалог
	if hasattr(rp, "run"):
		final_msg, _ = rp.run(n_rounds=rounds) if "n_rounds" in rp.run.__code__.co_varnames else rp.run()
		return final_msg

	# Совместимость со старым API: init_chat() + цикл step(assistant_msg)
	if hasattr(rp, "init_chat"):
		try:
			init_out = rp.init_chat()
		except Exception:
			init_out = None
		assistant_msg = None
		user_msg = None
		# init_chat возвращает (assistant_resp, user_resp)
		if isinstance(init_out, (list, tuple)) and len(init_out) >= 2:
			assistant_msg = _to_base_message(init_out[0]) or init_out[0]
			user_msg = _to_base_message(init_out[1]) or init_out[1]
		else:
			assistant_msg = _to_base_message(init_out) or init_out

		last_assistant: Optional[BaseMessage] = None
		for _ in range(rounds):
			if not hasattr(rp, "step"):
				break
			# Старому API нужен именно BaseMessage на вход
			am = _to_base_message(assistant_msg) or assistant_msg
			if not isinstance(am, BaseMessage):
				break
			step_out = rp.step(am)
			# step возвращает (assistant_resp, user_resp)
			if isinstance(step_out, (list, tuple)) and len(step_out) >= 2:
				assistant_msg = _to_base_message(step_out[0]) or step_out[0]
				user_msg = _to_base_message(step_out[1]) or step_out[1]
				if isinstance(assistant_msg, BaseMessage):
					last_assistant = assistant_msg
			else:
				maybe_assistant = _to_base_message(step_out)
				if isinstance(maybe_assistant, BaseMessage):
					last_assistant = maybe_assistant
					assistant_msg = maybe_assistant
				else:
					break

		return last_assistant if isinstance(last_assistant, BaseMessage) else BaseMessage(role_name="assistant", role_type="assistant", meta_dict={}, content="(no output)")

	# Если ни один путь не сработал
	return BaseMessage(role_name="assistant", role_type="assistant", meta_dict={}, content="(no output)")


def maybe_judge(final_msg: BaseMessage) -> Optional[str]:
	# Однопроходная оценка результата «судьёй» (если задана модель)
	judge_model = os.getenv("JUDGE_MODEL")
	if not judge_model:
		return None

	prompt = (
		"Оцени итоговое решение агента. Дай краткий вердикт (OK/ISSUES) и 1-2 рекомендации.\n\n"
		f"Итоговое сообщение агента:\n{getattr(final_msg, 'content', str(final_msg))}"
	)

	# Простой однопроходный агент-судья (без длинной истории)
	judge = ChatAgent(model_config={
		"model": judge_model,
		"max_tokens": getenv_int("CAMEL_JUDGE_MAX_TOKENS", 256),
		"temperature": getenv_float("CAMEL_JUDGE_TEMPERATURE", 0.2),
	})

	resp = judge.step(prompt)
	return getattr(resp.msg, "content", str(resp.msg))


def main() -> None:
	# Предупреждаем, если ключ/база провайдера не заданы
	if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENAI_API_BASE"):
		print("Warning: OPENAI_API_KEY не задан. Установите переменную окружения перед запуском.", file=sys.stderr)

	# Запуск эксперимента
	final_msg = run_role_playing()
	print("\n=== FINAL MESSAGE ===\n")
	print(getattr(final_msg, "content", str(final_msg)))

	# Опциональный прогон «судьи»
	judge_text = maybe_judge(final_msg)
	if judge_text:
		print("\n=== JUDGE ===\n")
		print(judge_text)


if __name__ == "__main__":
	main()
