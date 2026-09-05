"""Фоновая оценка по критериям методиста.

Отчёт формируется ИНКРЕМЕНТАЛЬНО, а не одним вызовом в конце: после каждой
реплики отдельная сессия дописывает оценку в структурированный JSON. К моменту
`finish` отчёт уже готов и показывается мгновенно.

Фоновая сессия не на критическом пути и не имеет права его задерживать:
своя очередь, свои ошибки глотаются, задержка в секунду там не важна.
Поэтому она живёт в отдельном потоке и НИКОГДА не бросает наружу.
"""
from __future__ import annotations

import json
import queue
import re
import threading
import time
from dataclasses import dataclass, field

from .dialogue import DialogueState
from .scenario import Criterion

SYSTEM = """Ты методист, который оценивает тренировочный диалог по заданным
критериям. Ты НЕ участвуешь в диалоге и ничего не говоришь собеседнику.

Оцени последний обмен репликами по каждому критерию, где есть основание.
Если по критерию в этом обмене ничего не проявилось — не упоминай его.

К каждой оценке ОБЯЗАТЕЛЬНО указывай `turn` — номер той реплики собеседника,
которая тебя к ней привела. Номера стоят в квадратных скобках перед репликами.
Оценка без ссылки на конкретную реплику — это мнение; со ссылкой — разбор, с
которым человек может поспорить, и ради него всё и делается.

Ответ строго в JSON, без пояснений вокруг:
{"scores": [{"criterion": "<ключ>", "score": <число по шкале>, "turn": <номер>,
             "rationale": "<одно предложение, на чём основана оценка>"}]}"""

SUMMARY_SYSTEM = """Ты методист. По накопленным оценкам и расшифровке разговора
напиши общий вывод для отчёта — ДВА РАЗНЫХ ТЕКСТА, у них разные читатели.

Первый читает сам тренируемый. Обращайся к нему на «вы», говори о его работе:
сначала что получилось, потом что мешало, потом одна конкретная рекомендация,
что делать в следующий раз.

Второй читает методист или нанимающий — тот, кто принимает решение и в
разговоре не участвовал. О тренируемом в третьем лице. Ему нужно не
напутствие, а на что смотреть: где человек справился, где просел, и на какую
реплику стоит взглянуть самому. Не пересказывай первый текст другими словами —
это разный жанр: там обратная связь, здесь заключение для решения.

Оба по 2-4 предложения, без списков, без разметки, без цифр «3 из 5» — баллы в
отчёте и так есть, вывод нужен, чтобы их связать.

Ответ строго в JSON, без пояснений вокруг:
{"for_trainee": "<текст тренируемому>", "for_methodist": "<текст методисту>"}"""

_JSON = re.compile(r"\{.*\}", re.S)


@dataclass
class Assessment:
    criterion: str
    score: float
    rationale: str
    # Когда оценка выставлена: индекс последней реплики на момент разбора.
    turn_index: int = -1
    # На чём она основана: индекс реплики В ИСТОРИИ, а не её копия. Копию
    # пришлось бы держать в синхроне с транскриптом, а отчёт должен уметь
    # прокрутить разговор к нужному месту — для этого нужен номер.
    quote_turn: int = -1
    stage_id: str = ""


@dataclass
class EvaluationLog:
    """Накопленные оценки. Живёт рядом с состоянием диалога."""
    assessments: list[Assessment] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    calls: int = 0
    total_ms: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, a: Assessment) -> None:
        with self._lock:
            self.assessments.append(a)

    def for_criterion(self, key: str) -> list[Assessment]:
        with self._lock:
            return [a for a in self.assessments if a.criterion == key]

    def snapshot(self) -> list[Assessment]:
        with self._lock:
            return list(self.assessments)


def build_eval_prompt(state: DialogueState, window: int = 4) -> str:
    sc = state.scenario
    lines = [f"СЦЕНАРИЙ: {sc.title}", "", "КРИТЕРИИ:"]
    for c in sc.criteria:
        lo, hi = c.bounds
        lines.append(f"- {c.key}: {c.title}. Шкала {c.scale}: "
                     f"{lo} — {c.anchor_1}; {hi} — {c.anchor_5}.")
    tail = state.transcript(limit=window)
    base = len(state.turns) - len(tail)          # номера сквозные по истории
    lines += ["", "ПОСЛЕДНИЙ ОБМЕН (в скобках — номер реплики):"]
    for i, t in enumerate(tail):
        who = "Агент" if t.role == "agent" else "Собеседник"
        lines.append(f"[{base + i}] {who}: {t.text}")
    lines += ["", "Оцени ответы СОБЕСЕДНИКА. В каждой оценке укажи turn — "
                  "номер реплики собеседника, на которой она основана. JSON:"]
    return "\n".join(lines)


def parse_scores(blob: str, criteria: list[Criterion],
                 turns: int | None = None) -> list[Assessment]:
    """Разобрать ответ оценщика. Мусор -> пусто: фон не имеет права падать.

    `turns` — сколько реплик в истории. Ссылка за границы отбрасывается: клик
    по цитате, ведущий в пустоту, хуже отсутствующей цитаты.
    """
    if not blob:
        return []
    m = _JSON.search(blob)
    if not m:
        return []
    try:
        d = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return []
    by_key = {c.key: c for c in criteria}
    out = []
    for item in (d.get("scores") or []):
        if not isinstance(item, dict):
            continue
        key = str(item.get("criterion", "")).strip()
        c = by_key.get(key)
        if not c:
            continue                      # придуманный ключ отбрасываем
        try:
            score = float(item.get("score"))
        except (TypeError, ValueError):
            continue
        lo, hi = c.bounds
        if not (lo <= score <= hi):
            continue                      # оценка вне шкалы — не оценка
        quote = -1
        try:
            quote = int(item.get("turn"))
        except (TypeError, ValueError):
            quote = -1
        if quote < 0 or (turns is not None and quote >= turns):
            quote = -1
        out.append(Assessment(key, score, str(item.get("rationale", "")).strip(),
                              quote_turn=quote))
    return out


class BackgroundEvaluator:
    """Очередь оценок в отдельном потоке.

    Ошибки глотаются и копятся в логе: сбой оценщика не должен ни уронить
    диалог, ни задержать его.
    """

    def __init__(self, llm, log: EvaluationLog | None = None, window: int = 4,
                 summary_llm=None):
        self.llm = llm
        # Отдельный клиент под общий вывод: у оценки бюджет 300 токенов, и его
        # хватает — там одно предложение обоснования. Два вывода по 2-4
        # предложения в него не влезают, ответ обрывается на середине JSON, и
        # на экран уезжает сырая фигурная скобка. Поймано на живом прогоне.
        self.summary_llm = summary_llm or llm
        self.log = log or EvaluationLog()
        self.window = window
        self.q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._conclusion: str | None = None
        self._conclusion_lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, state: DialogueState) -> None:
        """Поставить оценку в очередь. Не блокирует диалог."""
        # Последняя реплика пользователя: к ней привязывается оценка, если
        # модель не назвала номер сама.
        last_user = max((i for i, t in enumerate(state.turns) if t.role == "user"),
                        default=len(state.turns) - 1)
        self.q.put({
            "prompt": build_eval_prompt(state, self.window),
            "criteria": list(state.scenario.criteria),
            "turn_index": len(state.turns) - 1,
            "turns": len(state.turns),
            "last_user": last_user,
            "stage_id": state.stage_id,
        })

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            if job is None:
                break
            t0 = time.perf_counter()
            try:
                raw = self.llm(SYSTEM, job["prompt"])
                for a in parse_scores(raw, job["criteria"], job.get("turns")):
                    a.turn_index = job["turn_index"]
                    if a.quote_turn < 0:
                        # Модель не назвала реплику — цитируем последний ответ
                        # пользователя. Он и был поводом для разбора.
                        a.quote_turn = job["last_user"]
                    a.stage_id = job["stage_id"]
                    self.log.add(a)
            except Exception as e:                          # noqa: BLE001
                self.log.errors.append(f"{type(e).__name__}: {e}")
            finally:
                self.log.calls += 1
                self.log.total_ms += (time.perf_counter() - t0) * 1000
                self.q.task_done()

    def conclusion(self, state: DialogueState, timeout: float = 25) -> dict:
        """Общие выводы для отчёта: тренируемому и методисту.

        Один вызов на оба текста, и только на `finish`. Считается один раз и
        запоминается: отчёт опрашивают каждые две секунды, и платить за вывод
        на каждом опросе незачем. Два вызова вместо одного тоже незачем —
        контекст у них общий, отличается только адресат.

        Сбой не оставляет отчёт без вывода: есть сводка по числам. Она хуже
        читается, но она честная и появляется мгновенно.
        """
        if self._conclusion is not None:
            return self._conclusion
        with self._conclusion_lock:
            if self._conclusion is not None:
                return self._conclusion
            texts = {}
            try:
                raw = self.summary_llm(SUMMARY_SYSTEM,
                                       build_summary_prompt(state, self.log))
                texts = parse_conclusions(raw)
            except Exception as e:                          # noqa: BLE001
                self.log.errors.append(f"вывод: {type(e).__name__}: {e}")
            spare = fallback_conclusion(state, self.log)
            self._conclusion = {
                "trainee": texts.get("trainee") or spare,
                # Методисту при сбое достаётся тот же запасной текст: он про
                # числа, а числа одинаковы для обоих читателей.
                "methodist": texts.get("methodist") or texts.get("trainee") or spare,
            }
            return self._conclusion

    def drain(self, timeout: float = 30) -> bool:
        """Дождаться очереди. Зовётся один раз, перед показом отчёта."""
        deadline = time.time() + timeout
        while not self.q.empty() and time.time() < deadline:
            time.sleep(0.05)
        return self.q.empty()

    def close(self) -> None:
        self._stop.set()
        self.q.put(None)


def build_summary_prompt(state: DialogueState, log: EvaluationLog) -> str:
    """Промпт общего вывода: чем закончилось, что накоплено, что говорили."""
    sc = state.scenario
    lines = [f"СЦЕНАРИЙ: {sc.title}",
             f"СОБЕСЕДНИК: {sc.persona.prompt_block()}",
             f"ЭТАПОВ ПРОЙДЕНО: {state.stage_index + 1} из {len(sc.stages)}"
             f" ({state.finish_reason or 'разговор не закончен'})",
             "", "НАКОПЛЕННЫЕ ОЦЕНКИ:"]
    by_key = {c.key: c for c in sc.criteria}
    for c in sc.criteria:
        scores = [a.score for a in log.for_criterion(c.key)]
        avg = round(sum(scores) / len(scores), 2) if scores else None
        lines.append(f"- {c.title}: {avg if avg is not None else 'не оценивалось'}"
                     f" (шкала {c.scale})")
    lines += ["", "РАЗГОВОР:"]
    for t in state.turns:
        lines.append(f"{'Собеседник-тренажёр' if t.role == 'agent' else 'Тренируемый'}: {t.text}")
    lines += ["", "Напиши общий вывод."]
    return "\n".join(lines)


_FIELD = re.compile(r'"(for_trainee|for_methodist)"\s*:\s*"((?:[^"\\]|\\.)*)',
                    re.S)


def parse_conclusions(blob: str) -> dict:
    """Разобрать ответ с двумя выводами. Мусор -> пусто, как и у оценок."""
    if not blob:
        return {}
    m = _JSON.search(blob)
    if not m:
        # Модель ответила простым текстом — считаем его выводом тренируемому:
        # лучше один вывод не тому читателю, чем пустой отчёт. Но если это
        # обрезанный JSON, спасаем из него строки: показать человеку сырую
        # фигурную скобку хуже, чем не показать вывод вовсе.
        return _salvage(blob)
    try:
        d = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return _salvage(blob)
    out = {}
    for key, field in (("trainee", "for_trainee"), ("methodist", "for_methodist")):
        text = " ".join(str(d.get(field, "") or "").split())
        if text:
            out[key] = text
    return out


def _salvage(blob: str) -> dict:
    """Достать выводы из ответа, который не разобрался целиком.

    Обычная причина — обрыв по лимиту токенов: первый текст на месте, второй
    оборван, закрывающей скобки нет. Терять из-за этого весь вывод незачем.
    """
    out = {}
    for name, value in _FIELD.findall(blob):
        text = " ".join(value.replace('\\n', ' ').replace('\\"', '"').split())
        if text:
            out["trainee" if name == "for_trainee" else "methodist"] = text
    if out:
        return out
    text = " ".join(blob.split())
    # Похоже на JSON и не разобралось — прозой это не покажешь.
    if not text or text.lstrip().startswith("{"):
        return {}
    return {"trainee": text}


def fallback_conclusion(state: DialogueState, log: EvaluationLog) -> str:
    """Вывод без модели: из тех же чисел, что уже в отчёте.

    Не украшение, а страховка. Отчёт без общего вывода выглядит незаконченным,
    а разговор к этому моменту уже состоялся — терять его из-за сбойного
    запроса нельзя.
    """
    sc = state.scenario
    rows = []
    for c in sc.criteria:
        scores = [a.score for a in log.for_criterion(c.key)]
        if not scores:
            continue
        lo, hi = c.bounds
        rows.append((c.title, sum(scores) / len(scores), lo, hi))
    if not rows:
        return ("Оценок по критериям не накопилось: разговор оказался слишком "
                "коротким, чтобы что-то заключить.")
    ratio = lambda r: (r[1] - r[2]) / (r[3] - r[2]) if r[3] > r[2] else 0.0  # noqa: E731
    best = max(rows, key=ratio)
    worst = min(rows, key=ratio)
    stages = f"{state.stage_index + 1} из {len(sc.stages)} этапов"
    if best[0] == worst[0]:
        return f"Пройдено {stages}. Оценка есть только по критерию «{best[0]}»: {best[1]:.1f}."
    return (f"Пройдено {stages}. Сильнее всего — «{best[0]}» ({best[1]:.1f}), "
            f"слабее всего — «{worst[0]}» ({worst[1]:.1f}). "
            f"Разбор стоит начать с реплик, отмеченных по второму критерию.")
