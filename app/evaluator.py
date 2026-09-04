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

Ответ строго в JSON, без пояснений вокруг:
{"scores": [{"criterion": "<ключ>", "score": <число по шкале>,
             "rationale": "<одно предложение, на чём основана оценка>"}]}"""

_JSON = re.compile(r"\{.*\}", re.S)


@dataclass
class Assessment:
    criterion: str
    score: float
    rationale: str
    turn_index: int = -1
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
    lines += ["", "ПОСЛЕДНИЙ ОБМЕН:"]
    for t in state.transcript(limit=window):
        who = "Агент" if t.role == "agent" else "Собеседник"
        lines.append(f"{who}: {t.text}")
    lines += ["", "Оцени ответы СОБЕСЕДНИКА. JSON:"]
    return "\n".join(lines)


def parse_scores(blob: str, criteria: list[Criterion]) -> list[Assessment]:
    """Разобрать ответ оценщика. Мусор -> пусто: фон не имеет права падать."""
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
        out.append(Assessment(key, score, str(item.get("rationale", "")).strip()))
    return out


class BackgroundEvaluator:
    """Очередь оценок в отдельном потоке.

    Ошибки глотаются и копятся в логе: сбой оценщика не должен ни уронить
    диалог, ни задержать его.
    """

    def __init__(self, llm, log: EvaluationLog | None = None, window: int = 4):
        self.llm = llm
        self.log = log or EvaluationLog()
        self.window = window
        self.q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, state: DialogueState) -> None:
        """Поставить оценку в очередь. Не блокирует диалог."""
        self.q.put({
            "prompt": build_eval_prompt(state, self.window),
            "criteria": list(state.scenario.criteria),
            "turn_index": len(state.turns) - 1,
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
                for a in parse_scores(raw, job["criteria"]):
                    a.turn_index = job["turn_index"]
                    a.stage_id = job["stage_id"]
                    self.log.add(a)
            except Exception as e:                          # noqa: BLE001
                self.log.errors.append(f"{type(e).__name__}: {e}")
            finally:
                self.log.calls += 1
                self.log.total_ms += (time.perf_counter() - t0) * 1000
                self.q.task_done()

    def drain(self, timeout: float = 30) -> bool:
        """Дождаться очереди. Зовётся один раз, перед показом отчёта."""
        deadline = time.time() + timeout
        while not self.q.empty() and time.time() < deadline:
            time.sleep(0.05)
        return self.q.empty()

    def close(self) -> None:
        self._stop.set()
        self.q.put(None)
