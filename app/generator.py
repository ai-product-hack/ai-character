"""Сценарий из произвольного текста методиста.

Кейс называется «оживи инструкцию»: методист вставляет ТЗ, вакансию, скрипт
продаж или регламент, и из этого получается тренировка. Один вызов модели даёт
ТРИ вещи, а не только этапы:

    персона   кто ведёт разговор — и есть «персонаж из текста»
    этапы     5-7 штук, у каждого цель, подсказка, условие перехода, бюджет
    критерии  требования из текста и есть критерии оценки

Персона — самая недооценённая часть. Из вакансии на Rust-разработчика и из
скрипта холодных продаж должны получиться заметно разные собеседники, иначе
«персонаж из текста» остаётся словами.

Надёжность. Это единственное место, где сломанный ответ модели виден залу,
поэтому защиты три и они складываются:

    1. строгая схема на выходе (`output_config.format`) — модель физически не
       может вернуть не тот JSON;
    2. валидация уже собранного сценария, и при провале — повтор с текстом
       ошибки в промпте, до двух раз;
    3. фолбэк на шаблон по типу тренировки с честным сообщением методисту.

Ключи критериев и id этапов генерируются ЗДЕСЬ, транслитерацией из названия:
HR не должен придумывать идентификаторы, а модель, которой это поручили,
возвращает то `communication`, то `коммуникация`, то `crit_2`.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field

from . import translit
from .emotion_tags import EMOTIONS
from .llm import load_env
from .scenario import Scenario
from .templates import TYPES, template

# Sonnet, а не Opus: задача разовая, ответ короткий и жёстко ограничен схемой,
# и платить за неё по верхнему тарифу незачем. Меняется одной строкой в
# app/config.json.
MODEL = "claude-sonnet-5"
MAX_TOKENS = 8000
# Усилие ниже умолчания. Задача узкая, форма ответа задана схемой, а разница в
# секундах видна методисту напрямую: генерация идёт до диалога, и он смотрит на
# индикатор всё это время.
EFFORT = "medium"
ATTEMPTS = 3            # первая попытка плюс два повтора с текстом ошибки

SYSTEM = """Ты методист-разработчик тренажёров общения. Из произвольного текста
— техзадания, описания вакансии, скрипта продаж, регламента, требований
свободным текстом — ты собираешь сценарий тренировочного диалога.

Собираешь ТРИ вещи.

ПЕРСОНА — кто ведёт разговор со стороны тренажёра. Она вытекает из текста: из
вакансии получается интервьюер по этой специальности, из скрипта продаж —
клиент, из регламента — проверяющий. Персона это НЕ тренируемый человек, а его
собеседник. Пиши роль конкретно: не «интервьюер», а «технический лид, который
собеседует на позицию Rust-разработчика в команду системного ПО».

ЭТАПЫ — от 5 до 7. Каждый двигает разговор дальше и отличается от соседних по
тому, ЧТО делает собеседник, а не только по теме. Подсказка (`hint`) написана
в повелительном наклонении, обращена к модели, играющей персону, и говорит,
как себя вести на этом этапе. Условие перехода (`advance_when`) — словами, что
должен сделать тренируемый, чтобы этап закрылся. `opening` — конкретная
реплика, которой этап можно открыть. `max_turns` — сколько обменов репликами
этап стоит: простой этап 2, требующий раскрытия 3-4.

Стартовая эмоция персоны выбирается из семи. Пять — шкала отношения к
тренируемому: neutral, skeptical, pressing, warming, impressed. Две отдельные:
`angry` — открытый гнев, когда роль разозлена по существу (сорванный срок,
испорченная работа); `anxious` — тревога и растерянность, когда роль волнуется
или боится. Не подменяй их на `pressing`: давление — это про напор на
собеседника, а гнев и тревога — про состояние самой роли.

КРИТЕРИИ — от 3 до 6, вытащенных ИЗ ТЕКСТА. Требования вакансии, пункты
регламента, обязательные шаги скрипта и есть критерии. Не пиши универсальных
«коммуникация» и «профессионализм», если в тексте есть что-то конкретнее.
Якоря пиши наблюдаемым поведением, а не оценкой: «сказал, что не знает, и
предложил, как выяснить», а не «плохо».

Язык всех полей — русский. Идентификаторы не придумывай, их не просят."""

# Стартовая эмоция берётся из того же белого списка, что и разметка [emo:...]:
# эмоциональный слой аватара получает ровно те пять состояний, что и раньше.
SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string",
                  "description": "название сценария, как его увидит методист"},
        "type": {"type": "string", "enum": list(TYPES)},
        "persona": {
            "type": "object",
            "properties": {
                "role": {"type": "string"},
                "tone": {"type": "string"},
                "strictness": {"type": "string"},
                "pressure": {"type": "string"},
                "start_emotion": {"type": "string", "enum": list(EMOTIONS)},
            },
            "required": ["role", "tone", "strictness", "pressure", "start_emotion"],
            "additionalProperties": False,
        },
        # Числовых границ в схеме нет намеренно. Структурированный вывод их
        # не принимает и отвечает 400: «For 'array' type, 'minItems' values
        # other than 0 or 1 are not supported» и «For 'integer' type,
        # properties maximum, minimum are not supported». Количество этапов и
        # критериев требуется словами в системном промпте и проверяется в
        # `extra_checks`; бюджет хода зажимается в `to_scenario`.
        "stages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "goal": {"type": "string"},
                    "hint": {"type": "string"},
                    "advance_when": {"type": "string"},
                    "opening": {"type": "string"},
                    "max_turns": {"type": "integer"},
                },
                "required": ["goal", "hint", "advance_when", "opening", "max_turns"],
                "additionalProperties": False,
            },
        },
        "criteria": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "anchor_1": {"type": "string"},
                    "anchor_5": {"type": "string"},
                },
                "required": ["title", "anchor_1", "anchor_5"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["title", "type", "persona", "stages", "criteria"],
    "additionalProperties": False,
}


class AnthropicGenerator:
    """Один вызов модели со строгой схемой на выходе."""

    def __init__(self, model: str = MODEL, max_tokens: int = MAX_TOKENS,
                 timeout: float = 90, effort: str = EFFORT):
        load_env()
        if not os.environ.get("ANTHROPIC_API_KEY"):
            # RuntimeError, а не SystemExit: генератор живёт в потоке сервера,
            # а не в скрипте. SystemExit наследуется от BaseException, мимо
            # `except Exception` проходит насквозь — задание оставалось в
            # состоянии «идёт» навсегда, и методист смотрел на крутящийся
            # индикатор вместо шаблона.
            raise RuntimeError("нет ANTHROPIC_API_KEY в окружении или .env")
        import anthropic                      # импорт здесь: пакет нужен только тут
        self.client = anthropic.Anthropic(timeout=timeout)
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.calls = 0
        self.tokens_in = 0
        self.tokens_out = 0

    def __call__(self, prompt: str) -> dict:
        r = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": {"type": "json_schema", "schema": SCHEMA},
                           "effort": self.effort},
        )
        self.calls += 1
        self.tokens_in += r.usage.input_tokens
        self.tokens_out += r.usage.output_tokens
        # Схема гарантирует, что первый текстовый блок — валидный JSON нужной
        # формы. Разбор всё равно в try у вызывающего: гарантия сервера не
        # повод ронять интерфейс, если сервер однажды ответит иначе.
        text = next(b.text for b in r.content if b.type == "text")
        return json.loads(text)


def build_prompt(source_text: str, kind: str = "", error: str = "") -> str:
    """Промпт генерации. `error` — то, чем не понравился прошлый ответ."""
    lines = []
    if kind and kind in TYPES:
        lines.append(f"Тип тренировки, который выбрал методист: {TYPES[kind]}. "
                     f"Это подсказка, а не приказ: если текст говорит о другом, "
                     f"доверяй тексту.")
    lines += ["ТЕКСТ МЕТОДИСТА:", "", source_text.strip(), ""]
    if error:
        lines += [
            "ПРЕДЫДУЩАЯ ПОПЫТКА НЕ ПРОШЛА ПРОВЕРКУ:",
            error,
            "Исправь именно это и верни сценарий целиком.",
            "",
        ]
    lines.append("Собери сценарий: персона, этапы, критерии.")
    return "\n".join(lines)


def _scenario_id(title: str, kind: str) -> str:
    base = translit.slug(title, kind or "scenario", limit=32)
    return f"{base}_{uuid.uuid4().hex[:6]}"


def to_scenario(raw: dict, source_text: str = "", source: str = "generated",
                scenario_id: str = "") -> Scenario:
    """Ответ модели -> сценарий движка.

    Здесь появляются идентификаторы: id этапов и ключи критериев считаются
    транслитерацией из названий, а не берутся у модели. Повторы разводятся
    суффиксом — совпавший ключ уводит оценку не к тому критерию, и в отчёте
    это выглядит как выдуманный моделью балл.
    """
    kind = raw.get("type") if raw.get("type") in TYPES else "generic"
    stages_raw = raw.get("stages") or []
    criteria_raw = raw.get("criteria") or []

    stage_ids = translit.unique_slugs([s.get("goal", "") for s in stages_raw], "stage")
    criteria_keys = translit.unique_slugs([c.get("title", "") for c in criteria_raw], "crit")

    stages = []
    for i, s in enumerate(stages_raw):
        budget = s.get("max_turns")
        try:
            budget = int(budget)
        except (TypeError, ValueError):
            budget = None
        # Потолок жёсткий: модель охотно ставит по пять ходов каждому этапу, а
        # семь этапов по пять — это диалог, который не заканчивается за показ.
        if budget is not None:
            budget = max(2, min(4, budget))
        stages.append({
            "id": stage_ids[i],
            "goal": (s.get("goal") or "").strip(),
            "hint": (s.get("hint") or "").strip(),
            "advance_when": (s.get("advance_when") or "").strip(),
            "opening": (s.get("opening") or "").strip(),
            "max_turns": budget,
        })

    criteria = [{
        "key": criteria_keys[i],
        "title": (c.get("title") or "").strip(),
        "scale": "1-5",
        "anchor_1": (c.get("anchor_1") or "").strip(),
        "anchor_5": (c.get("anchor_5") or "").strip(),
    } for i, c in enumerate(criteria_raw)]

    title = (raw.get("title") or "").strip() or "Тренировка без названия"
    return Scenario.from_dict({
        "id": scenario_id or _scenario_id(title, kind),
        "title": title,
        "type": kind,
        "persona": raw.get("persona") or {},
        "stages": stages,
        "criteria": criteria,
        "source": source,
        "source_text": source_text or raw.get("source_text", ""),
    })


def fill_identifiers(raw: dict) -> dict:
    """Дописать недостающие id этапов и ключи критериев.

    Нужно не только генератору. Методист, добавивший критерий руками в панели,
    оставляет ключ пустым — придумывать идентификаторы его никто не просил, и
    в этом весь смысл. Пустой ключ роняет approve, а повторившийся уводит
    оценку не к тому критерию, поэтому чинятся оба случая.

    Уже заданные ключи сохраняются: сценарий из репозитория не должен менять
    идентификаторы от того, что его открыли в панели.
    """
    out = dict(raw)
    for field, items, prefix, name_key in (
            ("stages", raw.get("stages") or [], "stage", "goal"),
            ("criteria", raw.get("criteria") or [], "crit", "title")):
        id_key = "id" if field == "stages" else "key"
        fresh = translit.unique_slugs([i.get(name_key, "") for i in items], prefix)
        seen: set[str] = set()
        fixed = []
        for i, item in enumerate(items):
            item = dict(item)
            key = str(item.get(id_key) or "").strip()
            if not key or key in seen or not key.isascii() \
                    or not key.replace("_", "").isalnum():
                key = fresh[i]
                while key in seen:
                    key += "_2"
            seen.add(key)
            item[id_key] = key
            fixed.append(item)
        out[field] = fixed
    return out


def extra_checks(sc: Scenario) -> list[str]:
    """Проверки сверх формальной валидации сценария.

    Схема гарантирует форму, но не содержание: модель умеет вернуть пять
    этапов с пустыми подсказками и персону в две буквы. Такой сценарий
    формально валиден и бесполезен, поэтому провал здесь — тоже повод к
    повтору.
    """
    problems = []
    if not 5 <= len(sc.stages) <= 7:
        problems.append(f"этапов {len(sc.stages)}, нужно от 5 до 7")
    if not 3 <= len(sc.criteria) <= 6:
        problems.append(f"критериев {len(sc.criteria)}, нужно от 3 до 6")
    if len(sc.persona.role) < 12:
        problems.append("роль персоны слишком короткая, опиши её конкретно")
    for s in sc.stages:
        if len(s.hint) < 15:
            problems.append(f"этап «{s.goal}»: подсказка пустая или слишком короткая")
        if len(s.advance_when) < 8:
            problems.append(f"этап «{s.goal}»: не сказано, когда переходить дальше")
    for c in sc.criteria:
        if not c.anchor_1 or not c.anchor_5:
            problems.append(f"критерий «{c.title}»: пустой якорь шкалы")
    goals = [s.goal.strip().lower() for s in sc.stages]
    if len(set(goals)) < len(goals):
        problems.append("этапы повторяются по цели")
    return problems


def _hopeless(e: Exception) -> bool:
    """Ошибка, которую повтор не исправит: сам запрос неверен."""
    status = getattr(e, "status_code", None)
    return status is not None and 400 <= status < 500 and status != 429


@dataclass
class Attempt:
    n: int
    ok: bool
    problems: list[str] = field(default_factory=list)
    error: str = ""
    ms: float = 0.0


@dataclass
class Result:
    scenario: Scenario
    fallback: bool
    attempts: list[Attempt]
    message: str = ""
    total_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario.to_dict(),
            "fallback": self.fallback,
            "message": self.message,
            "total_ms": round(self.total_ms),
            "attempts": [{"n": a.n, "ok": a.ok, "problems": a.problems,
                          "error": a.error, "ms": round(a.ms)} for a in self.attempts],
        }


def generate(llm, source_text: str, kind: str = "", attempts: int = ATTEMPTS,
             on_progress=None) -> Result:
    """Сгенерировать сценарий. Никогда не бросает: всегда есть шаблон.

    `llm` — вызываемое: prompt -> dict по схеме. `on_progress(state)` зовётся
    перед каждой попыткой: генерация идёт до диалога, десять-двадцать секунд
    приемлемы, тишина — нет.
    """
    t_all = time.perf_counter()
    log: list[Attempt] = []
    error = ""
    for n in range(1, attempts + 1):
        if on_progress:
            on_progress({"attempt": n, "of": attempts,
                         "note": "повтор с исправлением" if error else "первая попытка"})
        t0 = time.perf_counter()
        try:
            raw = llm(build_prompt(source_text, kind, error))
            sc = to_scenario(raw, source_text)
            problems = sc.validate() + extra_checks(sc)
            ms = (time.perf_counter() - t0) * 1000
            if not problems:
                log.append(Attempt(n, True, ms=ms))
                return Result(sc, False, log, total_ms=(time.perf_counter() - t_all) * 1000)
            log.append(Attempt(n, False, problems=problems, ms=ms))
            error = "; ".join(problems[:6])
        except Exception as e:                                  # noqa: BLE001
            ms = (time.perf_counter() - t0) * 1000
            log.append(Attempt(n, False, error=f"{type(e).__name__}: {e}", ms=ms))
            if _hopeless(e):
                # 400 повтором не чинится: неверна не выдача модели, а сам
                # запрос. Три одинаковых отказа подряд — только потерянное
                # время методиста перед пустым экраном.
                break
            error = f"ответ не разобрался: {type(e).__name__}"

    sc = to_scenario(template(kind or "generic", source_text=source_text),
                     source_text, source="template")
    why = log[-1].error or "; ".join(log[-1].problems)
    return Result(sc, True, log, total_ms=(time.perf_counter() - t_all) * 1000,
                  message=f"Модель не выдала пригодный сценарий за {len(log)} "
                          f"попытки ({why}). Ниже шаблон по типу тренировки — "
                          f"его нужно править руками.")


# ------------------------------------------------------------------- задания

@dataclass
class Job:
    """Одна генерация. Живёт в памяти процесса, как и всё остальное.

    Нужна ради индикатора прогресса: генерация занимает секунды, и всё это
    время методист должен видеть, что происходит, а не белый экран.
    """
    id: str
    state: str = "running"          # running | done | failed
    started_at: float = field(default_factory=time.time)
    progress: dict = field(default_factory=dict)
    result: Result | None = None
    error: str = ""

    @property
    def elapsed_ms(self) -> int:
        return round((time.time() - self.started_at) * 1000)

    def to_dict(self) -> dict:
        d = {"id": self.id, "state": self.state, "elapsed_ms": self.elapsed_ms,
             "progress": self.progress}
        if self.result is not None:
            d.update(self.result.to_dict())
        if self.error:
            d["error"] = self.error
        return d


class Jobs:
    """Реестр генераций. Словарь в памяти, никакой базы."""

    def __init__(self, llm_factory):
        self.llm_factory = llm_factory
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def start(self, source_text: str, kind: str = "") -> Job:
        job = Job(id=uuid.uuid4().hex[:10])
        with self._lock:
            self.jobs[job.id] = job
        threading.Thread(target=self._run, args=(job, source_text, kind),
                         daemon=True).start()
        return job

    def _run(self, job: Job, source_text: str, kind: str) -> None:
        try:
            llm = self.llm_factory()
            job.result = generate(llm, source_text, kind,
                                  on_progress=lambda p: setattr(job, "progress", p))
            job.state = "done"
        except (Exception, SystemExit) as e:                    # noqa: BLE001
            # Даже сюда — с шаблоном: нет ключа, нет сети, упал импорт. Это не
            # повод показать методисту пустой экран.
            job.result = Result(
                to_scenario(template(kind or "generic", source_text=source_text),
                            source_text, source="template"),
                True, [], message=f"Генератор недоступен ({type(e).__name__}: {e}). "
                                  f"Ниже шаблон по типу тренировки.")
            job.error = f"{type(e).__name__}: {e}"
            job.state = "done"

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)


# ---------------------------------------------------------------- чат-слой

REFINE_SYSTEM = SYSTEM + """

Сейчас сценарий уже существует, и методист просит его поправить. Верни его
ЦЕЛИКОМ в том же формате, изменив только то, о чём просят. Всё остальное
оставь буквально как было — включая формулировки, которых просьба не касается.
Не переставляй этапы и не переписывай их «заодно»."""


def build_refine_prompt(current: dict, message: str, protect: list[str]) -> str:
    lines = ["ТЕКУЩИЙ СЦЕНАРИЙ:", "",
             json.dumps(current, ensure_ascii=False, indent=2), ""]
    if current.get("source_text"):
        lines += ["ИСХОДНЫЙ ТЕКСТ, ИЗ КОТОРОГО ОН СОБРАН:", "",
                  current["source_text"], ""]
    if protect:
        lines += ["ЭТИ ПОЛЯ МЕТОДИСТ ПРАВИЛ РУКАМИ — не трогай их:",
                  ", ".join(protect), ""]
    lines += [f"ПРОСЬБА МЕТОДИСТА: {message}", "",
              "Верни исправленный сценарий целиком."]
    return "\n".join(lines)


def _get_path(d: dict, path: str):
    """Значение по пути вида `persona.role`, `stage.2.hint`, `crit.0.title`."""
    parts = path.split(".")
    try:
        if parts[0] == "stage":
            return d["stages"][int(parts[1])][parts[2]]
        if parts[0] == "crit":
            return d["criteria"][int(parts[1])][parts[2]]
        if parts[0] == "persona":
            return d["persona"][parts[1]]
        return d[parts[0]]
    except (KeyError, IndexError, ValueError, TypeError):
        return None


def _set_path(d: dict, path: str, value) -> bool:
    parts = path.split(".")
    try:
        if parts[0] == "stage":
            d["stages"][int(parts[1])][parts[2]] = value
        elif parts[0] == "crit":
            d["criteria"][int(parts[1])][parts[2]] = value
        elif parts[0] == "persona":
            d["persona"][parts[1]] = value
        else:
            d[parts[0]] = value
        return True
    except (KeyError, IndexError, ValueError, TypeError):
        return False


def refine(llm, current: dict, message: str, protect: list[str] | None = None) -> dict:
    """Частичная перегенерация артефакта по просьбе текстом.

    Ручные правки методиста возвращаются на место ПОСЛЕ ответа модели, а не
    только защищаются словами в промпте: просьба «сделай третий этап жёстче»
    не должна стирать вечер работы над первым, а полагаться в этом на
    послушность модели нельзя.

    Защита идёт по позиции. Если модель переставила или удалила этапы, позиция
    больше ничего не значит — тогда правка не возвращается, и об этом сказано
    вызывающему в `kept`, а не замолчано.
    """
    protect = protect or []
    raw = llm(build_refine_prompt(current, message, protect))
    updated = to_scenario(raw, current.get("source_text", ""),
                          source=current.get("source", "generated"),
                          scenario_id=current.get("id", "")).to_dict()

    # Защита по позиции проверяется отдельно для этапов и для критериев.
    # Общий флаг был слишком осторожен: просьба «добавь критерий» меняет длину
    # списка критериев и снимала защиту заодно с этапов, которых никто не
    # трогал, — методисту сообщалось о потере правки, которая на месте.
    same = {
        "stage": len(updated.get("stages", [])) == len(current.get("stages", [])),
        "crit": len(updated.get("criteria", [])) == len(current.get("criteria", [])),
    }
    kept, lost = [], []
    for path in protect:
        value = _get_path(current, path)
        if value is None:
            continue
        collection = path.split(".")[0]
        if collection in same and not same[collection]:
            lost.append(path)
            continue
        if _get_path(updated, path) != value and _set_path(updated, path, value):
            kept.append(path)
    return {"scenario": updated, "kept": kept, "lost": lost}
