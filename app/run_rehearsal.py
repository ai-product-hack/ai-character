#!/usr/bin/env python3
"""Репетиция: все сценарии подряд через настоящий сервер, с перебиваниями.

    bench/r1-stt/.venv/bin/python app/run_rehearsal.py
    bench/r1-stt/.venv/bin/python app/run_rehearsal.py --scenario s1_interview_backend

Отличие от `run_pipeline.py` принципиальное. Тот дёргает конвейер напрямую и
поэтому не видит ничего, что живёт в сессии: заполнитель, спекуляцию, теги
эмоций, динамику набора, второй разбор действия. Репетиция ходит по HTTP —
ровно тем же путём, что и браузер, — и потому меряет то, что человек увидит на
показе, а не то, что удобно померить.

Не меряется здесь ровно одно: как это выглядит. Рендер требует браузера, и
цифры FPS берутся оттуда, а не отсюда.
"""
import argparse
import json
import pathlib
import random
import statistics
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.run_dialogue import FALLBACK_LINES, USER_LINES     # noqa: E402
from app.scenario import load_all                           # noqa: E402

BASE = "http://127.0.0.1:{port}"


# ------------------------------------------------------------------- сервер

def wait_for_server(port: int, proc, timeout: float = 240) -> None:
    """Ждём, пока поднимутся модели. Silero и GigaAM грузятся десятки секунд."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"сервер упал на старте, код {proc.returncode}")
        try:
            with urllib.request.urlopen(f"{BASE.format(port=port)}/api/health",
                                        timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:                                   # noqa: BLE001
            time.sleep(0.5)
    raise SystemExit("сервер не поднялся за отведённое время")


def post(port, path, obj) -> dict:
    req = urllib.request.Request(f"{BASE.format(port=port)}{path}",
                                 data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def get(port, path) -> dict:
    with urllib.request.urlopen(f"{BASE.format(port=port)}{path}", timeout=90) as r:
        return json.loads(r.read().decode())


# -------------------------------------------------------------------- поток

class Frames(threading.Thread):
    """Читает бинарный поток кадров ровно как клиент в браузере."""

    daemon = True

    def __init__(self, port):
        super().__init__()
        self.port = port
        self.frames: list[dict] = []
        self.audio_bytes = 0
        self.lock = threading.Lock()
        self.stop_flag = False

    def run(self):
        try:
            r = urllib.request.urlopen(f"{BASE.format(port=self.port)}/api/stream",
                                       timeout=600)
        except Exception:                                   # noqa: BLE001
            return
        # Читаем ровно по длине кадра, а не буфером «сколько дадут».
        # HTTPResponse.read(n) на chunked-потоке ЖДЁТ, пока наберётся n байт, и
        # кадры выходили пачками по 64 КБ — то есть замер задержки мерил
        # наполнение буфера, а не приход кадра.
        while not self.stop_flag:
            head = self._exact(r, 8)
            if head is None:
                break
            hlen, plen = struct.unpack("<II", head)
            body = self._exact(r, hlen + plen)
            if body is None:
                break
            hdr = json.loads(body[:hlen].decode())
            if hdr.get("kind"):                             # {} — сердцебиение
                with self.lock:
                    hdr["_at"] = time.perf_counter()
                    self.frames.append(hdr)
                    self.audio_bytes += plen

    @staticmethod
    def _exact(r, n: int) -> bytes | None:
        out = b""
        while len(out) < n:
            try:
                part = r.read(n - len(out))
            except Exception:                               # noqa: BLE001
                return None
            if not part:
                return None
            out += part
        return out

    def since(self, n: int) -> list[dict]:
        with self.lock:
            return self.frames[n:]

    def count(self) -> int:
        with self.lock:
            return len(self.frames)

    def wait_turn_end(self, since: int, timeout: float) -> dict | None:
        """Кадр состояния, закрывающий реплику.

        Кадров `state` за реплику приходит два: один после заполнителя (лицо
        уходит думать), второй — итоговый. Отличает их поле `action`: у
        промежуточного его нет. Первая версия брала любой и потому мерила
        заполнитель вместо реплики.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            for f in self.since(since):
                if f.get("kind") == "state" and f.get("action") is not None:
                    return f
            time.sleep(0.02)
        return None


# ---------------------------------------------------------------- репетиция

def type_like_a_human(port, text, rng, pause_ms=(60, 220)) -> float:
    """Набирает текст порциями, как человек: с паузами и одним стиранием.

    Это не украшение прогона. Спекуляция запускается по растущему тексту, а
    сигнал уверенности считается по паузам и стираниям — отправив текст одним
    куском, мы измерили бы обе подсистемы вхолостую.
    """
    t0 = time.perf_counter()
    i = 0
    erased = False
    while i < len(text):
        i = min(len(text), i + rng.randint(4, 12))
        post(port, "/api/typing", {"text": text[:i]})
        # Одно стирание в середине: так выглядит настоящая правка.
        if not erased and i > len(text) * 0.4 and rng.random() < 0.35:
            erased = True
            post(port, "/api/typing", {"text": text[:max(0, i - rng.randint(5, 15))]})
        time.sleep(rng.randint(*pause_ms) / 1000)
    return (time.perf_counter() - t0) * 1000


def run_scenario(port, sc, args, rng) -> dict:
    # Затравку запускает сам /api/start — отдельного сообщения на неё нет.
    t_start = time.perf_counter()
    post(port, "/api/start", {"scenario": sc.id, "criteria": ""})
    frames = Frames(port)
    frames.start()

    lines = USER_LINES.get(sc.id, FALLBACK_LINES)
    turns, interrupts = [], []
    print(f"\n########## {sc.id} | {sc.title}")

    # Приёмка требует «минимум по два перебивания в каждом сценарии», а
    # вероятностный бросок этого не гарантирует: на seed'е, где не выпало ни
    # одного, прогон молча проверял бы не то. Поэтому точки назначаются заранее.
    # Точки берутся из НАЧАЛА диалога, а не из всего диапазона `--turns`:
    # сценарий обычно заканчивается раньше лимита, и назначенное на 27-й ход
    # перебивание просто не наступает. Первая версия так и делала — на пять
    # сценариев пришлось два перебивания вместо десяти.
    forced_cuts = set()
    if args.min_interrupts:
        pool = list(range(1, args.min_interrupts * 3 + 1))
        rng.shuffle(pool)
        forced_cuts = set(pool[:args.min_interrupts])

    for turn in range(args.turns):
        base = frames.count()
        user = None if turn == 0 else lines[(turn - 1) % len(lines)]
        typed_ms = 0.0
        if user is not None:
            typed_ms = type_like_a_human(port, user, rng)
            print(f"\nпольз. ({typed_ms:.0f} мс набора): {user}")

        if user is None:
            t_enter = t_start
        else:
            t_enter = time.perf_counter()
            post(port, "/api/message", {"text": user})

        # Перебивание: с заданной вероятностью гасим реплику посреди речи.
        cut = None
        if turn > 0 and (turn in forced_cuts or rng.random() < args.interrupt_rate):
            delay = rng.uniform(*args.interrupt_after)
            time.sleep(delay)
            t_cut = time.perf_counter()
            seen_before = frames.count()
            gid = post(port, "/api/cancel", {}).get("cancelled")
            time.sleep(args.settle)             # даём шанс просочиться лишнему
            stray = [f for f in frames.since(seen_before)
                     if f.get("generation_id") == gid
                     and f.get("kind") in ("audio", "visemes", "subtitles")]
            cut = {"after_ms": round(delay * 1000), "generation_id": gid,
                   "leaked_frames": len(stray),
                   # Не длительность моего сна, а момент ПОСЛЕДНЕГО кадра
                   # погашенной генерации. Ноль — значит после отмены не
                   # пришло ничего.
                   "last_stray_ms": round((max(f["_at"] for f in stray) - t_cut) * 1000)
                                    if stray else 0}
            interrupts.append(cut)
            print(f"  ПЕРЕБИЛИ через {cut['after_ms']} мс -> "
                  f"просочилось кадров: {len(stray)}")

        # Погашенная реплика намеренно НЕ досылает итоговый кадр состояния:
        # сервер выходит раньше, чем доходит до него. Ждать его после отмены
        # значит ждать вечно — перебивание и есть конец этого хода.
        st = None if cut else frames.wait_turn_end(base, timeout=args.timeout)
        if st is None and not cut:
            print("  !! реплика не завершилась за отведённое время")
            break
        st = st or {}

        got = frames.since(base)
        # Клауза приезжает тремя кадрами (звук, висемы, субтитры); считаем по
        # звуковым — они и есть то, что человек слышит.
        audio = [f for f in got if f.get("kind") == "audio"]
        clauses = [f for f in audio if not f.get("backchannel")]
        filler = [f for f in audio if f.get("backchannel")]
        emos = [m for f in got if f.get("kind") == "emotions"
                for m in (f.get("marks") or [])]
        first_audio = audio[0]["_at"] if audio else None

        turns.append({
            "turn": turn, "stage": st.get("stage"),
            "typed_ms": round(typed_ms),
            "t_first_audio_ms": round((first_audio - t_enter) * 1000) if first_audio else None,
            "reported_first_audio_ms": st.get("t_first_audio_ms"),
            "reported_first_speech_ms": st.get("t_first_speech_ms"),
            "clauses": len(clauses), "filler": bool(filler),
            "emotions": [m.get("emotion") for m in emos],
            "action": st.get("action"), "interrupted": cut,
        })
        t = turns[-1]
        print(f"  клауз {t['clauses']}, заполнитель {'да' if t['filler'] else 'нет'}, "
              f"первый звук {t['t_first_audio_ms']} мс "
              f"(сервер: {t['reported_first_audio_ms']}), "
              f"речь {t['reported_first_speech_ms']} мс, "
              f"эмоции {t['emotions'] or '—'}, действие {t['action']}")
        if st.get("finished"):
            print("  сценарий завершён")
            break

    rep = get(port, "/api/report")
    frames.stop_flag = True
    return {"scenario_id": sc.id, "title": sc.title, "turns": turns,
            "interrupts": interrupts, "report": rep,
            "audio_bytes": frames.audio_bytes}


# ------------------------------------------------------------------- отчёт

def med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs)) if xs else None


def summarise(runs) -> dict:
    turns = [t for r in runs for t in r["turns"]]
    marks = [m for r in runs for m in r["report"].get("marks", [])]
    cuts = [c for r in runs for c in r["interrupts"]]
    first = [t["t_first_audio_ms"] for t in turns]
    speech = [t["reported_first_speech_ms"] for t in turns
              if t.get("reported_first_speech_ms")]
    with_typing = [t for t in turns if t["typed_ms"]]
    spec_hits = sum(1 for m in marks if m.get("spec_hit"))
    spec_tries = sum(1 for m in marks if m.get("spec_cover") is not None)
    fb = [m for m in marks if m.get("fell_back")]
    return {
        "scenarios": len(runs),
        "finished": sum(1 for r in runs if r["report"].get("completed")),
        "turns": len(turns),
        # Первый звук — то, ради чего заполнитель и делался.
        "first_audio_ms": {"median": med(first), "max": max([f for f in first if f], default=None)},
        "first_speech_ms": {"median": med(speech), "max": max(speech, default=None)},
        "filler_share": round(sum(1 for t in turns if t["filler"]) / len(turns), 2) if turns else 0,
        "speculation": {"tries": spec_tries, "hits": spec_hits,
                        "hit_rate": round(spec_hits / spec_tries, 2) if spec_tries else None},
        "emotions": len([e for t in turns for e in t["emotions"]]),
        "interrupts": {"count": len(cuts),
                       "per_scenario": [len(r["interrupts"]) for r in runs],
                       "leaked_frames": sum(c["leaked_frames"] for c in cuts),
                       "last_stray_ms_max": max((c["last_stray_ms"] for c in cuts),
                                                default=None)},
        # Доля реплик без управляющей строки и сколько из них удалось починить.
        "control_line": {"missing": len(fb), "repaired": sum(1 for m in fb if m.get("repaired")),
                         "missing_share": round(len(fb) / len(marks), 3) if marks else None,
                         "repair_ms_median": med([m.get("t_repair_ms") for m in fb])},
        "typing": {"answers": len(with_typing),
                   "confidence": {k: sum(1 for r in runs
                                         for line in r["report"].get("transcript", [])
                                         if (line.get("typing") or {}).get("confidence") == k)
                                  for k in ("уверенно", "с заминками", "неуверенно")}},
    }


def report(runs, out: pathlib.Path, tts=None, args=None) -> dict:
    s = summarise(runs)
    print("\n================ РЕПЕТИЦИЯ ================")
    if tts:
        print(f"синтез: {tts}")
    print(f"сценариев {s['scenarios']}, дошли до конца {s['finished']}, реплик {s['turns']}")
    print(f"первый звук: медиана {s['first_audio_ms']['median']} мс, "
          f"макс {s['first_audio_ms']['max']} мс")
    print(f"первая содержательная речь: медиана {s['first_speech_ms']['median']} мс, "
          f"макс {s['first_speech_ms']['max']} мс")
    print(f"заполнитель: {s['filler_share']:.0%} реплик")
    sp = s["speculation"]
    print(f"спекуляция: попаданий {sp['hits']}/{sp['tries']}"
          + (f" ({sp['hit_rate']:.0%})" if sp["hit_rate"] is not None else ""))
    print(f"эмоций проставлено: {s['emotions']}")
    it = s["interrupts"]
    print(f"перебиваний {it['count']} {it['per_scenario']}, "
          f"просочилось кадров {it['leaked_frames']}"
          + (f", последний через {it['last_stray_ms_max']} мс после отмены"
             if it["leaked_frames"] else ""))
    cl = s["control_line"]
    print(f"без управляющей строки: {cl['missing']} реплик"
          + (f" ({cl['missing_share']:.1%})" if cl["missing_share"] is not None else "")
          + f", починено {cl['repaired']}"
          + (f", разбор {cl['repair_ms_median']} мс" if cl["repair_ms_median"] else ""))
    print(f"динамика набора: ответов {s['typing']['answers']}, "
          f"{s['typing']['confidence']}")
    for r in runs:
        rep = r["report"]
        print(f"  {r['scenario_id']:<24} этап {rep.get('stages_reached')}/"
              f"{rep.get('stages_total')}, реплик {len(r['turns'])}, "
              f"завершён={rep.get('completed')}, "
              f"критериев с оценкой {rep.get('coverage')}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": s, "tts": tts, "args": args, "runs": runs},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {out}")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=None, help="id; по умолчанию все")
    # Этапов в сценариях 6–7, и модель редко закрывает этап одной репликой.
    # На шести ходах ни один сценарий до конца не доходит — репетиция тогда
    # меряет задержки, но не прохождение. Четырнадцати хватает.
    ap.add_argument("--turns", type=int, default=14)
    ap.add_argument("--port", type=int, default=8021)
    ap.add_argument("--interrupt-rate", type=float, default=0.35,
                    help="доля реплик, которые перебиваем")
    ap.add_argument("--min-interrupts", type=int, default=2,
                    help="сколько перебиваний гарантировать в каждом сценарии")
    ap.add_argument("--interrupt-after", type=float, nargs=2, default=(0.4, 2.0),
                    metavar=("МИН", "МАКС"), help="секунды до перебивания")
    ap.add_argument("--timeout", type=float, default=90)
    ap.add_argument("--settle", type=float, default=0.6,
                    help="сколько ждать после отмены, ловя просочившееся")
    ap.add_argument("--seed", type=int, default=7, help="перебивания воспроизводимы")
    ap.add_argument("--out", default="bench/results/rehearsal.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    scenarios = load_all(ROOT / "data" / "scenarios")
    if args.scenario:
        scenarios = [s for s in scenarios if s.id == args.scenario] or scenarios

    print(f"поднимаю сервер на порту {args.port}…")
    env_port = {**dict(__import__("os").environ), "PORT": str(args.port)}
    proc = subprocess.Popen([sys.executable, str(ROOT / "app" / "server.py"),
                             "--port", str(args.port)],
                            cwd=str(ROOT), env=env_port,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)
    drain = threading.Thread(target=lambda: [None for _ in proc.stdout], daemon=True)
    drain.start()
    try:
        wait_for_server(args.port, proc)
        print("  сервер готов")
        tts = get(args.port, "/api/health").get("tts")
        runs = [run_scenario(args.port, sc, args, rng) for sc in scenarios]
        s = report(runs, ROOT / args.out, tts, vars(args))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    # Ненулевой код, если репетиция вскрыла то, ради чего её и гоняют.
    if s["interrupts"]["leaked_frames"]:
        raise SystemExit("после отмены просочились кадры погашенной реплики")
    if s["finished"] < s["scenarios"]:
        raise SystemExit(f"до finish дошли {s['finished']} из {s['scenarios']}")


if __name__ == "__main__":
    main()
