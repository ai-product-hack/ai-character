#!/usr/bin/env python3
"""Проверка боевого эндпоинтера на живых записях R2.

    bench/r1-stt/.venv/bin/python bench/r2-endpoint/verify_voice.py

Отличие от `run.py`: там сравнивались 16 конфигураций детекторов, здесь
проверяется ровно тот класс, который стоит в продукте (`app.voice`), и ровно с
тем порогом, что выбран в R-фазе.

Записи подрезаны по концу речи — хвостовой тишины в них 0–180 мс. Ни один
порог в 1000 мс на них сработать не может, поэтому терминальным хвост
достраивается: живой микрофон продолжает передавать тишину после того, как
человек замолчал, и именно её детектор и ждёт.

Заминкам хвост НЕ достраивается, и это не небрежность. Ложный срез — это
срабатывание ВНУТРИ реплики, на паузе посреди мысли. Дописанная в конец тишина
означает, что человек договорил, и сработать там детектор обязан — считать это
ошибкой значит мерить собственную приписку.
"""
import argparse
import json
import pathlib
import sys
import wave

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.voice import SR_ASR, SileroEndpointer                # noqa: E402

MANIFEST = ROOT / "data" / "audio" / "manifest_live_norm.jsonl"


def load(path: pathlib.Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != SR_ASR:
            raise SystemExit(f"{path}: ожидалось {SR_ASR} Гц, а там {w.getframerate()}")
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768


def fires(ep, pcm, chunk_ms=100) -> bool:
    ep.reset()
    step = int(SR_ASR * chunk_ms / 1000)
    for i in range(0, len(pcm), step):
        if ep.push(pcm[i:i + step]):
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--silence-ms", type=int, default=1000)
    ap.add_argument("--tail-ms", type=int, default=1500,
                    help="сколько тишины дописать, имитируя живой микрофон")
    args = ap.parse_args()

    rows = [json.loads(l) for l in MANIFEST.read_text(encoding="utf-8").splitlines()]
    ep = SileroEndpointer(silence_ms=args.silence_ms)
    tail = np.zeros(int(SR_ASR * args.tail_ms / 1000), dtype=np.float32)

    false_cuts, hesitations = [], 0
    caught, terminals = 0, 0
    for r in rows:
        which = r.get("set")
        if which not in ("r2_hesitations", "r2_terminals"):
            continue
        pcm = load(ROOT / r["wav"])
        if which == "r2_hesitations":
            # Без хвоста: любое срабатывание здесь — срез посреди мысли.
            hesitations += 1
            if fires(ep, pcm):
                false_cuts.append(r["id"])
        else:
            terminals += 1
            caught += fires(ep, np.concatenate([pcm, tail]))

    print(f"порог тишины {args.silence_ms} мс, хвост терминальным {args.tail_ms} мс\n")
    print(f"заминки (резать НЕЛЬЗЯ):    {hesitations}, ложных срезов "
          f"{len(false_cuts)} ({100 * len(false_cuts) / max(1, hesitations):.0f}%)"
          + (f"  {false_cuts}" if false_cuts else ""))
    print(f"терминальные (резать НАДО): {terminals}, поймано {caught} "
          f"({100 * caught / max(1, terminals):.0f}%)")

    ok = not false_cuts and caught == terminals
    print("\n" + ("порог держится" if ok else "порог не держится — см. выше"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
