#!/usr/bin/env python3
"""Доклеивает живым записям хвост из их собственного фонового шума.

Запись останавливали сразу после фразы, и хвоста осталось ~430 мс. Детектор,
которому нужно 700–1500 мс тишины, при этом не может сработать в принципе —
файл кончается раньше. Это артефакт длины записи, а не свойство детектора.

Клеится не цифровая тишина, а собственный фон клипа (взятый из ведущей паузы):
цифровая тишина уронила бы шумовой пол и сделала задачу детектору легче, чем
она есть на самом деле.

    ../r1-stt/.venv/bin/python pad_live.py --manifest manifest_live.jsonl
"""
import argparse, json, pathlib
import numpy as np
import soundfile as sf

ROOT = pathlib.Path(__file__).resolve().parents[2]
SR = 16000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest_live.jsonl")
    ap.add_argument("--tail-ms", type=int, default=2500)
    a = ap.parse_args()

    mf = ROOT / "data" / "audio" / a.manifest
    rows = [json.loads(l) for l in mf.read_text().splitlines()]
    out, padded = [], 0
    for r in rows:
        path = ROOT / r["wav"]
        pcm, sr = sf.read(path, dtype="float32")
        assert sr == SR
        onset = r.get("speech_onset_s", 0.0)
        have_ms = (len(pcm) / SR - r["speech_end_s"]) * 1000
        need_ms = a.tail_ms - have_ms
        if need_ms > 0:
            # фон берём из ведущей паузы; если её мало — из самого тихого куска
            lead = pcm[: int(max(0.1, onset - 0.05) * SR)]
            if len(lead) < SR // 10:
                w = int(0.1 * SR)
                frames = [(float(np.sqrt(np.mean(pcm[i:i + w] ** 2))), i)
                          for i in range(0, max(1, len(pcm) - w), w)]
                _, i0 = min(frames)
                lead = pcm[i0:i0 + w]
            reps = int(np.ceil(need_ms / 1000 * SR / len(lead)))
            tail = np.tile(lead, reps)[: int(need_ms / 1000 * SR)]
            sf.write(path, np.concatenate([pcm, tail]).astype("float32"), SR)
            padded += 1
        newlen = sf.info(path).frames / SR
        r["duration_s"] = round(newlen, 3)
        r["tail_ms"] = round((newlen - r["speech_end_s"]) * 1000)
        r["padded"] = need_ms > 0
        out.append(r)
    mf.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    tails = [r["tail_ms"] for r in out]
    print(f"доклеено {padded} из {len(out)} клипов")
    print(f"хвост теперь: медиана {int(np.median(tails))} мс, мин {min(tails)}, макс {max(tails)}")


if __name__ == "__main__":
    main()
