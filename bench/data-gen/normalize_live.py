#!/usr/bin/env python3
"""Нормализует живые записи по пику и кладёт рядом второй набор.

Записи вышли на ~-30 dBFS: браузерный диктофон намеренно пишет с отключённым
autoGainControl, чтобы не портить замер эха в R6. Для STT это не помеха
(проверено), но smart-turn — нейросеть, обученная на материале нормального
уровня, поэтому оба набора прогоняются и сравниваются, а не выбираются на глаз.
"""
import json, pathlib
import numpy as np
import soundfile as sf

ROOT = pathlib.Path(__file__).resolve().parents[2]
TARGET_PEAK = 0.7          # -3 dBFS


def main():
    src = ROOT / "data" / "audio" / "manifest_live.jsonl"
    rows = [json.loads(l) for l in src.read_text().splitlines()]
    out = []
    for r in rows:
        pcm, sr = sf.read(ROOT / r["wav"], dtype="float32")
        peak = float(np.max(np.abs(pcm)))
        gain = TARGET_PEAK / max(peak, 1e-9)
        dst = ROOT / r["wav"].replace("audio/live/", "audio/live_norm/")
        dst.parent.mkdir(parents=True, exist_ok=True)
        sf.write(dst, (pcm * gain).astype("float32"), sr)
        n = dict(r)
        n["wav"] = str(dst.relative_to(ROOT))
        n["source"] = "live_norm"
        n["gain_applied_db"] = round(20 * np.log10(gain), 1)
        out.append(n)
        print(f"{r['id']:5} пик {peak:.4f} -> {TARGET_PEAK}  (+{n['gain_applied_db']} дБ)")
    mf = ROOT / "data" / "audio" / "manifest_live_norm.jsonl"
    mf.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    print(f"\n{len(out)} файлов -> {mf.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
