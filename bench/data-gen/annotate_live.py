#!/usr/bin/env python3
"""Разметка живых записей по выравниванию, а не по детектору тишины.

Живая заминка заполненная — «эээ», дыхание, тянущийся гласный. Энергии там
столько же, сколько в речи, поэтому silencedetect её не видит и даёт ложное
«окон нет». Здесь границы берутся из посимвольных таймкодов GigaAM: запретное
окно — межсловный интервал не короче порога, конец речи — конец последнего слова.

    ../r1-stt/.venv/bin/python annotate_live.py --manifest manifest_live.jsonl
"""
import argparse, json, pathlib
import numpy as np
import sherpa_onnx
import soundfile as sf
from huggingface_hub import hf_hub_download

ROOT = pathlib.Path(__file__).resolve().parents[2]
REPO = "csukuangfj/sherpa-onnx-nemo-ctc-giga-am-v3-russian-2025-12-16"


def words_of(rec, pcm):
    st = rec.create_stream()
    st.accept_waveform(16000, pcm)
    rec.decode_stream(st)
    toks, ts = list(st.result.tokens or []), list(st.result.timestamps or [])
    words, cur, start = [], "", None
    for t, tm in zip(toks, ts):
        if t == " ":
            if cur:
                words.append({"w": cur, "start": round(start, 3), "end": round(tm, 3)})
            cur, start = "", None
        else:
            if not cur:
                start = tm
            cur += t
    if cur and ts:
        words.append({"w": cur, "start": round(start, 3), "end": round(ts[-1], 3)})
    return words, st.result.text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest_live.jsonl")
    ap.add_argument("--min-gap", type=float, default=0.4,
                    help="минимальная пауза — пустая или заполненная, с")
    ap.add_argument("--normalize", action="store_true",
                    help="нормировать по пику перед распознаванием")
    a = ap.parse_args()

    rec = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
        model=hf_hub_download(REPO, "model.int8.onnx"),
        tokens=hf_hub_download(REPO, "tokens.txt"),
        num_threads=4, sample_rate=16000, feature_dim=80, decoding_method="greedy_search")

    mf = ROOT / "data" / "audio" / a.manifest
    rows = [json.loads(l) for l in mf.read_text().splitlines()]

    # Базовая длительность символа выводится из самих данных, а не задаётся
    # константой: темп речи у разных людей разный, и порог «аномально долгого
    # слова» должен быть относительно этого говорящего.
    durs = []
    for r in rows:
        pcm, sr = sf.read(ROOT / r["wav"], dtype="float32")
        if a.normalize:
            pcm = (pcm / max(1e-9, float(np.max(np.abs(pcm)))) * 0.7).astype("float32")
        ws, _ = words_of(rec, pcm)
        durs += [(w["end"] - w["start"]) / max(1, len(w["w"])) for w in ws]
    per_char = float(np.median(durs)) if durs else 0.08
    print(f"базовая длительность символа (медиана по говорящему): "
          f"{per_char*1000:.0f} мс\n")

    out, nwin, gaps_all = [], 0, []
    for r in rows:
        pcm, sr = sf.read(ROOT / r["wav"], dtype="float32")
        assert sr == 16000
        if a.normalize:
            pcm = (pcm / max(1e-9, float(np.max(np.abs(pcm)))) * 0.7).astype("float32")
        words, text = words_of(rec, pcm)
        wins = []
        # 1. Пустая пауза: интервал между словами.
        for i in range(len(words) - 1):
            g = words[i + 1]["start"] - words[i]["end"]
            gaps_all.append(g)
            if g >= a.min_gap:
                wins.append([words[i]["end"], words[i + 1]["start"]])
        # 2. Заполненная пауза: аномально долгое слово. Живая заминка — это чаще
        #    всего тянущийся гласный («как бы-ы-ы…»), и CTC относит весь этот
        #    участок к слову. Межсловного интервала не возникает вообще, поэтому
        #    правило «пауза = интервал» такие заминки не видит ни одной.
        for w in words:
            expected = len(w["w"]) * per_char
            excess = (w["end"] - w["start"]) - expected
            if excess >= a.min_gap:
                wins.append([round(w["end"] - excess, 3), w["end"]])
        wins.sort()
        r["words"] = words
        r["asr_text"] = text
        r["forbidden_windows_s"] = wins
        r["speech_onset_s"] = words[0]["start"] if words else 0.0
        r["speech_end_s"] = words[-1]["end"] if words else round(len(pcm) / sr, 3)
        r["annotation"] = f"alignment(min_gap={a.min_gap},per_char={per_char:.3f})"
        nwin += len(wins)
        out.append(r)
        if r["set"] == "r2_hesitations":
            print(f"{r['id']:5} окон={len(wins)} "
                  f"конец речи={r['speech_end_s']:.2f} с из {len(pcm)/sr:.2f} с "
                  f"| {sorted((round(w[1]-w[0],2) for w in wins), reverse=True)[:3]}")
    mf.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    hes = [r for r in out if r["set"] == "r2_hesitations"]
    empty = [r["id"] for r in hes if not r["forbidden_windows_s"]]
    print(f"\n{len(out)} клипов размечено по выравниванию, окон {nwin}")
    print(f"фраз с заминками без окон: {len(empty)} {empty or ''}")
    if gaps_all:
        print(f"межсловные интервалы: медиана {np.median(gaps_all)*1000:.0f} мс, "
              f"p95 {np.percentile(gaps_all,95)*1000:.0f} мс, макс {max(gaps_all)*1000:.0f} мс")


if __name__ == "__main__":
    main()
