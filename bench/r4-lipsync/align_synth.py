#!/usr/bin/env python3
"""Насколько хорошо GigaAM выравнивает выход Silero.

Вопрос из задания, и он не риторический: распознаватель мерили на живой речи,
а выравнивать он будет синтез — ровный темп, другая просодия, нет пауз дыхания.
Может оказаться и лучше, и хуже. Здесь это измеряется, а не предполагается.

Меряется три вещи:

    WER               расходится ли расшифровка с текстом, который мы синтезировали;
                      если расходится, таймкоды сядут не на те символы
    интервал символов медиана; поправка 80 мс выведена из живой речи, у синтеза
                      темп свой, и ограничитель быстрой речи может промахнуться
    время             выравнивание на критическом пути реплики

    bench/r1-stt/.venv/bin/python bench/r4-lipsync/align_synth.py --voice eugene
"""
import argparse, json, pathlib, re, statistics, sys, time

import numpy as np
import sherpa_onnx
from huggingface_hub import hf_hub_download

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench" / "r3-tts"))
from compare_voices import REPLIES          # noqa: E402  те же 10 реплик

RESULTS = ROOT / "bench" / "results"
SR_TTS, SR_ASR = 24000, 16000


def normalize(s: str) -> str:
    """К виду, в котором отдаёт распознаватель: нижний регистр, без пунктуации."""
    s = s.lower().replace("ё", "е")
    s = re.sub(r"[^а-я0-9\s]", " ", s)
    return " ".join(s.split())


def wer(ref: str, hyp: str) -> tuple[float, int, int]:
    r, h = normalize(ref).split(), normalize(hyp).split()
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.int32)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return (d[-1, -1] / len(r) if r else 0.0), int(d[-1, -1]), len(r)


def resample_linear(x, sr_from, sr_to):
    if sr_from == sr_to:
        return x
    n = int(round(len(x) * sr_to / sr_from))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="eugene")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(4)
    tts, _ = torch.hub.load("snakers4/silero-models", "silero_tts",
                            language="ru", speaker="v4_ru", trust_repo=True)
    tts.to(torch.device("cpu"))
    for t in ("Прогрев.", "Ещё один прогрев, подлиннее."):
        tts.apply_tts(text=t, speaker=args.voice, sample_rate=SR_TTS)

    repo = "csukuangfj/sherpa-onnx-nemo-ctc-giga-am-v3-russian-2025-12-16"
    rec = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
        model=hf_hub_download(repo, "model.int8.onnx"),
        tokens=hf_hub_download(repo, "tokens.txt"),
        num_threads=4, sample_rate=SR_ASR, feature_dim=80,
        decoding_method="greedy_search")

    rows = []
    for rep in REPLIES:
        au = np.asarray(tts.apply_tts(text=rep["text"], speaker=args.voice,
                                      sample_rate=SR_TTS), dtype=np.float32)
        x = resample_linear(au, SR_TTS, SR_ASR)

        t0 = time.perf_counter()
        st = rec.create_stream()
        st.accept_waveform(SR_ASR, x)
        rec.decode_stream(st)
        res = st.result
        align_ms = (time.perf_counter() - t0) * 1000

        chars = [{"ch": t, "ms": int(round(ts * 1000))}
                 for t, ts in zip(res.tokens, res.timestamps)]
        gaps = [chars[i + 1]["ms"] - chars[i]["ms"] for i in range(len(chars) - 1)]
        # Интервал считаем только внутри слов: пробелы несут паузы между
        # словами и смещают медиану вверх, а ограничитель быстрой речи
        # интересует именно плотность символов.
        inner = [chars[i + 1]["ms"] - chars[i]["ms"] for i in range(len(chars) - 1)
                 if chars[i]["ch"] != " " and chars[i + 1]["ch"] != " "]
        w, errs, n_ref = wer(rep["text"], res.text)
        rows.append({
            "reply_id": rep["id"], "voice": args.voice,
            "text": rep["text"], "asr_text": res.text,
            "audio_s": round(len(au) / SR_TTS, 3),
            "align_ms": round(align_ms),
            "rtf_align": round(align_ms / 1000 / (len(au) / SR_TTS), 4),
            "wer": round(w, 4), "errors": errs, "ref_words": n_ref,
            "n_chars": len(chars),
            "gap_median_ms": statistics.median(gaps) if gaps else None,
            "gap_inner_median_ms": statistics.median(inner) if inner else None,
            "gap_min_ms": min(gaps) if gaps else None,
            "chars": chars,
        })
        print(f"{rep['id']}: WER {w:.3f} ({errs}/{n_ref} слов), "
              f"символов {len(chars)}, интервал внутри слов "
              f"{statistics.median(inner) if inner else '?'} мс, "
              f"выравнивание {align_ms:.0f} мс")

    out = RESULTS / f"r4_align_synth_{args.voice}.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                   encoding="utf-8")

    wers = sorted(r["wer"] for r in rows)
    inner = [r["gap_inner_median_ms"] for r in rows if r["gap_inner_median_ms"]]
    allg = [r["gap_median_ms"] for r in rows if r["gap_median_ms"]]
    ms = sorted(r["align_ms"] for r in rows)
    print(f"\n=== голос {args.voice}, {len(rows)} реплик ===")
    print(f"WER: медиана {statistics.median(wers):.3f}, среднее {statistics.mean(wers):.3f}, "
          f"макс {wers[-1]:.3f}")
    print(f"интервал символов внутри слов: медиана {statistics.median(inner):.0f} мс")
    print(f"интервал символов со всеми пробелами: медиана {statistics.median(allg):.0f} мс")
    print(f"выравнивание: медиана {ms[len(ms)//2]} мс, макс {ms[-1]} мс, "
          f"RTF {statistics.mean(r['rtf_align'] for r in rows):.4f}")
    print(f"-> {out}")


if __name__ == "__main__":
    sys.exit(main())
