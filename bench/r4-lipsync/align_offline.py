#!/usr/bin/env python3
"""R4 уровень 2 — висемы из таймингов, полностью офлайн.

Silero не отдаёт пословных таймингов, а ElevenLabs требует ключа и сети. Но
синтез идёт в ~100 раз быстрее реального времени, значит можно синтезировать
фразу целиком и выровнять её тем же CTC-распознавателем, что стоит на входе:
GigaAM выдаёт посимвольные таймкоды с шагом 40 мс.

    ../r1-stt/.venv/bin/python align_offline.py
"""
import json, pathlib, time
import numpy as np
import sherpa_onnx
import torch
from huggingface_hub import hf_hub_download

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "bench" / "results" / "r4_align_offline.jsonl"
SR_TTS, SR_ASR = 24000, 16000

PHRASES = [
    "Здравствуйте! Расскажите, пожалуйста, о вашем последнем проекте.",
    "Понятно. А что именно делали лично вы?",
    "Вы сказали, что держали задержку на уровне ста миллисекунд. За счёт чего?",
    "Хорошо, давайте перейдём к следующему вопросу.",
    "Спасибо за ответ. У меня есть уточнение по поводу вашей роли в команде.",
]

# Russian graphemes -> a small viseme set. Coarse on purpose: at 200 ms
# tolerance the mouth shape matters far less than the timing.
VISEME = {
    **{c: "AA" for c in "аяъ"},
    **{c: "EE" for c in "еэ"},
    **{c: "IH" for c in "иый"},
    **{c: "OH" for c in "оё"},
    **{c: "OU" for c in "ую"},
    **{c: "MBP" for c in "мбп"},
    **{c: "FV" for c in "фв"},
    **{c: "L" for c in "л"},
    **{c: "WQ" for c in "шжщч"},
    **{c: "SS" for c in "сзц"},
    **{c: "TH" for c in "тдн"},
    **{c: "KG" for c in "кгхр"},
    " ": "SIL", "ь": "SIL",
}


def main():
    torch.set_num_threads(4)
    t0 = time.perf_counter()
    tts, _ = torch.hub.load("snakers4/silero-models", "silero_tts",
                            language="ru", speaker="v4_ru", trust_repo=True)
    tts.to(torch.device("cpu"))
    repo = "csukuangfj/sherpa-onnx-nemo-ctc-giga-am-v3-russian-2025-12-16"
    rec = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
        model=hf_hub_download(repo, "model.int8.onnx"),
        tokens=hf_hub_download(repo, "tokens.txt"),
        num_threads=4, sample_rate=SR_ASR, feature_dim=80,
        decoding_method="greedy_search")
    load_s = time.perf_counter() - t0
    for w in ("Прогрев.", "Ещё прогрев подлиннее."):
        tts.apply_tts(text=w, speaker="baya", sample_rate=SR_TTS)
    print(f"загрузка {load_s:.1f} с, прогрев готов\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, text in enumerate(PHRASES):
        t0 = time.perf_counter()
        au = np.asarray(tts.apply_tts(text=text, speaker="baya", sample_rate=SR_TTS),
                        dtype=np.float32)
        t_tts = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        idx = np.arange(0, len(au), SR_TTS / SR_ASR)
        au16 = np.interp(idx, np.arange(len(au)), au).astype(np.float32)
        t_rs = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        st = rec.create_stream()
        st.accept_waveform(SR_ASR, au16)
        rec.decode_stream(st)
        t_align = (time.perf_counter() - t0) * 1000

        toks, ts = list(st.result.tokens or []), list(st.result.timestamps or [])
        vis, last = [], None
        for tok, t in zip(toks, ts):
            v = VISEME.get(tok.lower(), "AA")
            if v != last:
                vis.append({"t_ms": round(t * 1000), "viseme": v})
                last = v
        dur = len(au) / SR_TTS
        gaps = np.diff(ts) if len(ts) > 1 else np.array([0.0])
        row = {
            "phrase_id": i, "text": text, "audio_s": round(dur, 3),
            "tts_ms": round(t_tts), "resample_ms": round(t_rs), "align_ms": round(t_align),
            "total_ms": round(t_tts + t_rs + t_align),
            "rtf": round((t_tts + t_rs + t_align) / 1000 / dur, 4),
            "n_tokens": len(toks), "n_visemes": len(vis),
            "timestamp_step_ms": round(float(np.median(gaps)) * 1000, 1),
            "asr_text": st.result.text,
            "visemes": vis,
            "run_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        rows.append(row)
        print(f"[{i}] {dur:.2f} с | синтез {row['tts_ms']:3} + выравнивание {row['align_ms']:3} "
              f"= {row['total_ms']:3} мс (RTF {row['rtf']}) | висем {len(vis)} | "
              f"шаг таймкодов {row['timestamp_step_ms']} мс")

    with OUT.open("a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tot = [r["total_ms"] for r in rows]
    print(f"\nнакладные расходы на фразу: медиана {sorted(tot)[len(tot)//2]} мс, макс {max(tot)} мс")
    print(f"худший RTF: {max(r['rtf'] for r in rows)}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
