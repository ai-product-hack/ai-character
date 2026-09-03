#!/usr/bin/env python3
"""R3 — TTS: time to first audio, stream stability, word timings.

TTFB is measured to the first *playable* chunk, not to the end of synthesis,
because that is what the user hears. For engines that only return a whole
utterance, the honest TTFB is the synthesis time of the first phrase — so the
bench also reports how short a first phrase has to be to stay inside budget.
"""
import argparse, json, pathlib, re, time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "bench" / "results"
PHRASES = [
    "Здравствуйте! Расскажите, пожалуйста, о вашем последнем проекте.",
    "Понятно. А что именно делали лично вы?",
    "Вы сказали, что держали latency на уровне ста миллисекунд. За счёт чего?",
    "Хорошо, давайте перейдём к следующему вопросу.",
    "Спасибо за ответ. У меня есть уточнение по поводу вашей роли в команде.",
]


def split_first_phrase(text):
    """Where a streaming pipeline would cut the first chunk: first clause."""
    m = re.search(r"[.!?,:;]", text)
    return (text[:m.end()], text[m.end():].strip()) if m else (text, "")


class SileroTTS:
    # Named after what actually loads: the hub entry `speaker="v4_ru"` resolves
    # to class TTSModelMultiAcc_v3 with voices aidar/baya/kseniya/xenia/eugene.
    # The public models.yml lists only up to v3_1_ru for Russian, and the
    # v5_cis_base referenced in Silero's docs is not reachable through this
    # entry point — so this is v4_ru, not v5.
    name = "silero_tts v4_ru (TTSModelMultiAcc_v3)"
    license = "MIT (snakers4/silero-models)"

    def __init__(self, speaker="eugene", sr=48000):
        import torch
        self.torch = torch
        torch.set_num_threads(4)
        t0 = time.perf_counter()
        self.model, _ = torch.hub.load("snakers4/silero-models", "silero_tts",
                                       language="ru", speaker="v4_ru", trust_repo=True)
        self.load_s = time.perf_counter() - t0
        self.model.to(torch.device("cpu"))
        self.speaker, self.sr = speaker, sr
        self.voices = [v for v in getattr(self.model, "speakers", []) if v != "random"]
        # Measured: the first TWO calls cost ~520 ms each, then it drops to
        # ~10 ms. One warm-up is not enough — the cost would land on the
        # agent's opening line.
        self.warmup_ms = []
        for t in ("Прогрев.", "Ещё один прогрев, подлиннее."):
            t0 = time.perf_counter()
            self.model.apply_tts(text=t, speaker=speaker, sample_rate=sr)
            self.warmup_ms.append(round((time.perf_counter() - t0) * 1000))

    def synth(self, text):
        t0 = time.perf_counter()
        au = self.model.apply_tts(text=text, speaker=self.speaker, sample_rate=self.sr)
        return np.asarray(au, dtype=np.float32), (time.perf_counter() - t0) * 1000

    @property
    def word_timings(self):
        return False


def bench(engine, out_path):
    rows = []
    for i, text in enumerate(PHRASES):
        first, rest = split_first_phrase(text)
        au_first, ms_first = engine.synth(first)
        au_full, ms_full = engine.synth(text)
        dur_first = len(au_first) / engine.sr
        dur_full = len(au_full) / engine.sr
        row = {
            "engine": engine.name, "license": engine.license, "phrase_id": i,
            "text": text, "first_chunk_text": first,
            "ttfb_first_chunk_ms": round(ms_first),
            "first_chunk_audio_s": round(dur_first, 3),
            "full_synth_ms": round(ms_full), "full_audio_s": round(dur_full, 3),
            "rtf_full": round(ms_full / 1000 / dur_full, 3),
            "word_timings": engine.word_timings,
            "sample_rate": engine.sr, "voice": engine.speaker,
            "voices_available": engine.voices,
            "run_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        rows.append(row)
        print(f"  [{i}] TTFB={row['ttfb_first_chunk_ms']:5} мс на «{first[:38]}» "
              f"| полностью {row['full_synth_ms']:5} мс / {dur_full:.2f} с "
              f"| RTF {row['rtf_full']}", flush=True)
        wav = OUT.parent / "r3-tts" / "samples" / f"{engine.name}_{i}.wav"
        wav.parent.mkdir(parents=True, exist_ok=True)
        import soundfile as sf
        sf.write(wav, au_full, engine.sr)
    with out_path.open("a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="silero")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    eng = {"silero": SileroTTS}[a.engine]()
    print(f"### {eng.name} — загружен за {eng.load_s:.1f} с, "
          f"прогрев {eng.warmup_ms} мс (два вызова)")
    print(f"    голоса: {', '.join(eng.voices)} | пословные тайминги: "
          f"{'да' if eng.word_timings else 'НЕТ'}")
    rows = bench(eng, OUT / "r3_tts.jsonl")
    t = [r["ttfb_first_chunk_ms"] for r in rows]
    print(f"\nTTFB первой фразы: медиана {sorted(t)[len(t)//2]} мс, макс {max(t)} мс")
    print(f"-> {OUT / 'r3_tts.jsonl'}")
