#!/usr/bin/env python3
"""Посимвольные таймкоды для готовых WAV — вход для слоя g2p модуля аватара.

`align_offline.py` меряет конвейер целиком и сохраняет уже готовые висемы своей
грубой раскладкой. Модулю аватара нужно другое: СЫРЫЕ посимвольные таймкоды,
потому что раскладку в висемы делает его собственный слой `avatar/src/g2p.js`,
где живут йотированные, мягкий знак и аканье.

Здесь только выравнивание, без синтеза: WAV уже лежат в bench/r3-tts/samples.

    bench/r1-stt/.venv/bin/python bench/r4-lipsync/chartimes.py
"""
import json, pathlib, struct, sys, time

import numpy as np
import sherpa_onnx
from huggingface_hub import hf_hub_download

ROOT = pathlib.Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "bench" / "r3-tts" / "samples"
OUT_DIR = ROOT / "avatar" / "dev" / "samples"
SR_ASR = 16000

# Тексты из bench/r3-tts/run.py, по индексу в имени файла.
PHRASES = [
    "Здравствуйте! Расскажите, пожалуйста, о вашем последнем проекте.",
    "Понятно. А что именно делали лично вы?",
    "Вы сказали, что держали latency на уровне ста миллисекунд. За счёт чего?",
    "Хорошо, давайте перейдём к следующему вопросу.",
    "Спасибо за ответ. У меня есть уточнение по поводу вашей роли в команде.",
]


def read_wav(path):
    """WAV -> (float32 моно, частота). Стдлиб, без soundfile."""
    b = path.read_bytes()
    assert b[:4] == b"RIFF" and b[8:12] == b"WAVE", f"не WAV: {path}"
    off, fmt, data = 12, None, None
    while off < len(b) - 8:
        cid = b[off:off + 4]
        sz = struct.unpack_from("<I", b, off + 4)[0]
        if cid == b"fmt ":
            fmt = struct.unpack_from("<HHIIHH", b, off + 8)
        elif cid == b"data":
            data = b[off + 8: off + 8 + sz]
        off += 8 + sz + (sz & 1)
    _, ch, sr, _, _, bits = fmt
    assert bits == 16 and ch == 1, f"ожидался 16-бит моно, а тут {bits} бит / {ch} кан"
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0, sr


def resample_linear(x, sr_from, sr_to):
    """Линейная передискретизация. Для выравнивания качества хватает."""
    if sr_from == sr_to:
        return x
    n = int(round(len(x) * sr_to / sr_from))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def main():
    repo = "csukuangfj/sherpa-onnx-nemo-ctc-giga-am-v3-russian-2025-12-16"
    t0 = time.perf_counter()
    rec = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
        model=hf_hub_download(repo, "model.int8.onnx"),
        tokens=hf_hub_download(repo, "tokens.txt"),
        num_threads=4, sample_rate=SR_ASR, feature_dim=80,
        decoding_method="greedy_search")
    print(f"распознаватель загружен за {(time.perf_counter() - t0) * 1000:.0f} мс")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index = []
    for wav in sorted(SAMPLES.glob("*.wav")):
        idx = int(wav.stem.rsplit("_", 1)[1])
        audio, sr = read_wav(wav)
        x = resample_linear(audio, sr, SR_ASR)

        t1 = time.perf_counter()
        stream = rec.create_stream()
        stream.accept_waveform(SR_ASR, x)
        rec.decode_stream(stream)
        res = stream.result
        align_ms = (time.perf_counter() - t1) * 1000

        # sherpa отдаёт токены и их таймкоды. Для CTC-модели GigaAM токены —
        # это символы, что нам и нужно: слой g2p сам разложит их в висемы.
        chars = [{"ch": t, "ms": int(round(ts * 1000))}
                 for t, ts in zip(res.tokens, res.timestamps)]
        step = min((chars[i + 1]["ms"] - chars[i]["ms"]
                    for i in range(len(chars) - 1)), default=40)

        # Звук отдаём странице как есть — она сама декодирует WAV.
        dst = OUT_DIR / f"phrase_{idx}.wav"
        dst.write_bytes(wav.read_bytes())

        rec_json = {
            "id": idx,
            "text": PHRASES[idx] if idx < len(PHRASES) else "",
            "asr_text": res.text,
            "audio": f"samples/phrase_{idx}.wav",
            "audio_s": round(len(audio) / sr, 3),
            "sample_rate": sr,
            "chars": chars,
            "n_chars": len(chars),
            "min_step_ms": step,
            "align_ms": round(align_ms),
        }
        index.append(rec_json)
        print(f"фраза {idx}: {len(chars)} символов, шаг {step} мс, "
              f"{rec_json['audio_s']} с, выравнивание {align_ms:.0f} мс")
        print(f"   распознано: {res.text[:70]}")

    (OUT_DIR / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {OUT_DIR / 'index.json'}")


if __name__ == "__main__":
    sys.exit(main())
