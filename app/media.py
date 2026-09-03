"""Боевые TTS и выравнивание. Загружаются один раз и живут в процессе.

Обе модели держат заметное время холодного старта, поэтому создаются на старте
сервера, а не по запросу: 520 мс прогрева Silero, замеренные в R3, иначе
пришлись бы на первую реплику агента.
"""
from __future__ import annotations

import pathlib
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SR_TTS = 24000
SR_ASR = 16000


class SileroTTS:
    def __init__(self, voice: str = "eugene", sample_rate: int = SR_TTS):
        import torch
        torch.set_num_threads(4)
        t0 = time.perf_counter()
        self.model, _ = torch.hub.load("snakers4/silero-models", "silero_tts",
                                       language="ru", speaker="v4_ru", trust_repo=True)
        self.model.to(torch.device("cpu"))
        self.load_s = time.perf_counter() - t0
        self.voice = voice
        self.sr = sample_rate
        # Замерено в R3: первые ДВА вызова стоят ~520 мс каждый, потом ~10 мс.
        # Одного прогрева мало — цена легла бы на открывающую реплику агента.
        self.warmup_ms = []
        for t in ("Прогрев.", "Ещё один прогрев, подлиннее."):
            t0 = time.perf_counter()
            self.model.apply_tts(text=t, speaker=voice, sample_rate=self.sr)
            self.warmup_ms.append(round((time.perf_counter() - t0) * 1000))

    def __call__(self, text: str):
        au = self.model.apply_tts(text=text, speaker=self.voice, sample_rate=self.sr)
        return np.asarray(au, dtype=np.float32), self.sr


class GigaAMAligner:
    """Посимвольные таймкоды. Тот же распознаватель, что стоит на входе."""

    REPO = "csukuangfj/sherpa-onnx-nemo-ctc-giga-am-v3-russian-2025-12-16"

    def __init__(self, threads: int = 4):
        import sherpa_onnx
        from huggingface_hub import hf_hub_download
        t0 = time.perf_counter()
        self.rec = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
            model=hf_hub_download(self.REPO, "model.int8.onnx"),
            tokens=hf_hub_download(self.REPO, "tokens.txt"),
            num_threads=threads, sample_rate=SR_ASR, feature_dim=80,
            decoding_method="greedy_search")
        self.load_s = time.perf_counter() - t0

    def __call__(self, pcm, sr: int):
        x = resample_linear(np.asarray(pcm, dtype=np.float32), sr, SR_ASR)
        st = self.rec.create_stream()
        st.accept_waveform(SR_ASR, x)
        self.rec.decode_stream(st)
        res = st.result
        return [{"ch": t, "ms": int(round(ts * 1000))}
                for t, ts in zip(res.tokens, res.timestamps)]


def resample_linear(x, sr_from, sr_to):
    if sr_from == sr_to:
        return x
    n = int(round(len(x) * sr_to / sr_from))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


class CancellableStream:
    """Потоковый DeepSeek, который можно оборвать снаружи.

    Обычного `generator.close()` мало: пока не пришёл первый токен, поток стоит
    в блокирующем чтении сокета, и закрыть его может только другой поток.
    Замерено: отмена до первого токена не освобождала поток ~1.9 секунды —
    ровно на величину TTFT. Слышно это не было (кадры с чужим generation_id и
    так отбрасываются), но соединение и токены оплачивались впустую, а задание
    требует гасить всю цепочку, включая запрос к модели.
    """

    def __init__(self, model, max_tokens: int = 300, temperature: float = 0.7):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._resp = None
        self._closed = False

    def cancel(self) -> None:
        """Оборвать соединение. Зовётся из другого потока."""
        self._closed = True
        r, self._resp = self._resp, None
        if r is not None:
            try:
                r.close()
            except Exception:                              # noqa: BLE001
                pass

    def __call__(self, system: str, prompt: str):
        import json
        import urllib.request

        body = json.dumps({
            "model": self.model.model, "stream": True,
            "max_tokens": self.max_tokens, "temperature": self.temperature,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
        }).encode()
        req = urllib.request.Request(
            "https://api.deepseek.com/chat/completions", data=body,
            headers={"Authorization": f"Bearer {self.model.key}",
                     "Content-Type": "application/json"})
        try:
            resp = urllib.request.urlopen(req, timeout=60)
        except Exception:                                  # noqa: BLE001
            if self._closed:
                return
            raise
        self._resp = resp
        try:
            for raw in resp:
                if self._closed:
                    break
                line = raw.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    delta = json.loads(payload)["choices"][0]["delta"].get("content")
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                if delta:
                    yield delta
        except Exception:                                  # noqa: BLE001
            # Оборванное соединение — это и есть отмена, а не сбой.
            if not self._closed:
                raise
        finally:
            self._resp = None
            try:
                resp.close()
            except Exception:                              # noqa: BLE001
                pass


def stream_deepseek(model, max_tokens: int = 300, temperature: float = 0.7):
    """Потоковый DeepSeek. Первый звук зависит от того, как быстро придёт
    первая клауза, поэтому поток обязателен."""
    return CancellableStream(model, max_tokens, temperature)
