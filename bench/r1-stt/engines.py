"""Streaming STT adapters, all behind one interface.

Each engine is labelled with how it actually streams, because the distinction
decides whether partials are stable:

  native      — the model is trained to consume audio incrementally.
  incremental — batch-native model adapted with a carried encoder cache
                (approximate: only `depth` layers match a full forward pass).
  window      — no state at all; the whole buffer is re-decoded every hop.
                Partials churn: earlier words can change on the next hop.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field


@dataclass
class Event:
    t: float          # seconds since stream start (wall clock)
    kind: str         # "partial" | "final"
    text: str
    compute_ms: float = 0.0


@dataclass
class Engine:
    name: str
    mode: str
    events: list = field(default_factory=list)

    def feed(self, pcm, t: float) -> None: ...
    def finish(self, t: float) -> str: ...
    @property
    def final_text(self) -> str:
        f = [e for e in self.events if e.kind == "final"]
        return f[-1].text if f else ""


class ParakeetStream(Engine):
    """parakeet-mlx StreamingParakeet: rotating KV cache + local attention."""

    def __init__(self, repo="mlx-community/parakeet-tdt-0.6b-v3", context=(256, 256), depth=1):
        super().__init__(name=f"parakeet-tdt-0.6b-v3-mlx(ctx={context[0]},d={depth})",
                         mode="incremental")
        from parakeet_mlx import from_pretrained
        self.model = from_pretrained(repo)
        self._ctx, self._depth = context, depth
        self._stream = None

    def start(self):
        self._cm = self.model.transcribe_stream(context_size=self._ctx, depth=self._depth)
        self._stream = self._cm.__enter__()

    def feed(self, pcm, t):
        import mlx.core as mx
        t0 = time.perf_counter()
        self._stream.add_audio(mx.array(pcm))
        txt = self._stream.result.text.strip()
        dt = (time.perf_counter() - t0) * 1000
        if txt:
            self.events.append(Event(t, "partial", txt, dt))

    def finish(self, t):
        txt = self._stream.result.text.strip()
        self.events.append(Event(t, "final", txt, 0.0))
        self._cm.__exit__(None, None, None)
        return txt


class WhisperWindow(Engine):
    """mlx-whisper re-decoding a growing buffer. The honest baseline for what
    'streaming whisper' actually is: a wrapper, with all the churn that implies."""

    def __init__(self, repo="mlx-community/whisper-large-v3-turbo", hop_s=1.0, lang="ru"):
        super().__init__(name=f"whisper-large-v3-turbo-mlx(hop={hop_s}s)", mode="window")
        self.repo, self.hop_s, self.lang = repo, hop_s, lang
        import mlx_whisper  # noqa: F401  (import cost paid up front, not in the loop)
        self.mlx_whisper = mlx_whisper
        self._buf = None
        self._last_hop = -1e9

    def start(self):
        import numpy as np
        self._buf = np.zeros(0, dtype=np.float32)
        self._last_hop = -1e9

    def _decode(self, t, kind):
        t0 = time.perf_counter()
        r = self.mlx_whisper.transcribe(self._buf, path_or_hf_repo=self.repo,
                                        language=self.lang, fp16=True,
                                        condition_on_previous_text=False)
        dt = (time.perf_counter() - t0) * 1000
        txt = r["text"].strip()
        self.events.append(Event(t, kind, txt, dt))
        return txt

    def feed(self, pcm, t):
        import numpy as np
        self._buf = np.concatenate([self._buf, pcm])
        if t - self._last_hop >= self.hop_s:
            self._last_hop = t
            self._decode(t, "partial")

    def finish(self, t):
        return self._decode(t, "final")


class FasterWhisperWindow(Engine):
    """faster-whisper (CTranslate2) on CPU — the offline fallback candidate."""

    def __init__(self, size="large-v3-turbo", hop_s=1.0, lang="ru", compute="int8"):
        super().__init__(name=f"faster-whisper-{size}-{compute}(hop={hop_s}s)", mode="window")
        from faster_whisper import WhisperModel
        self.model = WhisperModel(size, device="cpu", compute_type=compute)
        self.hop_s, self.lang = hop_s, lang
        self._buf = None
        self._last_hop = -1e9

    def start(self):
        import numpy as np
        self._buf = np.zeros(0, dtype=np.float32)
        self._last_hop = -1e9

    def _decode(self, t, kind):
        t0 = time.perf_counter()
        segs, _ = self.model.transcribe(self._buf, language=self.lang, beam_size=1,
                                        condition_on_previous_text=False, vad_filter=False)
        txt = " ".join(s.text for s in segs).strip()
        dt = (time.perf_counter() - t0) * 1000
        self.events.append(Event(t, kind, txt, dt))
        return txt

    def feed(self, pcm, t):
        import numpy as np
        self._buf = np.concatenate([self._buf, pcm])
        if t - self._last_hop >= self.hop_s:
            self._last_hop = t
            self._decode(t, "partial")

    def finish(self, t):
        return self._decode(t, "final")


class GigaAMWindow(Engine):
    """GigaAM-v3 CTC (MIT, Sber) via sherpa-onnx, int8 on CPU. Russian-specialised
    and the only candidate here trained primarily on Russian.

    Windowed like the Whisper engines, but CTC decoding is monotonic: a word once
    emitted does not get rewritten on the next hop the way an autoregressive
    decoder's output does. Cheap enough to re-decode often.
    """

    def __init__(self, hop_s=0.5, threads=4):
        super().__init__(name=f"gigaam-v3-ctc-int8-onnx(hop={hop_s}s)", mode="window")
        import sherpa_onnx
        from huggingface_hub import hf_hub_download
        repo = "csukuangfj/sherpa-onnx-nemo-ctc-giga-am-v3-russian-2025-12-16"
        self.rec = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
            model=hf_hub_download(repo, "model.int8.onnx"),
            tokens=hf_hub_download(repo, "tokens.txt"),
            num_threads=threads, sample_rate=16000, feature_dim=80,
            decoding_method="greedy_search")
        self.hop_s = hop_s
        self._buf = None
        self._last_hop = -1e9

    def start(self):
        import numpy as np
        self._buf = np.zeros(0, dtype=np.float32)
        self._last_hop = -1e9

    def _decode(self, t, kind):
        t0 = time.perf_counter()
        st = self.rec.create_stream()
        st.accept_waveform(16000, self._buf)
        self.rec.decode_stream(st)
        dt = (time.perf_counter() - t0) * 1000
        txt = st.result.text.strip()
        self.events.append(Event(t, kind, txt, dt))
        return txt

    def feed(self, pcm, t):
        import numpy as np
        self._buf = np.concatenate([self._buf, pcm])
        if t - self._last_hop >= self.hop_s:
            self._last_hop = t
            self._decode(t, "partial")

    def finish(self, t):
        return self._decode(t, "final")


REGISTRY = {
    "parakeet":        lambda: ParakeetStream(),
    "parakeet-d2":     lambda: ParakeetStream(depth=2),
    "whisper-mlx":     lambda: WhisperWindow(hop_s=1.0),
    "whisper-mlx-h05": lambda: WhisperWindow(hop_s=0.5),
    "faster-whisper":  lambda: FasterWhisperWindow(hop_s=1.0),
    "gigaam":          lambda: GigaAMWindow(hop_s=0.5),
    "gigaam-h025":     lambda: GigaAMWindow(hop_s=0.25),
}
