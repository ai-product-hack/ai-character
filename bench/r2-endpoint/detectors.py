"""End-of-turn detectors, all fed the same real-time audio stream.

Every detector answers one question per frame: "has the person finished?"
A YES inside a mid-phrase pause is a false cut — the product failure we are
hunting. A late YES after a genuinely finished turn is dead air.
"""
from __future__ import annotations
import numpy as np

SR = 16000


class Detector:
    name = "base"
    kind = "audio"          # audio | text | hybrid
    needs_partials = False

    def reset(self):
        self.fires: list[float] = []

    def push(self, pcm: np.ndarray, t: float) -> None: ...
    def push_partial(self, text: str, t: float) -> None: ...


class EnergySilence(Detector):
    """The naive design: N ms below an adaptive floor and we call it a turn.
    Included because it is what every first-day hackathon build actually ships.

    The floor is a causal low percentile over a rolling window, not the first
    300 ms. Seeding from the opening frames only works if the stream happens to
    start with silence; when it starts mid-speech the threshold lands above the
    speech itself and the detector silently never fires.
    """
    kind = "audio"

    def __init__(self, n_ms=700, over_floor_db=9.0, win_s=2.0, pctl=15):
        self.n_ms, self.over_floor_db = n_ms, over_floor_db
        self.win_s, self.pctl = win_s, pctl
        self.name = f"energy-silence-{n_ms}ms"
        self.reset()

    def reset(self):
        super().reset()
        self._hist: list[float] = []
        self._maxhist = 1
        self._sil_since, self._spoke, self._armed = None, False, True

    def _floor(self, rms, dur):
        self._maxhist = max(4, int(self.win_s / max(dur, 1e-6)))
        self._hist.append(rms)
        if len(self._hist) > self._maxhist:
            self._hist.pop(0)
        return float(np.percentile(self._hist, self.pctl))

    def push(self, pcm, t):
        rms = float(np.sqrt(np.mean(pcm ** 2)) + 1e-9)
        dur = len(pcm) / SR
        floor = self._floor(rms, dur)
        thr = floor * (10 ** (self.over_floor_db / 20))
        if rms > thr:
            self._spoke, self._sil_since, self._armed = True, None, True
        elif self._spoke:
            if self._sil_since is None:
                self._sil_since = t - dur
            elif self._armed and (t - self._sil_since) * 1000 >= self.n_ms:
                self.fires.append(t)
                # One decision per pause. A real pipeline commits on the first
                # "done" and starts the LLM; it does not keep re-asking inside
                # the same silence. Counting every re-ask inflates false cuts.
                self._armed = False


class SileroSilence(Detector):
    """Same trigger rule, but silence is decided by Silero VAD instead of energy.
    Separates 'is the gate wrong' from 'is the rule wrong'."""
    kind = "audio"

    def __init__(self, n_ms=700, thr=0.5):
        self.n_ms, self.thr = n_ms, thr
        self.name = f"silero-vad-{n_ms}ms"
        from silero_vad import load_silero_vad
        import torch
        self.torch = torch
        self.model = load_silero_vad(onnx=True)
        self.reset()

    def reset(self):
        super().reset()
        try:
            self.model.reset_states()
        except Exception:
            pass
        self._buf = np.zeros(0, dtype=np.float32)
        self._sil_since, self._spoke, self._t = None, False, 0.0
        self._armed = True

    def push(self, pcm, t):
        # Silero wants exactly 512-sample frames at 16 kHz (32 ms).
        self._buf = np.concatenate([self._buf, pcm])
        start_t = t - len(pcm) / SR
        i = 0
        while len(self._buf) - i >= 512:
            fr = self._buf[i:i + 512]
            i += 512
            ft = start_t + (i / SR)
            p = float(self.model(self.torch.from_numpy(fr), SR).item())
            if p >= self.thr:
                self._spoke, self._sil_since, self._armed = True, None, True
            elif self._spoke:
                if self._sil_since is None:
                    self._sil_since = ft
                elif self._armed and (ft - self._sil_since) * 1000 >= self.n_ms:
                    self.fires.append(ft)
                    self._armed = False
        self._buf = self._buf[i:]


class SmartTurnV3(Detector):
    """pipecat-ai/smart-turn-v3.2 (BSD-2-Clause): Whisper-tiny encoder + linear
    head over the raw waveform. Asked only at candidate moments — after a short
    VAD pause — exactly as Pipecat uses it, since running it every frame would
    be both wasteful and unlike production."""
    kind = "audio"

    def __init__(self, candidate_ms=250, thr=0.5, variant="smart-turn-v3.2-cpu.onnx"):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        self.thr, self.candidate_ms = thr, candidate_ms
        self.name = f"smart-turn-v3.2(cand={candidate_ms}ms,thr={thr})"
        path = hf_hub_download("pipecat-ai/smart-turn-v3", variant)
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        import mlx_whisper.audio as A
        self.mel = A.log_mel_spectrogram
        self.gate = EnergySilence(n_ms=candidate_ms)
        self.infer_ms: list[float] = []
        self.probs: list[tuple[float, float]] = []
        self.reset()

    def reset(self):
        super().reset()
        self.gate.reset()
        self.gate.fires = []
        self._audio = np.zeros(0, dtype=np.float32)
        self.infer_ms, self.probs = [], []

    def _predict(self) -> float:
        import time
        w = self._audio[-8 * SR:]
        if len(w) < 8 * SR:                       # pad at the front, per pipecat
            w = np.concatenate([np.zeros(8 * SR - len(w), np.float32), w])
        m = np.array(self.mel(w, n_mels=80))
        if m.shape[0] != 80:
            m = m.T
        m = m[:, :800]
        if m.shape[1] < 800:
            m = np.pad(m, ((0, 0), (0, 800 - m.shape[1])))
        t0 = time.perf_counter()
        out = self.sess.run(None, {"input_features": m[None].astype(np.float32)})[0]
        self.infer_ms.append((time.perf_counter() - t0) * 1000)
        v = float(np.ravel(out)[0])
        # The output is named "logits" but pipecat treats it as a probability.
        # Trust the range, not the name.
        return v if 0.0 <= v <= 1.0 else 1 / (1 + np.exp(-v))

    def push(self, pcm, t):
        self._audio = np.concatenate([self._audio, pcm])
        before = len(self.gate.fires)
        self.gate.push(pcm, t)
        if len(self.gate.fires) > before:          # a pause worth asking about
            p = self._predict()
            self.probs.append((t, p))
            if p > self.thr:
                self.fires.append(t)


CONTINUATION = set("""в на с о у к из по для от до за при над под про без между
и а но или что чтобы как когда если то ну вот это как-то типа значит просто
мой моя мое мои то-есть я мы он она они не ни же бы ли""".split())
TERMINAL_PUNCT = ".!?"


class PartialRule(Detector):
    """Our own candidate: pause length + punctuation of the STT partial +
    whether the last word can end a Russian sentence. Costs nothing, runs on
    text we already have, and is the thing to beat before paying for a model."""
    kind = "text"
    needs_partials = True

    def __init__(self, short_ms=500, long_ms=1600):
        self.short_ms, self.long_ms = short_ms, long_ms
        self.name = f"partial-rule({short_ms}/{long_ms}ms)"
        self.short_gate = EnergySilence(n_ms=short_ms)
        self.long_gate = EnergySilence(n_ms=long_ms)
        self.reset()

    def reset(self):
        super().reset()
        for g in (self.short_gate, self.long_gate):
            g.reset(); g.fires = []
        self._text = ""

    def push_partial(self, text, t):
        if text:
            self._text = text

    def _dangling(self) -> bool:
        txt = self._text.strip()
        if not txt:
            return True                     # nothing recognised yet: stay silent
        last = txt.rstrip(".,!?\u2026").split()
        return (last[-1].lower() in CONTINUATION) if last else True

    def push(self, pcm, t):
        s_before, l_before = len(self.short_gate.fires), len(self.long_gate.fires)
        self.short_gate.push(pcm, t)
        self.long_gate.push(pcm, t)
        short_fired = len(self.short_gate.fires) > s_before
        long_fired = len(self.long_gate.fires) > l_before
        if not (short_fired or long_fired):
            return
        # A dangling preposition or conjunction means the person is mid-thought,
        # however long the pause. Otherwise: end on punctuation after a short
        # pause, or on a pause too long to still be a hesitation.
        if self._dangling():
            return
        txt = self._text.strip()
        if (short_fired and txt[-1] in TERMINAL_PUNCT) or long_fired:
            self.fires.append(t)


class Hybrid(Detector):
    """smart-turn as an early exit, silence timeout as the backstop.

    Neither half is good alone: smart-turn is fast but cuts people off, the
    timeout is safe but costs 1.5 s. Combined, a confidently finished turn ends
    in ~200 ms and anything the model is unsure about falls back to waiting.
    Optionally vetoed by the STT partial: a dangling preposition is never an
    end of turn no matter how confident the audio model is.
    """
    kind = "hybrid"
    needs_partials = True

    def __init__(self, st_thr=0.97, timeout_ms=1500, candidate_ms=250, use_text_veto=True):
        self.name = (f"hybrid(smart-turn>{st_thr} | таймаут {timeout_ms}мс"
                     f"{' | вето по тексту' if use_text_veto else ''})")
        self.st = SmartTurnV3(candidate_ms=candidate_ms, thr=st_thr)
        self.backstop = EnergySilence(n_ms=timeout_ms)
        self.use_text_veto = use_text_veto
        self._text = ""
        self.reset()

    def reset(self):
        super().reset()
        self.st.reset(); self.st.fires = []
        self.backstop.reset(); self.backstop.fires = []
        self._text = ""

    def push_partial(self, text, t):
        if text:
            self._text = text

    def _dangling(self) -> bool:
        if not self.use_text_veto:
            return False
        txt = self._text.strip()
        if not txt:
            return False
        w = txt.rstrip(".,!?…").split()
        return bool(w) and w[-1].lower() in CONTINUATION

    def push(self, pcm, t):
        a, b = len(self.st.fires), len(self.backstop.fires)
        self.st.push(pcm, t)
        self.backstop.push(pcm, t)
        fired_fast = len(self.st.fires) > a
        fired_slow = len(self.backstop.fires) > b
        if fired_fast and self._dangling():
            fired_fast = False          # audio says done, grammar says not yet
        if fired_fast or fired_slow:
            if not self.fires or t - self.fires[-1] > 0.05:
                self.fires.append(t)

    @property
    def probs(self):
        return self.st.probs

    @property
    def infer_ms(self):
        return self.st.infer_ms
