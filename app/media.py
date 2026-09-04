"""Боевые TTS и выравнивание. Загружаются один раз и живут в процессе.

Обе модели держат заметное время холодного старта, поэтому создаются на старте
сервера, а не по запросу: 520 мс прогрева Silero, замеренные в R3, иначе
пришлись бы на первую реплику агента.
"""
from __future__ import annotations

import json
import os
import pathlib
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SR_TTS = 24000
SR_ASR = 16000


class SileroTTS:
    """Офлайновый синтез. Остаётся фолбэком независимо от выбора провайдера:
    если на площадке ляжет сеть, голос должен выжить."""

    name = "silero"

    def __init__(self, voice: str = "ru_roman", sample_rate: int = SR_TTS,
                 version: str = "v5_cis_base"):
        import torch
        torch.set_num_threads(4)
        t0 = time.perf_counter()
        self.model, _ = torch.hub.load("snakers4/silero-models", "silero_tts",
                                       language="ru", speaker=version, trust_repo=True)
        self.model.to(torch.device("cpu"))
        self.load_s = time.perf_counter() - t0
        self.version = version
        self.voice = voice
        self.sr = sample_rate
        if voice not in getattr(self.model, "speakers", [voice]):
            raise SystemExit(f"голоса «{voice}» нет в {version}; есть: "
                             f"{', '.join(v for v in self.model.speakers if v.startswith('ru'))}")
        # Замерено в R3: первые ДВА вызова стоят ~520 мс каждый, потом ~10 мс.
        # Одного прогрева мало — цена легла бы на открывающую реплику агента.
        self.warmup_ms = []
        for t in ("Прогрев.", "Ещё один прогрев, подлиннее."):
            t0 = time.perf_counter()
            self.model.apply_tts(text=t, speaker=self.voice, sample_rate=self.sr)
            self.warmup_ms.append(round((time.perf_counter() - t0) * 1000))

    def __call__(self, text: str):
        au = self.model.apply_tts(text=text, speaker=self.voice, sample_rate=self.sr)
        return np.asarray(au, dtype=np.float32), self.sr

    def describe(self) -> dict:
        return {"name": self.name, "version": self.version, "voice": self.voice,
                "sample_rate": self.sr, "load_s": round(self.load_s, 1),
                "offline": True}


class ElevenLabsTTS:
    """Сетевой синтез. Соединение держится пулом httpx между репликами.

    Потоковый websocket даёт TTFB 278 мс, но нам нужна КЛАУЗА ЦЕЛИКОМ: её всё
    равно надо выровнять локально, прежде чем отдать висемы. Поэтому берём
    обычный HTTP-стрим — он проще и не требует возни с жизненным циклом
    соединения, которое у ElevenLabs живёт ровно одну реплику.
    """

    name = "elevenlabs"
    ENDPOINT = "https://api.elevenlabs.io/v1/text-to-speech/{voice}/stream"

    def __init__(self, voice: str | None = None, model: str = "eleven_flash_v2_5",
                 sample_rate: int = SR_TTS, timeout: float = 20):
        import httpx
        from app.llm import load_env
        load_env()
        self.key = os.environ.get("ELEVENLABS_API_KEY")
        if not self.key:
            raise SystemExit("нет ELEVENLABS_API_KEY в окружении или .env")
        self.model = model
        self.sr = sample_rate
        # Постоянный клиент: соединение переиспользуется между клаузами, иначе
        # каждая платит установкой TLS — замерено 1451 мс на websocket.
        self.client = httpx.Client(timeout=timeout,
                                   limits=httpx.Limits(max_keepalive_connections=4))
        # None — голос задан в конфиге, и русский он или нет, мы не знаем.
        self.picked_russian = None
        self._voice_list: list[dict] | None = None
        self.voice = voice or self._pick_voice()
        self.downgraded_from: str | None = None
        t0 = time.perf_counter()
        self._warm()
        self.load_s = time.perf_counter() - t0

    def _warm(self) -> None:
        """Прогрев соединения — и заодно проверка, что голос вообще доступен.

        Тариф решает не меньше, чем ключ: библиотечные голоса на бесплатном
        плане отдают 402. Узнать об этом на первой реплике посреди показа
        нельзя, поэтому спотыкаемся здесь и сразу берём доступный голос.
        """
        try:
            self("Прогрев.")
            return
        except Exception as e:                              # noqa: BLE001
            if "402" not in str(e) and "paid_plan_required" not in str(e):
                raise
        fallback = self._first_premade()
        if fallback == self.voice:
            raise SystemExit("ElevenLabs: ни один голос не доступен на этом тарифе")
        print(f"  !! голос {self.voice} требует платного тарифа ElevenLabs "
              f"(«Free users cannot use library voices via the API»); беру "
              f"встроенный — он заговорит по-русски с акцентом")
        self.downgraded_from, self.voice = self.voice, fallback
        self.picked_russian = False
        self("Прогрев.")

    def _pick_voice(self) -> str:
        """Голос по умолчанию, если voice_id не задан в конфиге.

        Порядок: русский голос, иначе любой premade. Модель мультиязычная,
        поэтому по-русски заговорит любой — но с акцентом, и об этом надо
        сказать вслух, а не оставить человека гадать на показе.
        """
        for v in self._voices():
            lab = v.get("labels") or {}
            # Метка языка — код, а не название: `{"language": "ru"}`. Первая
            # версия искала подстроку «russian» и русский голос на аккаунте
            # проглядела, молча взяв английский.
            if str(lab.get("language", "")).lower() in ("ru", "russian") or \
                    "russian" in json.dumps(lab, ensure_ascii=False).lower():
                self.picked_russian = True
                return v["voice_id"]
        self.picked_russian = False
        return self._first_premade()

    def _voices(self) -> list[dict]:
        if self._voice_list is None:
            r = self.client.get("https://api.elevenlabs.io/v1/voices",
                                headers={"xi-api-key": self.key})
            r.raise_for_status()
            self._voice_list = r.json()["voices"]
        return self._voice_list

    def _first_premade(self) -> str:
        """Первый голос из встроенных.

        Именно premade, а не просто первый: библиотечные голоса на бесплатном
        тарифе API отдают 402 «Free users cannot use library voices», и
        единственный русский голос на этом аккаунте — как раз библиотечный.
        """
        vs = self._voices()
        for v in vs:
            if v.get("category") == "premade":
                return v["voice_id"]
        return vs[0]["voice_id"]

    def __call__(self, text: str):
        url = self.ENDPOINT.format(voice=self.voice)
        r = self.client.post(
            url, headers={"xi-api-key": self.key},
            params={"output_format": f"pcm_{self.sr}"},
            json={"text": text, "model_id": self.model,
                  "voice_settings": {"stability": 0.5, "similarity_boost": 0.8}})
        r.raise_for_status()
        pcm = np.frombuffer(r.content, dtype="<i2").astype(np.float32) / 32768.0
        return pcm, self.sr

    def describe(self) -> dict:
        return {"name": self.name, "model": self.model, "voice": self.voice,
                "russian_voice": self.picked_russian,
                "downgraded_from": self.downgraded_from,
                "sample_rate": self.sr, "load_s": round(self.load_s, 1),
                "offline": False}


class FallbackTTS:
    """Сетевой синтез с офлайновым запасным.

    Не украшение: демонстрация идёт с чужой сети, и если она ляжет, персонаж
    должен продолжать говорить. Первый же сбой переключает на локальный голос
    навсегда до перезапуска — метаться туда-обратно посреди реплики значит
    менять голос в середине фразы.
    """

    name = "fallback"

    def __init__(self, primary, backup):
        self.primary = primary
        self.backup = backup
        self.failed_over = False
        self.failures: list[str] = []

    @property
    def active(self):
        return self.backup if self.failed_over else self.primary

    @property
    def sr(self):
        return self.active.sr

    def __call__(self, text: str):
        if self.failed_over:
            return self.backup(text)
        try:
            return self.primary(text)
        except Exception as e:                             # noqa: BLE001
            self.failures.append(f"{type(e).__name__}: {str(e)[:120]}")
            self.failed_over = True
            print(f"  !! синтез {self.primary.name} отвалился ({self.failures[-1]}), "
                  f"переключаюсь на {self.backup.name} до перезапуска")
            return self.backup(text)

    def describe(self) -> dict:
        return {"name": self.name, "active": self.active.describe(),
                "primary": self.primary.name, "backup": self.backup.name,
                "failed_over": self.failed_over, "failures": self.failures}


# Что можно выбрать на экране. Порядок — порядок в списке.
PROVIDERS = ("silero", "elevenlabs")


class SwitchableTTS:
    """Синтезатор, который можно менять на ходу.

    Нужен ради показа: сравнить голоса, переключая конфиг и перезапуская
    сервер, значит каждый раз ждать подъёма моделей и терять диалог. Держатель
    даёт всем — конвейеру, заполнителям, оверлею — одну ссылку, а внутри
    меняется движок.

    Собранные движки кешируются: возвращаться к уже поднятому Silero заново
    через `torch.hub` было бы десятками секунд на ровном месте.
    """

    name = "switchable"

    def __init__(self, cfg: dict, provider: str | None = None):
        self.cfg = dict(cfg or {})
        self.provider = provider or self.cfg.get("provider", "silero")
        self._built: dict = {}
        self.engine = self._build(self.provider)

    def _build(self, provider: str):
        if provider not in self._built:
            self._built[provider] = build_tts(self.config_for(provider))
        return self._built[provider]

    def config_for(self, provider: str) -> dict:
        """Общие настройки плюс раздел выбранного провайдера.

        Голос общим быть не может: у Silero это имя из модели, у ElevenLabs —
        voice_id. Первая версия держала одно поле `voice` на двоих, и «ru_roman»,
        отправленный в ElevenLabs, возвращал 404.
        """
        shared = {k: v for k, v in self.cfg.items()
                  if not isinstance(v, dict) and k != "provider"}
        own = {k: v for k, v in (self.cfg.get(provider) or {}).items()
               if not k.startswith("_") and v is not None}
        cfg = {**shared, **own, "provider": provider}
        # Запасной Silero берёт свой голос из своего же раздела.
        cfg.setdefault("voice_offline",
                       (self.cfg.get("silero") or {}).get("voice", "ru_roman"))
        return cfg

    @property
    def sr(self):
        return self.engine.sr

    def __call__(self, text: str):
        return self.engine(text)

    def switch(self, provider: str):
        """Сменить движок. Возвращает True, если что-то изменилось."""
        if provider == self.provider:
            return False
        self.engine = self._build(provider)   # упадёт ДО подмены, если сломан
        self.provider = provider
        return True

    def describe(self) -> dict:
        return {"name": self.name, "provider": self.provider,
                "loaded": sorted(self._built),
                "engine": self.engine.describe()}


CONFIG = ROOT / "app" / "config.json"


def tts_config() -> dict:
    """Настройки синтеза из `app/config.json`, без служебных ключей.

    Живёт здесь, а не в сервере: скриптам замеров нужен тот же голос, что у
    сервера, но поднимать ради этого весь сервер незачем. Ключи с подчёркиванием
    — комментарии для человека, читающего конфиг.
    """
    try:
        cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"provider": "silero"}
    except json.JSONDecodeError as e:
        raise SystemExit(f"{CONFIG}: {e}")
    def clean(d):
        return {k: (clean(v) if isinstance(v, dict) else v)
                for k, v in d.items() if not k.startswith("_")}

    tts = clean(cfg.get("tts", {}))
    return tts or {"provider": "silero"}


def build_tts(cfg: dict | None = None):
    """Синтезатор по конфигу. Переключение — одна строка.

    Silero остаётся офлайновым запасным независимо от выбора: сеть на площадке
    может лечь, и голос должен выжить.
    """
    cfg = dict(cfg or {})
    provider = cfg.pop("provider", "silero")
    sr = cfg.get("sample_rate", SR_TTS)

    def silero(voice):
        return SileroTTS(voice=voice, version=cfg.get("version", "v5_cis_base"),
                         sample_rate=sr)

    if provider == "silero":
        return silero(cfg.get("voice", "ru_roman"))
    if provider == "elevenlabs":
        primary = ElevenLabsTTS(voice=cfg.get("voice"),
                                model=cfg.get("model", "eleven_flash_v2_5"),
                                sample_rate=sr)
        if not cfg.get("fallback", True):
            return primary
        # `voice` у сетевого провайдера — это voice_id, локальному он не
        # подходит: запасной голос задаётся отдельным полем.
        return FallbackTTS(primary, silero(cfg.get("voice_offline", "ru_roman")))
    raise SystemExit(f"провайдер синтеза «{provider}» не реализован")


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
