"""Голосовой ввод: накопление речи, решение о конце хода, распознавание.

Почему это дёшево именно у нас: GigaAM уже поднят и крутится — им выравнивается
синтез. Микрофон и определение конца хода ложатся поверх уже работающей модели,
второй распознаватель не нужен.

Порог тишины 1000 мс не подбирается заново: он выбран и померен в R-фазе на
живой речи с заминками (5% ложных срезов на 20 фразах). Здесь взят как данность
— см. `bench/r2-endpoint/`.

Решение о конце хода принимается ДО распознавания. Это принципиально: заполнитель
стартует по нему, а не по готовому тексту, иначе первый звук ждал бы ещё и
расшифровку.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

SR_ASR = 16000
FRAME = 512                      # Silero VAD хочет ровно 512 сэмплов на 16 кГц


@dataclass
class Utterance:
    """Дослушанная реплика пользователя."""
    pcm: np.ndarray
    sample_rate: int
    speech_ms: float             # сколько из неё было речью
    t_endpoint: float            # perf_counter в момент решения о конце хода

    @property
    def duration_ms(self) -> float:
        return len(self.pcm) / self.sample_rate * 1000


class SileroEndpointer:
    """Конец хода по тишине после речи. Обёртка над Silero VAD.

    Правило то же, что победило в R2: считаем речь по VAD, а не по энергии, и
    ждём `silence_ms` тишины подряд. Энергетический порог там проигрывал, когда
    запись начиналась не с тишины.
    """

    def __init__(self, silence_ms: int = 1000, threshold: float = 0.5,
                 min_speech_ms: int = 300):
        self.silence_ms = silence_ms
        self.threshold = threshold
        # Короче этого — не реплика, а кашель или щелчок мыши. Без порога
        # каждый посторонний звук начинал бы ход.
        self.min_speech_ms = min_speech_ms
        from silero_vad import load_silero_vad
        import torch
        self._torch = torch
        self.model = load_silero_vad(onnx=True)
        self.reset()

    def reset(self) -> None:
        try:
            self.model.reset_states()
        except Exception:                                     # noqa: BLE001
            pass
        self._tail = np.zeros(0, dtype=np.float32)
        self._silence_ms = 0.0
        self._speech_ms = 0.0
        self._spoke = False

    @property
    def speech_ms(self) -> float:
        return self._speech_ms

    def push(self, pcm: np.ndarray) -> bool:
        """Скормить кусок звука. True — ход закончился.

        Хвост короче кадра остаётся в буфере: VAD принимает только ровные 512
        сэмплов, а куски от браузера приходят произвольной длины.
        """
        self._tail = np.concatenate([self._tail, np.asarray(pcm, dtype=np.float32)])
        fired = False
        i = 0
        ms_per_frame = FRAME / SR_ASR * 1000
        while len(self._tail) - i >= FRAME:
            frame = self._tail[i:i + FRAME]
            i += FRAME
            p = float(self.model(self._torch.from_numpy(frame), SR_ASR).item())
            if p >= self.threshold:
                self._spoke = True
                self._speech_ms += ms_per_frame
                self._silence_ms = 0.0
            elif self._spoke:
                self._silence_ms += ms_per_frame
                if self._silence_ms >= self.silence_ms \
                        and self._speech_ms >= self.min_speech_ms:
                    fired = True
                    break
        self._tail = self._tail[i:]
        return fired


class VoiceInput:
    """Микрофонный ход целиком: копит звук, решает, когда он кончился.

    Живёт на сервере, а не в браузере: и VAD, и распознаватель уже здесь, а
    гонять их в браузере значило бы тащить туда onnxruntime ради того, что уже
    работает.
    """

    def __init__(self, endpointer=None, max_ms: int = 60000):
        self.endpointer = endpointer
        # Потолок на реплику. Без него зависший микрофон копил бы память до
        # конца сессии, а распознавание минутного куска стоит секунд.
        self.max_ms = max_ms
        self.enabled = False
        # Пока агент говорит, микрофон не слушаем. Наушники и эхоподавление
        # браузера — первая линия, но полагаться только на них нельзя: на
        # колонках VAD сработает на голос агента и перебьёт его сам собой.
        self.muted_by_agent = False
        self._chunks: list[np.ndarray] = []
        self._samples = 0
        self._lock = threading.Lock()
        self.stats = {"utterances": 0, "dropped_short": 0, "muted_chunks": 0}

    def start(self) -> None:
        with self._lock:
            self._chunks.clear()
            self._samples = 0
            if self.endpointer is not None:
                self.endpointer.reset()

    def snapshot(self) -> np.ndarray | None:
        """Копия накопленного звука, не трогая сам буфер.

        Нужна для распознавания на лету: пока человек говорит, мы прогоняем
        уже сказанное и показываем ему слова. Копия обязательна — буфер живёт
        в другом потоке и продолжает расти.
        """
        with self._lock:
            if not self._chunks:
                return None
            return np.concatenate(self._chunks)

    def push(self, pcm: np.ndarray) -> Utterance | None:
        """Кусок с микрофона. Вернёт реплику, когда ход закончится."""
        if not self.enabled:
            return None
        if self.muted_by_agent:
            self.stats["muted_chunks"] += 1
            return None
        with self._lock:
            pcm = np.asarray(pcm, dtype=np.float32)
            self._chunks.append(pcm)
            self._samples += len(pcm)
            over = self._samples / SR_ASR * 1000 >= self.max_ms
            fired = self.endpointer.push(pcm) if self.endpointer is not None else False
            if not (fired or over):
                return None
            return self._finish()

    def flush(self) -> Utterance | None:
        """Закончить ход принудительно: отпущена кнопка push-to-talk."""
        with self._lock:
            if not self._chunks:
                return None
            return self._finish(forced=True)

    def _finish(self, forced: bool = False) -> Utterance | None:
        audio = np.concatenate(self._chunks) if self._chunks else np.zeros(0, np.float32)
        speech = self.endpointer.speech_ms if self.endpointer is not None else 0.0
        self._chunks.clear()
        self._samples = 0
        if self.endpointer is not None:
            self.endpointer.reset()
        # Принудительный конец по кнопке доверяем человеку: он знает, что
        # сказал. Автоматический — проверяем, что речь вообще была.
        if not forced and speech <= 0:
            self.stats["dropped_short"] += 1
            return None
        self.stats["utterances"] += 1
        return Utterance(pcm=audio, sample_rate=SR_ASR, speech_ms=speech,
                         t_endpoint=time.perf_counter())


def transcribe(aligner, utt: Utterance) -> str:
    """Текст реплики из тех же таймкодов, что дают висемы.

    Латиница из GigaAM не возвращается никогда: распознаватель русский, и
    «system design» приходит как «систем дизайн». Это не дефект, который надо
    чинить обратным переводом — сопоставление со сценарием у нас смысловое,
    его делает модель, а ей транслитерация не мешает.
    """
    chars = aligner(utt.pcm, utt.sample_rate)
    return "".join(c["ch"] for c in chars).strip()
