#!/usr/bin/env python3
"""S1 — сквозной стенд задержки: mic/файл → STT → эндпоинтер → LLM → TTS →
выравнивание → «динамик», с отметкой времени на каждой границе.

Провайдеры подменяются через config.json. LLM по умолчанию mock с настраиваемым
TTFT: спекулятивная генерация — свойство механики, а не провайдера, и её выигрыш
меряется без ключей. Подставив реальный провайдер, тот же прогон даст числа с ним.

    ../../bench/r1-stt/.venv/bin/python pipeline.py --clip h02
    ../../bench/r1-stt/.venv/bin/python pipeline.py --all --no-speculative
"""
from __future__ import annotations
import argparse, asyncio, json, os, pathlib, sys, time

import numpy as np
import soundfile as sf

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "bench" / "r2-endpoint"))
sys.path.insert(0, str(ROOT / "bench" / "r1-stt"))

SR = 16000


class Trace:
    """Одна временная шкала на прогон. Всё меряется от конца речи, потому что
    именно от него отсчитывает пользователь и жюри."""

    def __init__(self):
        self.t0 = time.perf_counter()
        self.marks: list[tuple[str, float, dict]] = []

    def mark(self, name, **kw):
        self.marks.append((name, (time.perf_counter() - self.t0) * 1000, kw))

    def at(self, name):
        for n, t, _ in self.marks:
            if n == name:
                return t
        return None

    def rel(self, name, base):
        a, b = self.at(name), self.at(base)
        return None if a is None or b is None else round(a - b)


class LLMStream:
    """Общий интерфейс: запрос уходит, токены приходят в очередь, запрос можно
    отменить. Спекуляция строится поверх этого одинаково для mock и для реального
    провайдера, поэтому её выигрыш сравним между ними."""

    def __init__(self):
        self.q: asyncio.Queue = asyncio.Queue()
        self.task: asyncio.Task | None = None
        self.prompt = ""
        self.t_start = None
        self.t_first_token = None

    def start(self, prompt: str):
        self.prompt = prompt
        self.t_start = time.perf_counter()
        self.t_first_token = None
        self.q = asyncio.Queue()
        self.task = asyncio.create_task(self._run(prompt))

    def cancel(self):
        if self.task and not self.task.done():
            self.task.cancel()
        self.task = None

    async def _run(self, prompt: str):
        raise NotImplementedError


class MockLLM(LLMStream):
    """TTFT и скорость генерации задаются конфигом — чтобы механику спекуляции
    можно было мерить без ключей и без денег."""

    def __init__(self, cfg):
        super().__init__()
        self.ttft = cfg["ttft_ms"] / 1000
        self.tps = cfg["tok_per_s"]
        self.name = f"mock(ttft={cfg['ttft_ms']}мс)"

    async def _run(self, prompt):
        await asyncio.sleep(self.ttft)
        self.t_first_token = time.perf_counter()
        await self.q.put(("first_token", None))
        text = ("Понял вас. Расскажите, пожалуйста, подробнее о том, "
                "какие решения вы принимали лично.")
        for w in text.split():
            await asyncio.sleep(1 / self.tps)
            await self.q.put(("token", w))
        await self.q.put(("done", None))


SYSTEM = ("Ты интервьюер на тренировочном собеседовании. Отвечай коротко, "
          "одной-двумя фразами, по-русски, и задавай следующий вопрос.")


class DeepSeekLLM(LLMStream):
    """Реальный провайдер. OpenAI-совместимый SSE-стрим."""

    def __init__(self, cfg):
        super().__init__()
        self.model = cfg.get("deepseek_model", "deepseek-chat")
        self.name = f"deepseek({self.model})"
        self.key = os.environ.get("DEEPSEEK_API_KEY")
        if not self.key:
            raise SystemExit("нет DEEPSEEK_API_KEY в окружении или .env")
        import httpx
        self.httpx = httpx

    async def _run(self, prompt):
        body = {"model": self.model, "stream": True, "max_tokens": 120,
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": prompt or "..."}]}
        async with self.httpx.AsyncClient(timeout=30) as c:
            async with c.stream("POST", "https://api.deepseek.com/chat/completions",
                                headers={"Authorization": f"Bearer {self.key}"},
                                json=body) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    delta = json.loads(data)["choices"][0]["delta"].get("content")
                    if not delta:
                        continue
                    if self.t_first_token is None:
                        self.t_first_token = time.perf_counter()
                        await self.q.put(("first_token", None))
                    await self.q.put(("token", delta))
        await self.q.put(("done", None))


def build_llm(cfg):
    if cfg["provider"] == "mock":
        return MockLLM(cfg)
    if cfg["provider"] == "deepseek":
        return DeepSeekLLM(cfg)
    raise SystemExit(f"провайдер LLM '{cfg['provider']}' не реализован")


class Pipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        import detectors as D
        from engines import REGISTRY
        self.stt = REGISTRY["gigaam"]()
        t = cfg["turn"]
        self.turn = D.Hybrid(st_thr=t["st_thr"], timeout_ms=t["timeout_ms"])
        import torch
        torch.set_num_threads(4)
        self.tts, _ = torch.hub.load("snakers4/silero-models", "silero_tts",
                                     language="ru", speaker="v4_ru", trust_repo=True)
        self.tts.to(torch.device("cpu"))
        for w in ("Прогрев.", "Ещё прогрев подлиннее."):
            self.tts.apply_tts(text=w, speaker=cfg["tts"]["voice"], sample_rate=24000)
        self.llm = build_llm(cfg["llm"])
        # mock отдаёт слова, реальный провайдер — куски текста
        self.streaming_chars = cfg["llm"]["provider"] != "mock"

    def warm(self):
        pcm = (np.random.randn(SR * 2) * 0.01).astype(np.float32)
        n = int(SR * self.cfg["stt"]["chunk_ms"] / 1000)
        self.stt.start()
        for i in range(0, len(pcm), n):
            self.stt.feed(pcm[i:i + n], (i + n) / SR)
        self.stt.finish(len(pcm) / SR)
        self.stt.events.clear()

    async def run(self, rec, speculative: bool) -> dict:
        tr = Trace()
        pcm, sr = sf.read(ROOT / rec["wav"], dtype="float32")
        assert sr == SR
        n = int(SR * self.cfg["stt"]["chunk_ms"] / 1000)
        self.stt.events.clear()
        self.stt.start()
        self.turn.reset()
        self.llm.cancel()

        spec_prompt = None
        spec_launched_at = None
        spec_cancels = 0
        spec_relaunches = 0
        primed = False
        fired_at = None

        loop_t0 = time.perf_counter()
        # Детектор конца хода тактируется кадрами 512 отсчётов (32 мс), а STT —
        # чанками 480 мс. Если гонять детектор пачкой после каждого чанка, его
        # решение квантуется по 480 мс и приезжает позже на пол-чанка в среднем.
        # Поэтому цикл идёт по мелким кадрам, а STT кормится каждые N из них.
        frames_per_chunk = max(1, n // 512)
        acc: list[np.ndarray] = []
        partial = ""
        for k in range(0, len(pcm), 512):
            target = min((k + 512) / SR, len(pcm) / SR)
            d = target - (time.perf_counter() - loop_t0)
            if d > 0:
                await asyncio.sleep(d)
            fr = pcm[k:k + 512]
            t_audio = time.perf_counter() - loop_t0
            acc.append(fr)

            before = len(self.turn.fires)
            self.turn.push(fr, t_audio)

            if len(acc) >= frames_per_chunk:
                self.stt.feed(np.concatenate(acc), t_audio)
                acc = []
                partial = self.stt.events[-1].text if self.stt.events else ""
                self.turn.push_partial(partial, t_audio)

            # Спекуляция перезапускается по мере роста транскрипта. Один
            # ранний запуск бесполезен: к моменту эндпоинта его промпт покрывает
            # лишь треть финальной фразы, и запрос приходится выбрасывать.
            # Держим в полёте самый свежий вариант.
            if speculative and len(partial) >= self.cfg["speculative"]["min_chars"]:
                grow = self.cfg["speculative"].get("relaunch_growth", 0.4)
                if spec_prompt is None or len(partial) >= len(spec_prompt) * (1 + grow):
                    if spec_prompt is not None:
                        self.llm.cancel()
                        spec_relaunches += 1
                    spec_prompt = partial
                    spec_launched_at = t_audio
                    tr.mark("spec_launch", chars=len(partial))
                    self.llm.start(partial)

            if len(self.turn.fires) > before:
                fired_at = t_audio
                tr.mark("endpoint")
                break
        if acc:
            self.stt.feed(np.concatenate(acc), time.perf_counter() - loop_t0)

        if fired_at is None:
            tr.mark("endpoint")            # речь кончилась, детектор не сработал
        final = self.stt.finish(time.perf_counter() - loop_t0)
        tr.mark("stt_final", text=final[:60])

        # Спекуляция окупается, только если ранний промпт оказался префиксом
        # финального и покрыл его заметную часть. Иначе запрос отменяется, и мы
        # платим лишь потраченными токенами — задержка не растёт.
        if speculative and spec_prompt is not None:
            cover = len(spec_prompt) / max(1, len(final))
            if final.startswith(spec_prompt[:len(spec_prompt)]) and cover >= self.cfg["speculative"].get("reuse_cover", 0.6):
                primed = True
                tr.mark("spec_reused", cover=round(cover, 2))
            else:
                self.llm.cancel()
                spec_cancels += 1
                tr.mark("spec_discarded", cover=round(cover, 2))
                self.llm.start(final)
        elif not speculative or spec_prompt is None:
            self.llm.start(final)

        first_sentence, done = [], False
        while not done:
            kind, val = await self.llm.q.get()
            if kind == "first_token":
                tr.mark("llm_first_token", primed=primed)
            elif kind == "token":
                first_sentence.append(val)
                # Режем на первой клаузе — TTS не ждёт всего ответа.
                joined = "".join(first_sentence) if self.streaming_chars else " ".join(first_sentence)
                if any(joined.rstrip().endswith(c) for c in ".!?,") and len(joined) >= 15:
                    done = True
            elif kind == "done":
                done = True
        tr.mark("llm_first_clause", n=len(first_sentence))
        self.llm.cancel()

        text = ("".join(first_sentence) if self.streaming_chars
                else " ".join(first_sentence)).strip()
        t0 = time.perf_counter()
        au = np.asarray(self.tts.apply_tts(text=text, speaker=self.cfg["tts"]["voice"],
                                           sample_rate=24000), dtype=np.float32)
        tr.mark("tts_first_chunk", audio_s=round(len(au) / 24000, 2),
                ms=round((time.perf_counter() - t0) * 1000))
        t0 = time.perf_counter()
        idx = np.arange(0, len(au), 24000 / SR)
        au16 = np.interp(idx, np.arange(len(au)), au).astype(np.float32)
        st = self.stt.rec.create_stream()
        st.accept_waveform(SR, au16)
        self.stt.rec.decode_stream(st)
        tr.mark("visemes", ms=round((time.perf_counter() - t0) * 1000),
                n=len(st.result.timestamps or []))
        tr.mark("first_sound")             # «динамик»: момент, когда можно играть

        speech_end = rec["speech_end_s"] * 1000
        # Эндпоинт раньше конца речи = ложный срез: детектор оборвал человека.
        # Такой прогон нельзя усреднять с нормальными — это другой исход.
        false_cut = bool(fired_at is not None and fired_at * 1000 < speech_end - 20)
        return {
            "false_cut": false_cut,
            "clip": rec["id"], "set": rec["set"], "speculative": speculative,
            "llm": self.llm.name,
            "endpoint_after_speech_end_ms": round(tr.at("endpoint") - speech_end)
            if tr.at("endpoint") else None,
            "stt_final_ms": tr.rel("stt_final", "endpoint"),
            "llm_ttft_ms": tr.rel("llm_first_token", "stt_final"),
            "llm_first_clause_ms": tr.rel("llm_first_clause", "llm_first_token"),
            "tts_ms": tr.rel("tts_first_chunk", "llm_first_clause"),
            "viseme_ms": tr.rel("visemes", "tts_first_chunk"),
            "total_after_speech_end_ms": round(tr.at("first_sound") - speech_end),
            "primed": primed, "spec_cancelled": spec_cancels,
            "spec_relaunches": spec_relaunches,
            "spec_launch_ms_before_endpoint": (
                round(tr.at("endpoint") - tr.at("spec_launch"))
                if tr.at("spec_launch") and tr.at("endpoint") else None),
            "marks": [{"name": n, "t_ms": round(t), **kw} for n, t, kw in tr.marks],
            "run_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config.json"))
    ap.add_argument("--clip", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--no-speculative", action="store_true")
    ap.add_argument("--both", action="store_true", help="прогнать со спекуляцией и без")
    ap.add_argument("--provider", default=None, help="переопределить llm.provider")
    a = ap.parse_args()
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    cfg = json.loads(pathlib.Path(a.config).read_text())
    if a.provider:
        cfg["llm"]["provider"] = a.provider

    recs = [json.loads(l) for l in
            (ROOT / "data" / "audio" / "manifest_synth.jsonl").read_text().splitlines()
            if json.loads(l)["set"].startswith("r2_")]
    if a.clip:
        recs = [r for r in recs if r["id"] == a.clip]
    elif not a.all:
        recs = recs[:5]

    p = Pipeline(cfg)
    p.warm()
    modes = [True, False] if a.both else [not a.no_speculative]
    out = ROOT / "bench" / "results" / "s1_latency.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with out.open("a") as f:
        for spec in modes:
            print(f"\n### спекуляция: {'ВКЛ' if spec else 'выкл'} | LLM {p.llm.name}")
            for rec in recs:
                r = await p.run(rec, spec)
                rows.append(r)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                print(f"  {r['clip']:5} эндпоинт={r['endpoint_after_speech_end_ms']:5} "
                      f"STT={r['stt_final_ms']:4} LLM={r['llm_ttft_ms']:4} "
                      f"клауза={r['llm_first_clause_ms']:4} TTS={r['tts_ms']:3} "
                      f"висемы={r['viseme_ms']:3} | ИТОГО {r['total_after_speech_end_ms']:5} мс"
                      f"{'  [prefill попал]' if r['primed'] else ''}")
    for spec in modes:
        g = [r for r in rows if r["speculative"] == spec]
        ok = [r for r in g if not r["false_cut"]]
        tot = sorted(r["total_after_speech_end_ms"] for r in ok)
        fc = len(g) - len(ok)
        if tot:
            print(f"\nспекуляция {'ВКЛ ' if spec else 'выкл'}: итог медиана {tot[len(tot)//2]} мс, "
                  f"макс {tot[-1]} мс, prefill попал в {sum(1 for r in ok if r['primed'])}/{len(ok)}"
                  f"{f', ложных срезов {fc} (исключены)' if fc else ''}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    asyncio.run(main())
