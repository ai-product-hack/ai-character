#!/usr/bin/env python3
"""R1 — streaming STT bench. Feeds audio at wall-clock pace, exactly as a mic
would, so a model slower than real time shows up as lag instead of hiding in a
batch number.

  ./.venv/bin/python run.py --engines parakeet whisper-mlx --set r1_terms
  ./.venv/bin/python run.py --engines all --set all --chunk-ms 480
"""
import argparse, json, pathlib, sys, time
import numpy as np
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from engines import REGISTRY                      # noqa: E402
from metrics import wer, term_hits                # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "bench" / "results"


def load(wav):
    pcm, sr = sf.read(ROOT / wav, dtype="float32")
    assert sr == 16000, f"{wav}: {sr} Hz, ожидалось 16000"
    return pcm


def speech_onset(rec):
    """Audio time where speech starts. Живая разметка кладёт это прямо
    (из выравнивания); синтетическая выводится из ведущей тишины."""
    if rec.get("speech_onset_s") is not None:
        return rec["speech_onset_s"]
    for a, b in rec.get("silences_s", []):
        if a < 0.05:
            return b
    return 0.0


def warmup(engine, chunk_ms):
    """First inference pays for graph compilation / kernel autotune. Without a
    warm-up that cost lands on clip #1 and pollutes every first-partial number."""
    n = int(16000 * chunk_ms / 1000)
    pcm = (np.random.randn(16000 * 3) * 0.01).astype(np.float32)
    engine.events.clear()
    engine.start()
    for i in range(0, len(pcm), n):
        engine.feed(pcm[i:i + n], (i + n) / 16000)
    engine.finish(len(pcm) / 16000)
    engine.events.clear()


def run_clip(engine, rec, chunk_ms):
    pcm = load(rec["wav"])
    n = int(16000 * chunk_ms / 1000)
    engine.events.clear()
    engine.start()
    t0 = time.perf_counter()
    max_lag = 0.0
    for i in range(0, len(pcm), n):
        target = min((i + n) / 16000, len(pcm) / 16000)
        lag = (time.perf_counter() - t0) - target
        max_lag = max(max_lag, lag)
        if lag < 0:
            time.sleep(-lag)
        engine.feed(pcm[i:i + n], time.perf_counter() - t0)
    t_fin0 = time.perf_counter()
    engine.finish(time.perf_counter() - t0)
    finalize_ms = (time.perf_counter() - t_fin0) * 1000

    onset = speech_onset(rec)
    end = rec.get("speech_end_s", len(pcm) / 16000)
    parts = [e for e in engine.events if e.kind == "partial" and e.text]
    fin = engine.events[-1]

    w, wd = wer(rec["text"], fin.text)
    return {
        "engine": engine.name, "mode": engine.mode, "clip": rec["id"],
        "set": rec["set"], "source": rec.get("source"), "chunk_ms": chunk_ms,
        "audio_s": round(len(pcm) / 16000, 3),
        "speech_onset_s": round(onset, 3), "speech_end_s": round(end, 3),
        "t_first_partial_ms": round((parts[0].t - onset) * 1000) if parts else None,
        "t_final_after_speech_end_ms": round((fin.t - end) * 1000),
        "finalize_compute_ms": round(finalize_ms),
        "max_lag_ms": round(max_lag * 1000),
        "rtf_worst_chunk": round(max((e.compute_ms / chunk_ms) for e in engine.events
                                     if e.compute_ms) if any(e.compute_ms for e in engine.events) else 0, 2),
        "n_partials": len(parts),
        "wer": round(w, 4), **wd,
        "ref": rec["text"], "hyp": fin.text,
        "terms": term_hits(fin.text, rec["terms"]) if rec.get("terms") else None,
        "partials": [{"t_ms": round(e.t * 1000), "text": e.text,
                      "compute_ms": round(e.compute_ms)} for e in parts],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", nargs="+", default=["parakeet"])
    ap.add_argument("--set", dest="sets", nargs="+", default=["r1_terms"])
    ap.add_argument("--chunk-ms", type=int, default=480)
    ap.add_argument("--manifest", default="manifest_synth.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    recs = [json.loads(l) for l in (ROOT / "data" / "audio" / a.manifest).read_text().splitlines()]
    if "all" not in a.sets:
        recs = [r for r in recs if r["set"] in a.sets]
    if a.limit:
        recs = recs[:a.limit]
    names = list(REGISTRY) if a.engines == ["all"] else a.engines

    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / f"r1_stt_{a.manifest.replace('manifest_','').replace('.jsonl','')}.jsonl"
    with dst.open("a") as f:
        for name in names:
            print(f"\n### {name} — загрузка модели…", flush=True)
            t0 = time.perf_counter()
            eng = REGISTRY[name]()
            print(f"    загружена за {time.perf_counter()-t0:.1f} с ({eng.mode}); прогрев…", flush=True)
            t0 = time.perf_counter()
            warmup(eng, a.chunk_ms)
            print(f"    прогрет за {time.perf_counter()-t0:.1f} с", flush=True)
            for r in recs:
                res = run_clip(eng, r, a.chunk_ms)
                res["run_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                f.write(json.dumps(res, ensure_ascii=False) + "\n"); f.flush()
                print(f"  {r['id']:5} WER={res['wer']:.3f} "
                      f"1st={res['t_first_partial_ms']} мс "
                      f"fin={res['t_final_after_speech_end_ms']} мс "
                      f"lag={res['max_lag_ms']} мс | {res['hyp'][:60]}", flush=True)
    print(f"\n-> {dst}")


if __name__ == "__main__":
    main()
