#!/usr/bin/env python3
"""R2 — end-of-turn detection. The number that matters is the false-cut rate:
how often a detector decides the person is done while they are mid-sentence.

Latency alone is a trap — a detector that never fires has perfect latency on
nothing, so terminal control phrases are scored in the same run.

  ./.venv/bin/python run.py --detectors all
  ./.venv/bin/python run.py --detectors energy-700 smart-turn --manifest manifest_live.jsonl
"""
import argparse, json, pathlib, sys, time
import numpy as np
import soundfile as sf

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import detectors as D                                        # noqa: E402

SR = 16000
FRAME = 512          # 32 ms — what Silero requires; everything else follows


def build(spec: str):
    if spec.startswith("energy-"):
        return D.EnergySilence(n_ms=int(spec.split("-")[1]))
    if spec.startswith("silero-"):
        return D.SileroSilence(n_ms=int(spec.split("-")[1]))
    if spec == "smart-turn":
        return D.SmartTurnV3()
    if spec.startswith("smart-turn-"):
        return D.SmartTurnV3(thr=float(spec.rsplit("-", 1)[1]))
    if spec == "hybrid":
        return D.Hybrid()
    if spec == "hybrid-noveto":
        return D.Hybrid(use_text_veto=False)
    if spec.startswith("hybrid-st"):
        thr, *rest = spec[9:].split("-t")
        return D.Hybrid(st_thr=float(thr),
                        timeout_ms=int(rest[0]) if rest else 1500)
    if spec.startswith("hybrid-t"):
        return D.Hybrid(timeout_ms=int(spec.split("-t")[1]))
    if spec == "partial-rule":
        return D.PartialRule()
    raise SystemExit(f"неизвестный детектор: {spec}")


DEFAULT = ["energy-500", "energy-700", "energy-1000", "energy-1500",
           "silero-700", "silero-1000", "smart-turn", "partial-rule"]


def load_partials(path, engine_key):
    """STT partials from the R1 run, replayed by timestamp so the text-based
    detectors see exactly what the pipeline would have handed them. Which STT
    produced them matters: a text detector inherits that STT's latency."""
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if engine_key in r["engine"]:
            out[r["clip"]] = r["partials"]
    return out


def score(rec, fires):
    """A fire inside a mid-phrase pause cut the person off. A fire at or after
    the end of speech is a legitimate endpoint; the first one sets latency."""
    wins = rec.get("forbidden_windows_s", [])
    end = rec["speech_end_s"]
    false_cuts = [f for f in fires if any(a - 0.02 <= f <= b + 0.02 for a, b in wins)]
    valid = [f for f in fires if f >= end - 0.02]
    early = [f for f in fires if f < end - 0.02 and f not in false_cuts]
    return {
        "n_fires": len(fires),
        "false_cuts": len(false_cuts),
        "false_cut_windows": sorted({next(i for i, (a, b) in enumerate(wins)
                                          if a - 0.02 <= f <= b + 0.02) for f in false_cuts}),
        "cut_in_speech": len(early),
        "endpoint_latency_ms": round((valid[0] - end) * 1000) if valid else None,
        "fires_s": [round(f, 3) for f in fires],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detectors", nargs="+", default=["all"])
    ap.add_argument("--manifest", default="manifest_synth.jsonl")
    ap.add_argument("--partials", default="r1_stt_synth.jsonl")
    ap.add_argument("--partial-engine", default="gigaam",
                    help="какой STT из R1 подаёт партиалы текстовым детекторам")
    a = ap.parse_args()

    recs = [json.loads(l) for l in (ROOT / "data" / "audio" / a.manifest).read_text().splitlines()
            if json.loads(l)["set"].startswith("r2_")]
    partials = load_partials(ROOT / "bench" / "results" / a.partials, a.partial_engine)
    specs = DEFAULT if a.detectors == ["all"] else a.detectors
    out = ROOT / "bench" / "results" / f"r2_endpoint_{a.manifest.replace('manifest_','').replace('.jsonl','')}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("a") as fh:
        for spec in specs:
            det = build(spec)
            if det.needs_partials and not partials:
                print(f"!! {spec}: нет партиалов ({a.partials}) — пропуск"); continue
            print(f"\n### {det.name} [{det.kind}]", flush=True)
            agg = {"fc": 0, "clips_fc": 0, "hes": 0, "lat": [], "term_miss": 0, "term": 0}
            for rec in recs:
                pcm, sr = sf.read(ROOT / rec["wav"], dtype="float32")
                assert sr == SR
                det.reset()
                pl, pi = partials.get(rec["id"], []), 0
                for i in range(0, len(pcm), FRAME):
                    t = min((i + FRAME) / SR, len(pcm) / SR)
                    while pi < len(pl) and pl[pi]["t_ms"] / 1000 <= t:
                        det.push_partial(pl[pi]["text"], pl[pi]["t_ms"] / 1000); pi += 1
                    det.push(pcm[i:i + FRAME], t)
                s = score(rec, det.fires)
                row = {"detector": det.name, "kind": det.kind, "spec": spec,
                       "partial_engine": a.partial_engine if det.needs_partials else None,
                       "clip": rec["id"], "set": rec["set"], "source": rec.get("source"),
                       "text": rec["text"], "speech_end_s": rec["speech_end_s"],
                       "forbidden_windows_s": rec.get("forbidden_windows_s", []),
                       **s, "run_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
                if isinstance(det, (D.SmartTurnV3, D.Hybrid)):
                    row["probs"] = [(round(t, 2), round(p, 4)) for t, p in det.probs]
                    row["infer_ms_p50"] = round(float(np.median(det.infer_ms)), 1) if det.infer_ms else None
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                if rec["set"] == "r2_hesitations":
                    agg["hes"] += 1; agg["fc"] += s["false_cuts"]
                    agg["clips_fc"] += 1 if s["false_cuts"] else 0
                else:
                    agg["term"] += 1
                    if s["endpoint_latency_ms"] is None: agg["term_miss"] += 1
                if s["endpoint_latency_ms"] is not None:
                    agg["lat"].append(s["endpoint_latency_ms"])
            fh.flush()
            lat = sorted(agg["lat"])
            p50 = lat[len(lat)//2] if lat else None
            print(f"  ложных срезов: {agg['fc']} в {agg['clips_fc']}/{agg['hes']} фразах "
                  f"({100*agg['clips_fc']/max(1,agg['hes']):.0f}% фраз испорчено)")
            print(f"  задержка эндпоинта p50: {p50} мс | терминальных пропущено: "
                  f"{agg['term_miss']}/{agg['term']}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
