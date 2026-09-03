#!/usr/bin/env python3
"""Annotates each clip with measured silence intervals -> ground truth for R2.

  forbidden_windows : mid-phrase pauses. An endpointer firing inside one is a
                      FALSE CUT — it interrupted a person who was still talking.
  speech_end_s      : onset of the final trailing silence. Endpoint latency for
                      terminal phrases is measured from here.
"""
import json, pathlib, re, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
LIVE = "--live" in sys.argv
MF = ROOT / "data" / "audio" / ("manifest_live.jsonl" if LIVE else "manifest_synth.jsonl")
if not MF.exists():
    sys.exit(f"нет {MF} — сначала запишите набор (bench/data-gen/record_live.py)")
PAT = re.compile(r"silence_(start|end): ([0-9.]+)")


def noise_floor_db(wav: pathlib.Path) -> float:
    """Live recordings have room tone, so the silence gate has to be relative to
    the actual floor of the clip rather than a fixed -45 dBFS."""
    p = subprocess.run(["ffmpeg", "-i", str(wav), "-af", "volumedetect",
                        "-f", "null", "-"], capture_output=True, text=True)
    m = re.search(r"mean_volume: (-?[0-9.]+) dB", p.stderr)
    return float(m.group(1)) if m else -60.0


def silences(wav: pathlib.Path, thresh="-45dB", mind=0.30):
    p = subprocess.run(["ffmpeg", "-i", str(wav), "-af",
                        f"silencedetect=n={thresh}:d={mind}", "-f", "null", "-"],
                       capture_output=True, text=True)
    vals, out, cur = PAT.findall(p.stderr), [], None
    for kind, v in vals:
        if kind == "start":
            cur = float(v)
        elif cur is not None:
            out.append((round(cur, 3), round(float(v), 3)))
            cur = None
    return out


rows = []
for line in MF.read_text().splitlines():
    r = json.loads(line)
    wav = ROOT / r["wav"]
    thresh = f"{noise_floor_db(wav) - 12:.0f}dB" if LIVE else "-45dB"
    sil = silences(wav, thresh=thresh)
    r["silence_gate_db"] = thresh
    tail = [s for s in sil if abs(s[1] - r["duration_s"]) < 0.05]
    lead = [s for s in sil if s[0] < 0.05]
    mid = [s for s in sil if s not in tail and s not in lead]
    r["speech_onset_s"] = lead[0][1] if lead else 0.0
    r["silences_s"] = sil
    r["forbidden_windows_s"] = mid
    r["speech_end_s"] = tail[0][0] if tail else r["duration_s"]
    rows.append(r)

MF.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
h = [r for r in rows if r["set"] == "r2_hesitations"]
print(f"{len(rows)} клипов размечено ({'live' if LIVE else 'synth'})")
print(f"hesitations: {sum(len(r['forbidden_windows_s']) for r in h)} запретных окон "
      f"на {len(h)} фразах; средняя пауза "
      f"{sum(b-a for r in h for a,b in r['forbidden_windows_s'])/max(1,sum(len(r['forbidden_windows_s']) for r in h)):.2f} с")
miss = [r["id"] for r in h if not r["forbidden_windows_s"]]
print("без запретных окон (проверить вручную):", miss or "нет")
