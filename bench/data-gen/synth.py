#!/usr/bin/env python3
"""Synthesises the baseline audio set with macOS `say` (Milena, ru_RU).

Hesitation pauses are real silence spliced between spoken chunks, split on the
'…' markers in the source text. This reproduces the *timing* of a hesitation but
NOT its acoustics (no breath, no filled-pause formants, flat prosody), so it is a
strictly easier case than live speech. Live recordings are mandatory for R2.
"""
import json, pathlib, subprocess, sys, wave, contextlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "audio" / "synth"
VOICE, RATE, SR = "Milena", 175, 16000
LEAD_MS, TAIL_MS = 400, 2500   # lead: room for a detector to learn the noise
                               # floor; tail: room for a 1500 ms rule to fire


def say(text: str, dst: pathlib.Path):
    aiff = dst.with_suffix(".aiff")
    subprocess.run(["say", "-v", VOICE, "-r", str(RATE), "-o", str(aiff), text], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff),
                    "-ac", "1", "-ar", str(SR), str(dst)], check=True)
    aiff.unlink()


def silence(ms: int, dst: pathlib.Path):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", f"anullsrc=r={SR}:cl=mono", "-t", f"{ms/1000:.3f}",
                    "-ac", "1", "-ar", str(SR), str(dst)], check=True)


def concat(parts, dst: pathlib.Path):
    lst = dst.with_suffix(".txt")
    lst.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat",
                    "-safe", "0", "-i", str(lst), "-c", "copy", str(dst)], check=True)
    lst.unlink()


def dur(p: pathlib.Path) -> float:
    with contextlib.closing(wave.open(str(p))) as w:
        return w.getnframes() / w.getframerate()


def build_hesitation(rec, outdir):
    dst = outdir / f"{rec['id']}.wav"
    chunks = [c.strip() for c in rec["text"].split("…") if c.strip()]
    tmp, parts = [], []
    for i, c in enumerate(chunks):
        p = outdir / f".{rec['id']}_{i}.wav"
        say(c, p); tmp.append(p); parts.append(p)
        if i < len(chunks) - 1:
            s = outdir / f".{rec['id']}_sil{i}.wav"
            silence(rec["pause_ms"], s); tmp.append(s); parts.append(s)
    tail = outdir / f".{rec['id']}_tail.wav"
    silence(TAIL_MS, tail); tmp.append(tail); parts.append(tail)
    lead = outdir / f".{rec['id']}_lead.wav"
    silence(LEAD_MS, lead); tmp.append(lead)
    concat([lead] + parts, dst)
    for p in tmp:
        p.unlink(missing_ok=True)
    return dst


def main():
    sets = {
        "r2_hesitations": (OUT / "r2_hesitations", True),
        "r2_terminals":   (OUT / "r2_terminals", False),
        "r1_terms":       (OUT / "r1_terms", False),
    }
    manifest = []
    for name, (outdir, hes) in sets.items():
        outdir.mkdir(parents=True, exist_ok=True)
        for line in (ROOT / "data" / f"{name}.jsonl").read_text().splitlines():
            rec = json.loads(line)
            if hes:
                dst = build_hesitation(rec, outdir)
            else:
                dst = outdir / f"{rec['id']}.wav"
                body = outdir / f".{rec['id']}_b.wav"
                tail = outdir / f".{rec['id']}_t.wav"
                lead = outdir / f".{rec['id']}_l.wav"
                say(rec["text"], body); silence(TAIL_MS, tail); silence(LEAD_MS, lead)
                concat([lead, body, tail], dst)
                body.unlink(); tail.unlink(); lead.unlink()
            rec.update(set=name, wav=str(dst.relative_to(ROOT)),
                       duration_s=round(dur(dst), 3), source="synth:macos-say-Milena")
            manifest.append(rec)
            print(f"{rec['id']:5} {rec['duration_s']:6.2f}s  {rec['text'][:56]}")
    mf = ROOT / "data" / "audio" / "manifest_synth.jsonl"
    mf.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in manifest))
    print(f"\n{len(manifest)} файлов, манифест: {mf.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
