#!/usr/bin/env python3
"""Records the live half of the R2/R1 sets. Synthetic pauses have no breath and
no filled-pause formants, so an endpointer tested only on them measures the
wrong thing. Run this and read each prompt aloud NATURALLY — hesitate for real.

  python3 bench/data-gen/record_live.py            # all sets
  python3 bench/data-gen/record_live.py r2_hesitations
  python3 bench/data-gen/record_live.py --device 1 # pick input device
"""
import json, pathlib, subprocess, sys, wave, contextlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "audio" / "live"
SR = 16000


def devices():
    p = subprocess.run(["ffmpeg", "-f", "avfoundation", "-list_devices", "true",
                        "-i", ""], capture_output=True, text=True)
    return [l for l in p.stderr.splitlines() if "] [" in l and "AVFoundation" not in l]


def record(dst, dev, max_s=20):
    print("   [запись — Enter чтобы остановить]", end="", flush=True)
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "avfoundation",
                          "-i", f":{dev}", "-ac", "1", "-ar", str(SR),
                          "-t", str(max_s), str(dst)], stdin=subprocess.PIPE)
    try:
        input()
    except EOFError:
        pass
    p.communicate(b"q", timeout=5)


def dur(p):
    with contextlib.closing(wave.open(str(p))) as w:
        return w.getnframes() / w.getframerate()


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dev = "0"
    if "--device" in sys.argv:
        dev = sys.argv[sys.argv.index("--device") + 1]
    print("Аудиоустройства ввода:")
    for d in devices():
        print("  " + d.strip())
    print(f"\nПишу с устройства :{dev}. Читайте вслух ЕСТЕСТВЕННО — "
          f"на многоточии реально мнитесь, не делайте ровную паузу.\n")

    sets = args or ["r2_hesitations", "r2_terminals", "r1_terms"]
    manifest = []
    for name in sets:
        outdir = OUT / name
        outdir.mkdir(parents=True, exist_ok=True)
        for line in (ROOT / "data" / f"{name}.jsonl").read_text().splitlines():
            rec = json.loads(line)
            dst = outdir / f"{rec['id']}.wav"
            if dst.exists():
                print(f"{rec['id']}: уже записано, пропуск")
            else:
                print(f"\n{rec['id']}: «{rec['text']}»")
                record(dst, dev)
                print(f"   -> {dur(dst):.2f} с")
            rec.update(set=name, wav=str(dst.relative_to(ROOT)),
                       duration_s=round(dur(dst), 3), source="live")
            manifest.append(rec)
    mf = ROOT / "data" / "audio" / "manifest_live.jsonl"
    mf.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in manifest))
    print(f"\n{len(manifest)} записей -> {mf.relative_to(ROOT)}")
    print("Теперь: python3 bench/data-gen/annotate.py --live")


if __name__ == "__main__":
    main()
