#!/usr/bin/env python3
"""Записывает живую половину наборов R1/R2.

Синтетические паузы — ровная цифровая тишина без дыхания и без формант
заполненной паузы. Эндпоинтер, протестированный только на них, меряет не то,
поэтому живые записи обязательны.

    python3 bench/data-gen/record_live.py --device 1
    python3 bench/data-gen/record_live.py --device 1 --sets r2_hesitations
    python3 bench/data-gen/record_live.py --list-devices
    python3 bench/data-gen/record_live.py --device 1 --redo h03 h07

Enter — начать запись, Enter — остановить. 'п' + Enter — пропустить фразу,
'з' + Enter — перезаписать предыдущую.
"""
import argparse, contextlib, json, pathlib, signal, subprocess, sys, wave

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "audio" / "live"
SR = 16000
SETS = ["r2_hesitations", "r2_terminals", "r1_terms"]


def devices():
    p = subprocess.run(["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                       capture_output=True, text=True)
    out, audio = [], False
    for line in p.stderr.splitlines():
        if "AVFoundation audio devices" in line:
            audio = True
            continue
        if "AVFoundation video devices" in line:
            audio = False
        if audio and "] [" in line:
            out.append(line.split("] ", 1)[-1])
    return out


def dur_rms(path):
    """Длительность и громкость. Нулевая громкость почти всегда означает, что
    macOS не дал терминалу доступ к микрофону — ffmpeg при этом не падает,
    а пишет тишину, и без проверки это обнаружится только на бенчмарке."""
    with contextlib.closing(wave.open(str(path))) as w:
        n, sr = w.getnframes(), w.getframerate()
        raw = w.readframes(n)
    if not n:
        return 0.0, 0.0
    import array
    a = array.array("h")
    a.frombytes(raw)
    rms = (sum(x * x for x in a) / len(a)) ** 0.5 / 32768
    return n / sr, rms


def record(dst: pathlib.Path, dev: str) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".rec.wav")
    p = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "avfoundation",
         "-i", f":{dev}", "-ac", "1", "-ar", str(SR), str(tmp)],
        stdin=subprocess.DEVNULL)
    try:
        input("   ● запись… Enter — стоп ")
    except (EOFError, KeyboardInterrupt):
        pass
    # SIGINT, а не 'q' в stdin: ffmpeg по нему корректно закрывает WAV-заголовок.
    p.send_signal(signal.SIGINT)
    try:
        p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        p.kill(); p.wait()
    if not tmp.exists() or tmp.stat().st_size < 1000:
        print("   !! ffmpeg не записал файл. Проверьте номер устройства "
              "(--list-devices) и доступ к микрофону в Системных настройках → "
              "Конфиденциальность → Микрофон.")
        tmp.unlink(missing_ok=True)
        return False
    d, rms = dur_rms(tmp)
    if rms < 0.001:
        print(f"   !! записана тишина ({d:.1f} с, RMS {rms:.5f}). Скорее всего "
              "терминалу не разрешён доступ к микрофону.")
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(dst)
    print(f"   -> {d:.2f} с, RMS {rms:.3f}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="1", help="индекс аудиоустройства avfoundation")
    ap.add_argument("--sets", nargs="+", default=SETS, choices=SETS,
                    help="какие наборы писать")
    ap.add_argument("--redo", nargs="+", default=[], metavar="ID",
                    help="перезаписать конкретные фразы, например: --redo h03 h07")
    ap.add_argument("--list-devices", action="store_true")
    a = ap.parse_args()

    devs = devices()
    if a.list_devices:
        print("Аудиоустройства ввода (avfoundation):")
        for d in devs:
            print("  " + d)
        return

    print("Аудиоустройства ввода:")
    for d in devs:
        print("  " + d)
    print(f"\nПишу с устройства :{a.device}. Читайте вслух ЕСТЕСТВЕННО — "
          "на многоточии реально мнитесь, не делайте ровную паузу.")
    print("Enter — начать, Enter — стоп. 'п' — пропустить, 'з' — перезаписать.\n")

    manifest, total, done = [], 0, 0
    for name in a.sets:
        src = ROOT / "data" / f"{name}.jsonl"
        if not src.exists():
            sys.exit(f"нет файла набора: {src}")
        recs = [json.loads(l) for l in src.read_text().splitlines()]
        if a.redo:
            recs = [r for r in recs if r["id"] in a.redo]
        total += len(recs)
        outdir = OUT / name
        for rec in recs:
            dst = outdir / f"{rec['id']}.wav"
            if dst.exists() and rec["id"] not in a.redo:
                print(f"{rec['id']}: уже записано, пропуск")
            else:
                while True:
                    print(f"\n{rec['id']}: «{rec['text']}»")
                    cmd = input("   Enter — начать (п — пропустить): ").strip().lower()
                    if cmd in ("п", "p", "skip"):
                        break
                    if record(dst, a.device):
                        again = input("   Enter — дальше, 'з' — перезаписать: ").strip().lower()
                        if again not in ("з", "z"):
                            break
                    else:
                        again = input("   Повторить? Enter — да, 'п' — пропустить: ").strip().lower()
                        if again in ("п", "p"):
                            break
            if dst.exists():
                d, _ = dur_rms(dst)
                rec.update(set=name, wav=str(dst.relative_to(ROOT)),
                           duration_s=round(d, 3), source="live")
                manifest.append(rec)
                done += 1

    if not manifest:
        print("\nничего не записано")
        return
    mf = ROOT / "data" / "audio" / "manifest_live.jsonl"
    old = {}
    if mf.exists():
        old = {json.loads(l)["id"]: json.loads(l) for l in mf.read_text().splitlines()}
    for r in manifest:
        old[r["id"]] = r
    mf.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in old.values()))
    print(f"\nзаписано {done} из {total} -> {mf.relative_to(ROOT)}")
    print("Дальше: python3 bench/data-gen/annotate.py --live")


if __name__ == "__main__":
    main()
