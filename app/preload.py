#!/usr/bin/env python3
"""Скачать модели заранее, чтобы показ не зависел от сети.

    .venv/bin/python app/preload.py

Silero тянется через `torch.hub`, GigaAM — с Hugging Face. Оба кладут файлы в
кеш пользователя и при повторном запуске ничего не качают. Без этого шага
первый `server.py` уходит в сеть на минуты, и делает это ровно в тот момент,
когда на экран уже смотрят.

Скрипт печатает, что и куда легло, и завершается ненулевым кодом, если хоть
одна модель не поднялась — «кажется, скачалось» не годится за проверку.
"""
import argparse
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.media import GigaAMAligner, SileroTTS, tts_config    # noqa: E402


def human(n: float) -> str:
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} ТБ"


def dir_size(path: pathlib.Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default=None, help="по умолчанию из app/config.json")
    args = ap.parse_args()

    cfg = tts_config()
    silero = cfg.get("silero", {})
    voice = args.voice or silero.get("voice", "ru_roman")
    version = silero.get("version", "v5_cis_base")

    torch_cache = pathlib.Path.home() / ".cache" / "torch"
    hf_cache = pathlib.Path.home() / ".cache" / "huggingface"

    print("Предзагрузка моделей. Первый раз это минуты, дальше секунды.\n")
    ok = True

    print(f"1/2  Silero {version}, голос {voice}")
    t0 = time.perf_counter()
    try:
        tts = SileroTTS(voice=voice, version=version)
        pcm, sr = tts("Проверка.")
        print(f"     готово за {time.perf_counter() - t0:.1f} с, "
              f"пробный синтез {len(pcm) / sr:.1f} с звука")
        print(f"     кеш: {torch_cache}  ({human(dir_size(torch_cache))})")
    except Exception as e:                                     # noqa: BLE001
        ok = False
        print(f"     НЕ ПОДНЯЛОСЬ: {type(e).__name__}: {e}")

    print(f"\n2/2  GigaAM v3 CTC ({GigaAMAligner.REPO})")
    t0 = time.perf_counter()
    try:
        al = GigaAMAligner()
        chars = al(pcm, sr) if ok else []
        print(f"     готово за {time.perf_counter() - t0:.1f} с, "
              f"пробное выравнивание {len(chars)} символов")
        print(f"     кеш: {hf_cache}  ({human(dir_size(hf_cache))})")
    except Exception as e:                                     # noqa: BLE001
        ok = False
        print(f"     НЕ ПОДНЯЛОСЬ: {type(e).__name__}: {e}")

    if ok:
        print("\nОбе модели на месте. Дальше сервер поднимается без сети:")
        print("    .venv/bin/python app/server.py")
        return 0
    print("\nЧего-то не хватает — см. выше. Сеть? Место на диске?")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
