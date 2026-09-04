#!/usr/bin/env python3
"""Второй раунд сравнения TTS: Silero v5 против v4, SSML, ударения.

Повод пересобрать: заметка R-фазы утверждала, что `v5_cis_base` недостижим
через точку входа torch.hub. Это неверно — модель грузится, и разница
существенная:

    v4_ru          5 русских голосов, разрешения омографов нет
    v5_cis_base   29 русских голосов, put_stress_homo встроен

Омографы важны не для красоты: одна ошибка ударения («зАмок» вместо «замОк»)
выдаёт машину мгновенно. Внешний словарь ruaccent для этого не годится — его
ONNX-модель омографов падает на несовпадении входов, «замок» в обоих значениях
получает одно ударение.

Голосов у v5 слишком много, чтобы слушать все на десяти репликах, поэтому
раунд двухступенчатый: сначала короткий список, потом полный набор на
финалистах.

    bench/r1-stt/.venv/bin/python bench/r3-tts/round2.py --stage shortlist
    bench/r1-stt/.venv/bin/python bench/r3-tts/round2.py --stage full --voices ru_eduard,ru_bogdan
"""
import argparse, json, pathlib, re, statistics, sys, time, wave

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench" / "r3-tts"))
from compare_voices import REPLIES, PARENTHETICALS, to_ssml, first_clause  # noqa: E402

OUT = ROOT / "bench" / "r3-tts" / "round2"
RESULTS = ROOT / "bench" / "results"
SR = 24000


def write_wav(path, audio, sr=SR):
    x = np.clip(np.asarray(audio, dtype=np.float32), -1, 1)
    x = (x * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(x.tobytes())


def load(speaker):
    import torch
    torch.set_num_threads(4)
    t0 = time.perf_counter()
    m, _ = torch.hub.load("snakers4/silero-models", "silero_tts",
                          language="ru", speaker=speaker, trust_repo=True)
    m.to(torch.device("cpu"))
    return m, time.perf_counter() - t0


def warm(model, voice):
    # Замерено в R3: первые ДВА вызова стоят ~520 мс, потом ~10 мс.
    for t in ("Прогрев.", "Ещё один прогрев, подлиннее."):
        model.apply_tts(text=t, speaker=voice, sample_rate=SR)


def synth(model, voice, text=None, ssml=None, **kw):
    t0 = time.perf_counter()
    kwargs = dict(speaker=voice, sample_rate=SR, **kw)
    au = model.apply_tts(ssml_text=ssml, **kwargs) if ssml else \
        model.apply_tts(text=text, **kwargs)
    return np.asarray(au, dtype=np.float32), (time.perf_counter() - t0) * 1000


def stage_shortlist(rows):
    """Все русские голоса v5 на двух репликах: короткой и длинной."""
    m5, load5 = load("v5_cis_base")
    voices = [v for v in m5.speakers if v.startswith("ru_")]
    print(f"v5 загружена за {load5:.1f} с, русских голосов {len(voices)}")
    picks = [REPLIES[4], REPLIES[1]]          # короткая и длинная
    for v in voices:
        d = OUT / "shortlist" / v
        d.mkdir(parents=True, exist_ok=True)
        warm(m5, v)
        for rep in picks:
            au, ms = synth(m5, v, text=rep["text"])
            write_wav(d / f"{rep['id']}.wav", au)
            rows.append({"stage": "shortlist", "engine": "v5_cis_base", "voice": v,
                         "reply_id": rep["id"], "synth_ms": round(ms),
                         "audio_s": round(len(au) / SR, 3)})
        print(f"  {v}")
    # Тот же материал текущим голосом, чтобы было с чем сравнивать.
    m4, _ = load("v4_ru")
    warm(m4, "eugene")
    d = OUT / "shortlist" / "v4_eugene"
    d.mkdir(parents=True, exist_ok=True)
    for rep in picks:
        au, ms = synth(m4, "eugene", text=rep["text"])
        write_wav(d / f"{rep['id']}.wav", au)
        rows.append({"stage": "shortlist", "engine": "v4_ru", "voice": "eugene",
                     "reply_id": rep["id"], "synth_ms": round(ms),
                     "audio_s": round(len(au) / SR, 3)})
    print("  v4_eugene (текущий)")
    return voices


def stage_full(rows, voices):
    """Финалисты на всех десяти репликах, плюс SSML и омографы."""
    m5, _ = load("v5_cis_base")
    for v in voices:
        for tag, ssml in (("plain", False), ("ssml", True)):
            d = OUT / "full" / f"v5_{v}_{tag}"
            d.mkdir(parents=True, exist_ok=True)
            warm(m5, v)
            for rep in REPLIES:
                au, ms = synth(m5, v, text=None if ssml else rep["text"],
                               ssml=to_ssml(rep["text"]) if ssml else None)
                write_wav(d / f"{rep['id']}.wav", au)
                rows.append({"stage": "full", "engine": "v5_cis_base", "voice": v,
                             "variant": tag, "reply_id": rep["id"],
                             "synth_ms": round(ms), "audio_s": round(len(au) / SR, 3)})
            print(f"  v5_{v}_{tag}")

    m4, _ = load("v4_ru")
    warm(m4, "eugene")
    d = OUT / "full" / "v4_eugene_plain"
    d.mkdir(parents=True, exist_ok=True)
    for rep in REPLIES:
        au, ms = synth(m4, "eugene", text=rep["text"])
        write_wav(d / f"{rep['id']}.wav", au)
        rows.append({"stage": "full", "engine": "v4_ru", "voice": "eugene",
                     "variant": "plain", "reply_id": rep["id"],
                     "synth_ms": round(ms), "audio_s": round(len(au) / SR, 3)})
    print("  v4_eugene_plain (текущий)")

    # Омографы: одно предложение, где ошибка слышна сразу.
    omo = ("На двери висит замок, а вдали виден старинный замок. "
           "Большая часть команды уже дома.")
    d = OUT / "omographs"
    d.mkdir(parents=True, exist_ok=True)
    for name, model, voice, kw in (
            ("v4_eugene", m4, "eugene", {}),
            ("v5_homo_on", m5, voices[0], {"put_stress_homo": True}),
            ("v5_homo_off", m5, voices[0], {"put_stress_homo": False})):
        au, ms = synth(model, voice, text=omo, **kw)
        write_wav(d / f"{name}.wav", au)
        rows.append({"stage": "omographs", "variant": name,
                     "synth_ms": round(ms), "audio_s": round(len(au) / SR, 3)})
    print("  омографы: v4 против v5 с разрешением и без")

    # TTFB на реальной длине для финалистов.
    ttfb = []
    for v in voices:
        warm(m5, v)
        for rep in REPLIES:
            fc = first_clause(rep["text"])
            _, ms_first = synth(m5, v, text=fc)
            au_full, ms_full = synth(m5, v, text=rep["text"])
            ttfb.append({"engine": "v5_cis_base", "voice": v, "reply_id": rep["id"],
                         "ttfb_first_clause_ms": round(ms_first),
                         "synth_full_ms": round(ms_full),
                         "rtf_full": round(ms_full / 1000 / (len(au_full) / SR), 4)})
    (RESULTS / "r3_tts_round2_ttfb.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in ttfb) + "\n", encoding="utf-8")
    fc = sorted(r["ttfb_first_clause_ms"] for r in ttfb)
    fu = sorted(r["synth_full_ms"] for r in ttfb)
    print(f"\nv5 TTFB первой клаузы: медиана {fc[len(fc)//2]}, макс {fc[-1]} мс")
    print(f"v5 синтез целиком:     медиана {fu[len(fu)//2]}, макс {fu[-1]} мс")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("shortlist", "full"), default="shortlist")
    ap.add_argument("--voices", default="", help="через запятую, для стадии full")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    if args.stage == "shortlist":
        voices = stage_shortlist(rows)
        write_index_shortlist(voices)
    else:
        picked = [v.strip() for v in args.voices.split(",") if v.strip()]
        if not picked:
            raise SystemExit("для стадии full нужен --voices")
        stage_full(rows, picked)
        write_index_full(picked)

    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / f"r3_tts_round2_{args.stage}.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                 encoding="utf-8")
    ms = sorted(r["synth_ms"] for r in rows if "synth_ms" in r)
    print(f"\nсинтез: медиана {ms[len(ms)//2]} мс, макс {ms[-1]} мс")
    print(f"-> {OUT}")


def write_index_shortlist(voices):
    lines = [
        "# Второй раунд: короткий список голосов",
        "",
        "**Оценку на слух делает человек.** Здесь только материал.",
        "",
        "## Почему раунд вообще есть",
        "",
        "Заметка R-фазы утверждала, что `v5_cis_base` недостижим через точку входа",
        "`torch.hub`. Это неверно: модель грузится (87 МБ, 52 с в первый раз, дальше",
        "из кэша). Разница с тем, на чём мы сидели:",
        "",
        "| | v4_ru | v5_cis_base |",
        "|---|---|---|",
        f"| русских голосов | 5 | **{len(voices)}** |",
        "| разрешение омографов | нет | **`put_stress_homo`** |",
        "| постановка ё | `put_yo` | `put_yo` + `put_yo_homo` |",
        "| скорость синтеза | ~36 мс на клаузу | сопоставима |",
        "",
        "Омографы важны не для красоты: «зАмок» вместо «замОк» выдаёт машину",
        "мгновенно. Внешний словарь `ruaccent` для этого не подошёл — его ONNX-модель",
        "омографов падает на несовпадении входов, и оба «замка» получают одно",
        "ударение. У v5 разрешение встроено.",
        "",
        "## Что слушать",
        "",
        "Голосов слишком много для десяти реплик каждый, поэтому сначала короткий",
        "список: **две реплики на голос**, короткая и длинная.",
        "",
        f"- `shortlist/ru_*` — {len(voices)} голосов v5",
        "- `shortlist/v4_eugene` — то, на чём мы сейчас, для сравнения",
        "",
        "Реплики: `r05` (короткая, холодный звонок) и `r02` (длинная, с",
        "перечислением и числами).",
        "",
        "После выбора двух-трёх финалистов:",
        "",
        "```",
        "bench/r1-stt/.venv/bin/python bench/r3-tts/round2.py \\",
        "    --stage full --voices ru_eduard,ru_bogdan",
        "```",
        "",
        "Это даст полные десять реплик, вариант с SSML и отдельную пробу на",
        "омографах.",
        "",
    ]
    (OUT / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")


def write_index_full(voices):
    lines = [
        "# Второй раунд: финалисты на полном наборе",
        "",
        "**Оценку на слух делает человек.**",
        "",
        "## Папки",
        "",
        "| папка | что |",
        "|---|---|",
    ]
    for v in voices:
        lines.append(f"| `full/v5_{v}_plain` | v5, голос {v}, как есть |")
        lines.append(f"| `full/v5_{v}_ssml` | то же с паузами и замедлением на вводных |")
    lines += [
        "| `full/v4_eugene_plain` | текущий голос, для сравнения |",
        "| `omographs/` | «замок/замок» и «большая»: v4 против v5 с разрешением и без |",
        "",
        "## Как размечен SSML",
        "",
        "Правилами, а не вручную: после `.!?` пауза 300 мс, после `,:;` и тире 75 мс,",
        "вводные слова замедляются до 0.8 темпа.",
        "",
    ]
    (OUT / "INDEX_FULL.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
