#!/usr/bin/env python3
"""Сравнительные сэмплы TTS для прослушивания человеком.

Синтезирует 10 полноценных реплик агента (2-3 предложения, с вопросами и
перечислениями) во всех вариантах, между которыми надо выбирать:

    голоса        5 голосов Silero v4_ru при прочих равных
    ударения      без простановки / автопростановка / ручная разметка «+»
    SSML          паузы на знаках препинания и замедление на вводных

Реплики взяты из настоящих сценариев (data/scenarios), а не выдуманы: короткие
изолированные фразы из R-фазы не показывают, как голос ведёт себя на реальной
длине, а именно она и будет звучать в продукте.

Суждений о качестве скрипт не выносит — это делает человек на слух.

    bench/r1-stt/.venv/bin/python bench/r3-tts/compare_voices.py
"""
import json, pathlib, re, sys, time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "bench" / "r3-tts" / "compare"
RESULTS = ROOT / "bench" / "results"
SR = 24000          # для оценки голоса на слух хватает; 48 кГц раздувает папку вдвое
REF_VOICE = "baya"  # опорный голос для сравнений по ударениям и SSML

# Реплики агента: по две на каждый из пяти сценариев. Ручная разметка ударений
# сделана только там, где автопростановка ошибается чаще всего: омографы,
# редкие слова, заимствования.
REPLIES = [
    {
        "id": "r01", "scenario": "s1_interview_backend", "stage": "глубина роли",
        "text": "Спасибо, картина в целом понятна. Вы всё время говорите «мы»: "
                "команда собрала, команда выкатила. Расскажите, что именно делали "
                "лично вы — какой кусок держали и какие решения принимали сами?",
        "accented": "Спас+ибо, карт+ина в ц+елом пон+ятна. Вы всё вр+емя говор+ите «мы»: "
                    "ком+анда собрал+а, ком+анда в+ыкатила. Расскаж+ите, что +именно д+елали "
                    "л+ично вы — как+ой кус+ок держ+али и как+ие реш+ения приним+али с+ами?",
    },
    {
        "id": "r02", "scenario": "s1_interview_backend", "stage": "цифры",
        "text": "Хорошо, с архитектурой разобрались. Теперь цифры: какая была "
                "нагрузка в запросах в секунду, какую задержку вы держали на "
                "девяносто пятом перцентиле и сколько инстансов крутилось в проде? "
                "Только конкретные значения, «много» и «быстро» не подойдут.",
        "accented": "Хорош+о, с архитект+урой разобр+ались. Теп+ерь ц+ифры: как+ая был+а "
                    "нагр+узка в запр+осах в сек+унду, как+ую зад+ержку вы держ+али на "
                    "девян+осто п+ятом перцент+иле и ск+олько инст+ансов крут+илось в пр+оде? "
                    "Т+олько конкр+етные знач+ения, «мн+ого» и «б+ыстро» не подойд+ут.",
    },
    {
        "id": "r03", "scenario": "s2_interview_stress", "stage": "давление",
        "text": "Понял вас. Названная вилка выше нашей примерно на двадцать процентов, "
                "и бюджет на позицию уже утверждён. Что скажете — есть пространство "
                "для манёвра или это жёсткая нижняя граница?",
        "accented": "П+онял вас. Н+азванная в+илка в+ыше н+ашей примерно на дв+адцать проц+ентов, "
                    "и бюдж+ет на поз+ицию уж+е утверждён. Что ск+ажете — есть простр+анство "
                    "для манёвра +или это жёсткая н+ижняя гран+ица?",
    },
    {
        "id": "r04", "scenario": "s2_interview_stress", "stage": "подмена",
        "text": "Давайте посмотрим шире, чем на оклад. Мы можем добавить обучение за "
                "счёт компании, расширенный ДМС со стоматологией и пересмотр через "
                "полгода. Что из этого для вас действительно ценно, а что не считается?",
        "accented": "Дав+айте посм+отрим ш+ире, чем на окл+ад. Мы м+ожем доб+авить обуч+ение за "
                    "счёт комп+ании, расш+иренный ДМС со стоматол+огией и пересм+отр через "
                    "полг+ода. Что из +этого для вас действ+ительно ц+енно, а что не счит+ается?",
    },
    {
        "id": "r05", "scenario": "s3_sales_cold", "stage": "первые 10 секунд",
        "text": "Так, у меня ровно минута до совещания. Вы кто, откуда и что "
                "предлагаете? Только без презентаций, пожалуйста — суть в двух предложениях.",
        "accented": "Так, у мен+я р+овно мин+ута до совещ+ания. Вы кто, отк+уда и что "
                    "предлаг+аете? Т+олько без презент+аций, пож+алуйста — суть в двух предлож+ениях.",
    },
    {
        "id": "r06", "scenario": "s3_sales_cold", "stage": "цена",
        "text": "Хорошо, допустим. Сколько это стоит — назовите цифру, а не «зависит "
                "от конфигурации». И сразу скажите, что входит в эту сумму, а за что "
                "придётся доплачивать отдельно.",
        "accented": "Хорош+о, доп+устим. Ск+олько +это ст+оит — назов+ите ц+ифру, а не «зав+исит "
                    "от конфигур+ации». И ср+азу скаж+ите, что вх+одит в +эту с+умму, а за что "
                    "придётся допл+ачивать отд+ельно.",
    },
    {
        "id": "r07", "scenario": "s4_sales_objections", "stage": "вход",
        "text": "Мы посмотрели демо и сравнили с тем, что показывали до вас. Честно "
                "говоря, у конкурента выходит дешевле примерно на треть. Чем вы "
                "объясняете разницу?",
        "accented": "Мы посмотр+ели д+емо и сравн+или с тем, что пок+азывали до вас. Ч+естно "
                    "говор+я, у конкур+ента вых+одит деш+евле примерно на треть. Чем вы "
                    "объясн+яете р+азницу?",
    },
    {
        "id": "r08", "scenario": "s4_sales_objections", "stage": "риск внедрения",
        "text": "Допустим, с ценой разобрались. Но у нас команда из четырёх человек, и "
                "они уже тянут CRM, телефонию и складской учёт. Кто будет внедрять, "
                "сколько это займёт по времени и что мы делаем, если человек, который "
                "всё настроил, уйдёт?",
        "accented": "Доп+устим, с цен+ой разобр+ались. Но у нас ком+анда из четырёх челов+ек, и "
                    "он+и уж+е т+янут CRM, телеф+онию и складск+ой учёт. Кто б+удет внедр+ять, "
                    "ск+олько +это займёт по вр+емени и что мы д+елаем, +если челов+ек, кот+орый "
                    "всё настр+оил, уйдёт?",
    },
    {
        "id": "r09", "scenario": "s5_knowledge_check", "stage": "база",
        "text": "Начнём с базы. Что по нашему регламенту относится к персональным "
                "данным — перечислите категории: что считается, что не считается и где "
                "проходит граница. Отвечайте своими словами, цитировать документ не нужно.",
        "accented": "Начнём с б+азы. Что по н+ашему реглам+енту отн+осится к персон+альным "
                    "д+анным — перечисл+ите катег+ории: что счит+ается, что не счит+ается и где "
                    "прох+одит гран+ица. Отвеч+айте сво+ими слов+ами, цит+ировать докум+ент не н+ужно.",
    },
    {
        "id": "r10", "scenario": "s5_knowledge_check", "stage": "процедура",
        "text": "Хорошо. Теперь ситуация: клиент письменно просит удалить все свои "
                "данные. Опишите ваши действия по шагам — кого уведомляете, что "
                "проверяете перед удалением и в какой срок обязаны ответить?",
        "accented": "Хорош+о. Теп+ерь ситу+ация: кли+ент п+исьменно пр+осит удал+ить все сво+и "
                    "д+анные. Опиш+ите в+аши д+ействия по шаг+ам — ког+о увед+омляете, что "
                    "провер+яете п+еред удал+ением и в как+ой срок об+язаны отв+етить?",
    },
]

# Вводные, на которых стоит замедлиться: они несут не смысл, а интонацию.
PARENTHETICALS = ["Честно говоря", "честно говоря", "пожалуйста", "Допустим", "допустим",
                  "Хорошо", "хорошо", "Так", "Понял вас", "Теперь", "теперь"]


def to_ssml(text):
    """Текст -> SSML: паузы на знаках препинания, замедление на вводных.

    Ровно то, что делал бы продукт: правила, а не ручная разметка каждой
    реплики. Длительности пауз — из таблицы Silero: weak 75 мс, medium 150 мс,
    strong 300 мс.
    """
    s = text
    for p in PARENTHETICALS:
        s = s.replace(p, f'<prosody rate="slow">{p}</prosody>', 1)
    # Пауза ПОСЛЕ знака, поэтому вставляем следом за ним.
    s = re.sub(r"([.!?])(\s+)", r'\1<break strength="strong"/>\2', s)
    s = re.sub(r"([,:;—])(\s+)", r'\1<break strength="weak"/>\2', s)
    return f"<speak>{s}</speak>"


def write_wav(path, audio, sr=SR):
    """WAV 16 бит моно. Стдлиб, без soundfile."""
    import struct, wave
    pcm = np.clip(np.asarray(audio, dtype=np.float32), -1, 1)
    pcm = (pcm * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def first_clause(text):
    """Где потоковый конвейер отрезал бы первый кусок: до первой границы
    предложения или ~50 символов, что раньше."""
    m = re.search(r"[.!?]", text[:120])
    end = m.end() if m and m.end() <= 50 else None
    if end is None:
        m2 = re.search(r"[,:;—]", text[:60])
        end = m2.end() if m2 else min(50, len(text))
    return text[:end].strip()


def check_alphabets():
    """Слово не должно смешивать кириллицу и латиницу.

    Ловит опечатку, которая иначе тихо портит сравнение: латинская «e» внутри
    русского слова не читается синтезатором как «е». Найдено на первом же
    прогоне — «ужe» в двух репликах из ручной разметки ударений.
    """
    lat, cyr = re.compile(r"[a-zA-Z]"), re.compile(r"[а-яёА-ЯЁ]")
    bad = [(rep["id"], field, w)
           for rep in REPLIES for field in ("text", "accented")
           for w in re.findall(r"\S+", rep[field])
           if lat.search(w) and cyr.search(w)]
    if bad:
        for rid, field, w in bad:
            print(f"  !! {rid}.{field}: слово «{w}» смешивает алфавиты")
        raise SystemExit("реплики не прошли проверку алфавитов")


def main():
    import torch
    check_alphabets()
    torch.set_num_threads(4)
    t0 = time.perf_counter()
    model, _ = torch.hub.load("snakers4/silero-models", "silero_tts",
                              language="ru", speaker="v4_ru", trust_repo=True)
    model.to(torch.device("cpu"))
    load_s = time.perf_counter() - t0
    voices = [v for v in model.speakers if v != "random"]
    print(f"модель загружена за {load_s:.1f} с | голоса: {', '.join(voices)}")

    # Прогрев: замерено в R3, первые ДВА вызова стоят ~520 мс каждый.
    for t in ("Прогрев.", "Ещё один прогрев, подлиннее."):
        model.apply_tts(text=t, speaker=REF_VOICE, sample_rate=SR)

    # Варианты: (папка, как синтезировать)
    variants = []
    for v in voices:
        variants.append((f"voice_{v}", dict(kind="text", speaker=v, accent=True)))
    variants += [
        ("accent_off",    dict(kind="text", speaker=REF_VOICE, accent=False)),
        ("accent_auto",   dict(kind="text", speaker=REF_VOICE, accent=True)),
        ("accent_manual", dict(kind="accented", speaker=REF_VOICE, accent=True)),
        ("ssml_marked",   dict(kind="ssml", speaker=REF_VOICE, accent=True)),
    ]

    OUT.mkdir(parents=True, exist_ok=True)
    timings = []
    for folder, how in variants:
        d = OUT / folder
        d.mkdir(exist_ok=True)
        for rep in REPLIES:
            if how["kind"] == "ssml":
                payload = dict(ssml_text=to_ssml(rep["text"]))
            elif how["kind"] == "accented":
                payload = dict(text=rep["accented"])
            else:
                payload = dict(text=rep["text"])
            t1 = time.perf_counter()
            try:
                au = model.apply_tts(speaker=how["speaker"], sample_rate=SR,
                                     put_accent=how["accent"], put_yo=how["accent"],
                                     **payload)
            except Exception as e:
                print(f"  !! {folder}/{rep['id']}: {type(e).__name__}: {e}")
                continue
            ms = (time.perf_counter() - t1) * 1000
            au = np.asarray(au, dtype=np.float32)
            write_wav(d / f"{rep['id']}.wav", au)
            timings.append({
                "variant": folder, "reply_id": rep["id"], "scenario": rep["scenario"],
                "chars": len(rep["text"]), "synth_ms": round(ms),
                "audio_s": round(len(au) / SR, 3),
                "rtf": round(ms / 1000 / (len(au) / SR), 4),
            })
        print(f"  {folder}: {len(REPLIES)} файлов")

    # TTFB на РЕАЛЬНОЙ длине: отдельно первый кусок и реплика целиком.
    # Прежние 9 мс сняты на первой клаузе короткой фразы; вопрос в том, что
    # происходит, когда реплика длинная.
    ttfb = []
    for rep in REPLIES:
        fc = first_clause(rep["text"])
        t1 = time.perf_counter()
        au_first = model.apply_tts(text=fc, speaker=REF_VOICE, sample_rate=SR)
        ms_first = (time.perf_counter() - t1) * 1000
        t1 = time.perf_counter()
        au_full = model.apply_tts(text=rep["text"], speaker=REF_VOICE, sample_rate=SR)
        ms_full = (time.perf_counter() - t1) * 1000
        ttfb.append({
            "reply_id": rep["id"], "scenario": rep["scenario"],
            "chars_total": len(rep["text"]), "chars_first_clause": len(fc),
            "first_clause_text": fc,
            "ttfb_first_clause_ms": round(ms_first),
            "first_clause_audio_s": round(len(au_first) / SR, 3),
            "synth_full_ms": round(ms_full),
            "full_audio_s": round(len(au_full) / SR, 3),
            "rtf_full": round(ms_full / 1000 / (len(au_full) / SR), 4),
        })

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "r3_tts_compare_timings.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in timings) + "\n", encoding="utf-8")
    (RESULTS / "r3_tts_ttfb_long.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in ttfb) + "\n", encoding="utf-8")

    # Побитовая сверка ручной разметки с автопростановкой. Это измерение, а не
    # суждение о качестве: если файлы совпали, Silero расставила ударение там же,
    # где и я, и ручная разметка на этой реплике ничего не даёт.
    import hashlib
    accents = []
    for rep in REPLIES:
        h = {}
        for folder in ("accent_auto", "accent_manual"):
            f = OUT / folder / f"{rep['id']}.wav"
            h[folder] = hashlib.sha1(f.read_bytes()).hexdigest() if f.exists() else None
        accents.append({"reply_id": rep["id"],
                        "manual_changes_audio": h["accent_auto"] != h["accent_manual"]})
    changed = [a["reply_id"] for a in accents if a["manual_changes_audio"]]

    write_index(voices, ttfb, accents)
    print(f"\nручная разметка ударений меняет звук на {len(changed)} репликах "
          f"из {len(REPLIES)}: {', '.join(changed) or '—'}")
    fc_ms = sorted(r["ttfb_first_clause_ms"] for r in ttfb)
    full_ms = sorted(r["synth_full_ms"] for r in ttfb)
    print(f"\nTTFB первой клаузы: медиана {fc_ms[len(fc_ms)//2]} мс, макс {fc_ms[-1]} мс")
    print(f"синтез реплики целиком: медиана {full_ms[len(full_ms)//2]} мс, макс {full_ms[-1]} мс")
    print(f"-> {OUT}")


def write_index(voices, ttfb, accents):
    lines = [
        "# Сравнительные сэмплы TTS",
        "",
        "Материал для решения по голосу. **Оценку на слух делает человек** — здесь",
        "только исходные данные и условия синтеза, без суждений о качестве.",
        "",
        f"Движок: Silero v4_ru (`TTSModelMultiAcc_v3`), {SR} Гц, 16 бит моно.",
        f"Опорный голос для сравнений по ударениям и SSML: **{REF_VOICE}**.",
        "",
        "Собрано скриптом `bench/r3-tts/compare_voices.py`. Сами WAV в репозиторий",
        "не кладутся (около 40 МБ) — скрипт пересобирает их за пару минут.",
        "",
        "## Что с чем сравнивать",
        "",
        "| папка | что меняется | прочее |",
        "|---|---|---|",
    ]
    for v in voices:
        lines.append(f"| `voice_{v}` | голос **{v}** | автопростановка ударений включена |")
    lines += [
        f"| `accent_off` | `put_accent=False`, `put_yo=False` | голос {REF_VOICE} |",
        f"| `accent_auto` | автопростановка (умолчание Silero) | голос {REF_VOICE}, "
        f"совпадает с `voice_{REF_VOICE}` |",
        f"| `accent_manual` | ручная разметка `+` поверх автопростановки | голос {REF_VOICE} |",
        f"| `ssml_marked` | паузы на знаках, замедление на вводных | голос {REF_VOICE} |",
        "",
        "`accent_auto` намеренно дублирует `voice_" + REF_VOICE + "`: три варианта по",
        "ударениям должны лежать рядом, иначе сравнивать неудобно.",
        "",
        "## Как размечен SSML",
        "",
        "Правилами, а не вручную по каждой реплике — ровно то, что делал бы продукт:",
        "",
        "- после `.`, `!`, `?` — `<break strength=\"strong\"/>`, это 300 мс;",
        "- после `,`, `:`, `;`, тире — `<break strength=\"weak\"/>`, это 75 мс;",
        "- вводные слова обёрнуты в `<prosody rate=\"slow\">`, это 0.8 от темпа.",
        "",
        "## Одно измерение, которое стоит знать до прослушивания",
        "",
        "Файлы `accent_auto` и `accent_manual` сверены побитово. Там, где они",
        "совпали, автопростановка Silero поставила ударение туда же, куда и рука, —",
        "то есть ручная разметка на этой реплике не даёт ничего.",
        "",
        "| реплика | ручная разметка меняет звук |",
        "|---|---|",
    ] + [
        f"| `{a['reply_id']}` | {'да' if a['manual_changes_audio'] else 'нет'} |"
        for a in accents
    ] + [
        "",
        f"Итого: **{sum(a['manual_changes_audio'] for a in accents)} из {len(accents)}**.",
        "Это измерение, а не оценка: совпадение означает согласие с моей разметкой,",
        "а не то, что ударение поставлено верно. Слушать всё равно нужно.",
        "",
        "## Реплики",
        "",
        "Все десять взяты из настоящих сценариев (`data/scenarios`) и представляют",
        "реальную длину ответа агента: два-три предложения, вопросы, перечисления.",
        "Изолированные короткие фразы из R-фазы для оценки голоса не годятся —",
        "на них не слышно ни просодии перечисления, ни поведения на длинной паузе.",
        "",
    ]
    by_id = {t["reply_id"]: t for t in ttfb}
    for rep in REPLIES:
        t = by_id.get(rep["id"], {})
        lines += [
            f"### `{rep['id']}` — {rep['scenario']}, этап «{rep['stage']}»",
            "",
            f"> {rep['text']}",
            "",
            f"Длина {len(rep['text'])} символов, звучание "
            f"{t.get('full_audio_s', '?')} с.",
            "",
            "<details><summary>с ручной разметкой ударений</summary>",
            "",
            f"> {rep['accented']}",
            "",
            "</details>",
            "",
        ]
    (OUT / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
