"""WER and term-recall for Russian with embedded English terms."""
import re, unicodedata

ONES = "ноль один два три четыре пять шесть семь восемь девять".split()
TEEN = ("десять одиннадцать двенадцать тринадцать четырнадцать пятнадцать "
        "шестнадцать семнадцать восемнадцать девятнадцать").split()
TENS = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят",
        "семьдесят", "восемьдесят", "девяносто"]
HUND = ["", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот",
        "семьсот", "восемьсот", "девятьсот"]


def _ru_num(n: int) -> str:
    """Small integer -> Russian words. ASR writes '99', references write it out;
    without this the WER number measures orthography, not recognition."""
    if n < 0 or n > 999:
        return str(n)
    out = []
    if n >= 100:
        out.append(HUND[n // 100]); n %= 100
    if 10 <= n < 20:
        out.append(TEEN[n - 10]); n = 0
    elif n >= 20:
        out.append(TENS[n // 10]); n %= 10
    if n or not out:
        out.append(ONES[n])
    return " ".join(out)


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower().replace("ё", "е")
    s = s.replace("—", " ").replace("–", " ").replace("-", " ")
    s = re.sub(r"\d+", lambda m: _ru_num(int(m.group())), s)
    s = re.sub(r"[^\w\s+/]", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def wer(ref: str, hyp: str) -> tuple[float, dict]:
    r, h = normalize(ref).split(), normalize(hyp).split()
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1,
                          d[i-1][j-1] + (r[i-1] != h[j-1]))
    e = d[len(r)][len(h)]
    return (e / max(1, len(r))), {"errors": e, "ref_words": len(r), "hyp_words": len(h)}


# An ASR may render an English term in Cyrillic. That is a usable outcome for a
# report but a miss for a scenario matcher, so both are counted separately.
VARIANTS = {
    "k8s": ["k8s", "кубер", "k8", "к8с", "kubernetes", "кубернетес", "кубернетис"],
    "system design": ["system design", "систем дизайн", "систем дезайн", "system-design"],
    "latency": ["latency", "лейтенси", "латенси", "лэйтенси"],
    "оффер": ["оффер", "офер", "offer"],
    "CI/CD": ["ci/cd", "ci cd", "сиай сиди", "си ай си ди"],
    "trade-off": ["trade off", "тредофф", "трейд офф", "тред офф"],
    "code review": ["code review", "код ревью", "коде ревиев"],
    "backlog": ["backlog", "бэклог", "беклог"],
    "N+1": ["n+1", "n plus one", "н плюс один", "n +1", "эн плюс один"],
    "feedback": ["feedback", "фидбек", "фидбэк"],
    "grade": ["grade", "грейд"],
    "p99": ["p99", "p девяносто девять", "девяносто девятом"],
}


def term_hits(hyp: str, terms: list[str]) -> dict:
    n = normalize(hyp)
    res = {}
    for t in terms:
        cands = VARIANTS.get(t, [t])
        exact = normalize(t) in n
        loose = any(normalize(c) in n for c in cands)
        res[t] = "exact" if exact else ("variant" if loose else "miss")
    return res
