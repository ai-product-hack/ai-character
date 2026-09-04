#!/usr/bin/env python3
"""Сквозной прогон конвейера реплики на настоящих моделях.

    python3 app/run_pipeline.py --scenario s1_interview_backend --turns 4

Проверяет то, ради чего срез и делается: складывается ли путь целиком и где
он рвётся. Меряет время до первого звука, стоимость каждой ступени и
непрерывность PTS через клаузы.
"""
import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import llm as llm_mod                              # noqa: E402
from app.agent import SYSTEM, build_prompt                  # noqa: E402
from app.actions import parse_reply                         # noqa: E402
from app.agent import apply as apply_action                 # noqa: E402
from app.dialogue import DialogueState                      # noqa: E402
from app.generation import GenerationRegistry               # noqa: E402
from app.media import GigaAMAligner, SileroTTS, stream_deepseek   # noqa: E402
from app.pipeline import ReplyPipeline, subtitle_cues       # noqa: E402
from app.scenario import load_all                           # noqa: E402
from app.visemes_bridge import VisemeBridge                 # noqa: E402
from app.run_dialogue import FALLBACK_LINES, USER_LINES     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=None, help="id; по умолчанию все пять")
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--voice", default="eugene")
    ap.add_argument("--out", default="bench/results/pipeline_all.json")
    args = ap.parse_args()

    print("поднимаю модели…")
    t0 = time.perf_counter()
    tts = SileroTTS(voice=args.voice)
    aligner = GigaAMAligner()
    bridge = VisemeBridge()
    llm = llm_mod.DeepSeekLLM()
    print(f"  Silero {tts.load_s:.1f} с (прогрев {tts.warmup_ms} мс), "
          f"GigaAM {aligner.load_s:.1f} с, всего {time.perf_counter() - t0:.1f} с")

    scenarios = load_all(ROOT / "data" / "scenarios")
    if args.scenario:
        scenarios = [s for s in scenarios if s.id == args.scenario] or scenarios

    all_runs = []
    for sc in scenarios:
        all_runs.append(run_scenario(sc, tts, aligner, bridge, llm, args))
    report(all_runs, ROOT / args.out, args.voice)
    bridge.close()


def run_scenario(sc, tts, aligner, bridge, llm, args):
    state = DialogueState(sc)
    registry = GenerationRegistry()
    pipe = ReplyPipeline(stream_deepseek(llm), tts, aligner, bridge, registry)

    lines = USER_LINES.get(sc.id, FALLBACK_LINES)
    turns = []
    print(f"\n########## {sc.id} | {sc.title}")

    for turn in range(args.turns):
        user = None if turn == 0 else lines[(turn - 1) % len(lines)]
        if user is not None:
            state.add_user(user)
            print(f"\nпольз.: {user}")
        gen = registry.start()
        prompt = build_prompt(state, user)

        t_turn = time.perf_counter()
        results = pipe.run(SYSTEM, prompt, gen)
        wall_ms = (time.perf_counter() - t_turn) * 1000

        # Действие разбирается из СЫРОГО ответа модели, а не из клауз:
        # из клауз управляющий блок вырезан перед синтезом.
        reply = parse_reply(pipe.last_raw)
        apply_action(state, reply, gen.id)

        cues = [c for r in results for c in subtitle_cues(r)]
        gaps = [round(b.start_ms - (a.start_ms + a.audio_ms), 3)
                for a, b in zip(results, results[1:])]
        rec = {
            "turn": turn, "stage": state.stage_id,
            "generation_id": gen.id, "clauses": len(results),
            "t_first_token_ms": round(pipe.last_stats["t_first_token"] or 0),
            "t_first_audio_ms": round(pipe.last_stats["t_first_audio"] or 0),
            "wall_ms": round(wall_ms),
            "audio_total_ms": round(pipe.last_timeline.total_ms),
            "visemes": sum(len(r.visemes) for r in results),
            "subtitle_cues": len(cues),
            "pts_gaps_ms": gaps,
            "action": reply.action.action,
            "per_clause": [{"i": r.index, "chars": len(r.text),
                            "audio_ms": round(r.audio_ms),
                            "start_ms": round(r.start_ms), **r.timings}
                           for r in results],
        }
        turns.append(rec)
        print(f"агент [{gen.id}]: {' '.join(r.text for r in results)[:170]}")
        print(f"  клауз {len(results)}, первый токен {rec['t_first_token_ms']} мс, "
              f"ПЕРВЫЙ ЗВУК {rec['t_first_audio_ms']} мс, вся реплика {rec['wall_ms']} мс")
        print(f"  звука {rec['audio_total_ms']} мс, висем {rec['visemes']}, "
              f"субтитров {rec['subtitle_cues']}, разрывы PTS {gaps or '—'}")
        for c in rec["per_clause"]:
            print(f"    [{c['i']}] {c['chars']:>3} симв -> {c['audio_ms']:>5} мс звука "
                  f"| TTS {c['tts_ms']:>3} мс, выравнивание {c['align_ms']:>3} мс")
        if state.finished:
            break

    first_audio = sorted(t["t_first_audio_ms"] for t in turns if t["t_first_audio_ms"])
    gaps = [g for t in turns for g in t["pts_gaps_ms"]]
    empty = [t["turn"] for t in turns if t["clauses"] == 0]
    print(f"--- {sc.id}: реплик {len(turns)}, finished={state.finished}, "
          f"этап {state.stage_index + 1}/{len(sc.stages)}, "
          f"первый звук медиана "
          f"{first_audio[len(first_audio)//2] if first_audio else '—'} мс, "
          f"пустых реплик {len(empty)}")
    return {
        "scenario_id": sc.id, "finished": state.finished,
        "stages_reached": state.stage_index + 1, "stages_total": len(sc.stages),
        "turns": turns,
        "empty_replies": empty,
        "first_audio_median_ms": (first_audio[len(first_audio) // 2]
                                  if first_audio else None),
        "first_audio_max_ms": first_audio[-1] if first_audio else None,
        "pts_gap_max_ms": max((abs(g) for g in gaps), default=0),
        "audio_total_ms": sum(t["audio_total_ms"] for t in turns),
        "visemes_total": sum(t["visemes"] for t in turns),
        "cues_total": sum(t["subtitle_cues"] for t in turns),
    }


def report(runs, out, voice):
    print("\n================ ИТОГ ================")
    fa = sorted(r["first_audio_median_ms"] for r in runs if r["first_audio_median_ms"])
    print(f"сценариев: {len(runs)}, дошли до finish: {sum(r['finished'] for r in runs)}")
    print(f"время до первого звука по сценариям: "
          f"{[r['first_audio_median_ms'] for r in runs]}")
    if fa:
        print(f"  медиана медиан {fa[len(fa)//2]} мс")
    print(f"худший разрыв PTS: {max(r['pts_gap_max_ms'] for r in runs)} мс")
    empty = sum(len(r["empty_replies"]) for r in runs)
    print(f"пустых реплик (ни одной клаузы): {empty}")
    print(f"звука всего {sum(r['audio_total_ms'] for r in runs) / 1000:.1f} с, "
          f"висем {sum(r['visemes_total'] for r in runs)}, "
          f"субтитров {sum(r['cues_total'] for r in runs)}")
    for r in runs:
        print(f"  {r['scenario_id']:<24} finished={str(r['finished']):<5} "
              f"этап {r['stages_reached']}/{r['stages_total']}, "
              f"первый звук {r['first_audio_median_ms']} мс, "
              f"разрыв PTS {r['pts_gap_max_ms']} мс")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"voice": voice, "runs": runs},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    sys.exit(main())
