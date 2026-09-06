"""Регрессии протокола состояний лица между заполнителем и основной речью."""
import pathlib
import sys
import time
import types
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.evaluator import EvaluationLog                       # noqa: E402
from app.scenario import Persona                              # noqa: E402
from app.pipeline import ClauseResult                         # noqa: E402
from app.server import Session                               # noqa: E402


def clause(index=0):
    return ClauseResult(
        generation_id="gen-eye-contact", index=index, text="Продолжим.",
        first=index == 0, start_ms=index * 500, audio_ms=400,
        pcm=np.zeros(400, dtype=np.float32), sample_rate=24000,
        visemes=[], chars=[], emotions=[],
    )


class SpeakingAfterBackchannel(unittest.TestCase):
    def test_first_content_clause_returns_face_from_thinking_to_speaking(self):
        session = Session.__new__(Session)
        session.bc_cfg = {}
        session.spoken = {}          # обычно ставится в __init__
        # Настроение по оценкам берётся на первой клаузе — без этих двух
        # полей проверка состояния лица падала бы на постороннем.
        session.evaluator = types.SimpleNamespace(log=EvaluationLog())
        # Персона нужна ради стартовой эмоции: без оценок настроение берётся
        # из неё, и лицо не остаётся ровным с первой реплики.
        session.scenario = types.SimpleNamespace(
            criteria=[], persona=Persona(role="проверяющий"))
        frames = []
        session._emit = lambda header, payload=b"": frames.append((header, payload))
        em = {
            "offset_ms": 700, "clause_base": 1, "used_bc": True,
            "t_first_audio": None, "t_first_speech": None, "_placed": True,
        }

        session._emit_clause(None, clause(), em, time.perf_counter())

        self.assertEqual(frames[0][0], {
            "kind": "state", "generation_id": "gen-eye-contact",
            "face": "speaking",
        })
        self.assertEqual(frames[1][0]["kind"], "audio")

        frames.clear()
        session._emit_clause(None, clause(1), em, time.perf_counter())
        self.assertNotIn("state", [frame[0]["kind"] for frame in frames],
                         "возврат взгляда нужен один раз, а не перед каждой клаузой")


if __name__ == "__main__":
    unittest.main()


class EmotionReset(unittest.TestCase):
    def test_unmarked_neutral_clause_clears_previous_expression_at_audio_start(self):
        session = Session.__new__(Session)
        session.bc_cfg = {}; session.spoken = {}
        session.evaluator = types.SimpleNamespace(log=EvaluationLog())
        session.scenario = types.SimpleNamespace(criteria=[], persona=Persona(role='проверяющий'))
        frames = []
        session._emit = lambda header, payload=b'': frames.append(header)
        em = {'offset_ms':700, 'clause_base':1, 'used_bc':True,
              't_first_audio':None, 't_first_speech':None, '_placed':True}
        result = clause(); result.start_ms = 120
        session._emit_clause(None, result, em, time.perf_counter())
        emotions = [f for f in frames if f['kind'] == 'emotions']
        self.assertEqual(len(emotions), 1)
        self.assertEqual(emotions[0]['marks'], [
            {'pts_ms':820, 'emotion':'neutral', 'intensity':0.0}])

    def test_explicit_semantic_tag_takes_priority_over_score_fallback(self):
        session = Session.__new__(Session)
        session.bc_cfg = {}; session.spoken = {}
        frames = []
        session._emit = lambda header, payload=b'': frames.append(header)
        em = {'offset_ms':0, 'clause_base':0, 'used_bc':False,
              't_first_audio':None, 't_first_speech':None, '_placed':True}
        result = clause(); result.emotions = [{'pts_ms':0, 'emotion':'anxious', 'intensity':0.8}]
        session._emit_clause(None, result, em, time.perf_counter())
        emotions = [f for f in frames if f['kind'] == 'emotions']
        self.assertEqual(len(emotions), 1)
        self.assertEqual(emotions[0]['marks'][0]['emotion'], 'anxious')


class VoiceActivity(unittest.TestCase):
    def test_only_new_vad_speech_counts_as_listening_activity(self):
        session = Session.__new__(Session)
        endpointer = types.SimpleNamespace(speech_ms=100)
        def push(pcm):
            endpointer.speech_ms += float(pcm[0])
            return None
        session.voice = types.SimpleNamespace(endpointer=endpointer, push=push)
        self.assertFalse(session.push_audio([0])['user_speaking'])
        self.assertTrue(session.push_audio([32])['user_speaking'])
        self.assertFalse(session.push_audio([0])['user_speaking'])

    def test_no_vad_does_not_invent_user_activity(self):
        session = Session.__new__(Session)
        session.voice = types.SimpleNamespace(endpointer=None, push=lambda pcm: None)
        self.assertFalse(session.push_audio([1])['user_speaking'])
