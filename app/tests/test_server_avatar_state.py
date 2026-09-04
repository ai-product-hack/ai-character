"""Регрессии протокола состояний лица между заполнителем и основной речью."""
import pathlib
import sys
import time
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

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
