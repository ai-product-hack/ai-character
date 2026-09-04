"""Переключатель синтезаторов и офлайновый запасной.

Настоящие модели не поднимаем: проверяем логику выбора и, главное, то, что
падение сетевого синтеза не роняет реплику.
"""
import pathlib
import sys
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.media import (FallbackTTS, SwitchableTTS, build_tts,  # noqa: E402
                       provider_config)


class FakeTTS:
    def __init__(self, name, sr=24000, fail=False):
        self.name = name
        self.sr = sr
        self.fail = fail
        self.calls = []

    def __call__(self, text):
        self.calls.append(text)
        if self.fail:
            raise RuntimeError("сеть недоступна")
        return [0.0] * 100, self.sr

    def describe(self):
        return {"name": self.name}

    def synthesize_many(self, texts):
        return [self(text) for text in texts]


class Fallback(unittest.TestCase):
    def test_silent_while_primary_works(self):
        primary, backup = FakeTTS("net"), FakeTTS("local")
        tts = FallbackTTS(primary, backup)
        for _ in range(3):
            tts("реплика")
        self.assertFalse(tts.failed_over)
        self.assertEqual(backup.calls, [])
        self.assertEqual(tts.describe()["active"]["name"], "net")

    def test_survives_network_failure(self):
        primary, backup = FakeTTS("net", fail=True), FakeTTS("local")
        tts = FallbackTTS(primary, backup)
        pcm, sr = tts("реплика")
        self.assertTrue(pcm)                     # реплика не потеряна
        self.assertEqual(sr, 24000)
        self.assertTrue(tts.failed_over)
        self.assertEqual(backup.calls, ["реплика"])

    def test_batch_survives_network_failure_once(self):
        primary, backup = FakeTTS("net", fail=True), FakeTTS("local")
        tts = FallbackTTS(primary, backup)
        out = tts.synthesize_many(["раз", "два"])
        self.assertEqual(len(out), 2)
        self.assertTrue(tts.failed_over)
        self.assertEqual(backup.calls, ["раз", "два"])
        self.assertEqual(len(tts.failures), 1)

    def test_does_not_flap_back(self):
        """Голос не должен меняться посреди диалога, даже если сеть вернулась."""
        primary, backup = FakeTTS("net", fail=True), FakeTTS("local")
        tts = FallbackTTS(primary, backup)
        tts("первая")
        primary.fail = False                     # сеть «починилась»
        tts("вторая")
        self.assertEqual(primary.calls, ["первая"])   # к сетевому больше не ходим
        self.assertEqual(backup.calls, ["первая", "вторая"])
        self.assertEqual(len(tts.failures), 1)

    def test_sample_rate_follows_active(self):
        tts = FallbackTTS(FakeTTS("net", sr=16000, fail=True), FakeTTS("local", sr=24000))
        self.assertEqual(tts.sr, 16000)
        tts("реплика")
        self.assertEqual(tts.sr, 24000)


class Selection(unittest.TestCase):
    def test_unknown_provider_is_loud(self):
        with self.assertRaises(SystemExit) as cm:
            build_tts({"provider": "выдуманный"})
        self.assertIn("выдуманный", str(cm.exception))

    def test_provider_specific_voices_do_not_leak(self):
        cfg = {
            "provider": "silero", "sample_rate": 24000,
            "silero": {"version": "v5_cis_base", "voice": "ru_roman"},
            "elevenlabs": {"model": "eleven_flash_v2_5", "voice": "voice-id"},
        }
        self.assertEqual(provider_config(cfg, "silero")["voice"], "ru_roman")
        eleven = provider_config(cfg, "elevenlabs")
        self.assertEqual(eleven["voice"], "voice-id")
        self.assertEqual(eleven["voice_offline"], "ru_roman")

    def test_top_level_voice_is_an_explicit_override(self):
        cfg = {"provider": "silero", "voice": "ru_override",
               "silero": {"voice": "ru_roman"}}
        self.assertEqual(provider_config(cfg)["voice"], "ru_override")

    @patch("app.media.SileroTTS")
    def test_build_tts_understands_nested_app_config(self, silero):
        silero.return_value = FakeTTS("silero")
        build_tts({"provider": "silero",
                   "silero": {"version": "v5_cis_base", "voice": "ru_roman"}})
        silero.assert_called_once_with(voice="ru_roman", version="v5_cis_base",
                                       sample_rate=24000)

    @patch("app.media.build_tts")
    def test_switch_uses_each_providers_own_voice_and_caches(self, build):
        build.side_effect = lambda cfg, **kw: FakeTTS(cfg["provider"])
        tts = SwitchableTTS({
            "provider": "silero",
            "silero": {"voice": "ru_roman"},
            "elevenlabs": {"voice": "voice-id"},
        })
        self.assertEqual(build.call_args.args[0]["voice"], "ru_roman")
        self.assertTrue(tts.switch("elevenlabs"))
        self.assertEqual(build.call_args.args[0]["voice"], "voice-id")
        self.assertTrue(tts.switch("silero"))
        self.assertEqual(build.call_count, 2)  # Silero взят из кеша

    @patch("app.media.build_tts")
    def test_failed_switch_keeps_working_provider(self, build):
        def factory(cfg, **kw):
            if cfg["provider"] == "elevenlabs":
                raise SystemExit("нет ELEVENLABS_API_KEY")
            return FakeTTS("silero")
        build.side_effect = factory
        tts = SwitchableTTS({"provider": "silero"})
        with self.assertRaises(SystemExit):
            tts.switch("elevenlabs")
        self.assertEqual(tts.provider, "silero")
        self.assertEqual(tts.engine.name, "silero")

    @patch("app.media.build_tts")
    def test_provider_labels_make_silero_voice_explicit(self, build):
        build.return_value = FakeTTS("silero")
        tts = SwitchableTTS({"provider": "silero",
                             "silero": {"voice": "ru_roman"}})
        self.assertEqual(tts.provider_options(), [
            {"provider": "silero", "label": "Silero (ru_roman)"},
            {"provider": "elevenlabs", "label": "ElevenLabs"},
        ])

    @patch("app.media.ElevenLabsTTS")
    @patch("app.media.SileroTTS")
    def test_elevenlabs_reuses_loaded_silero_as_fallback(self, silero, eleven):
        local, remote = FakeTTS("silero"), FakeTTS("elevenlabs")
        silero.return_value, eleven.return_value = local, remote
        tts = SwitchableTTS({"provider": "silero",
                             "silero": {"voice": "ru_roman"},
                             "elevenlabs": {"voice": "voice-id"}})
        tts.switch("elevenlabs")
        self.assertIs(tts.engine.backup, local)
        self.assertEqual(silero.call_count, 1)


if __name__ == "__main__":
    unittest.main()
