#!/usr/bin/env bash
# Generates the agent-speech test signal used as the echo source.
# Uses macOS `say` (Milena, ru_RU) so the bench needs no API keys.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p assets
TXT="Расскажите, пожалуйста, о вашем последнем проекте. Какую роль вы там играли, \
и какие технические решения принимали лично вы? Меня интересуют детали: \
архитектура, нагрузка, и то, как вы измеряли результат."
if command -v say >/dev/null 2>&1; then
  say -v Milena -r 180 -o assets/agent_ru.aiff "$TXT"
  ffmpeg -y -loglevel error -i assets/agent_ru.aiff -ac 1 -ar 48000 assets/agent_ru.wav
  rm -f assets/agent_ru.aiff
  echo "assets/agent_ru.wav written ($(ffprobe -v error -show_entries format=duration -of csv=p=0 assets/agent_ru.wav)s)"
else
  echo "no macOS 'say'; provide assets/agent_ru.wav (mono 48k) yourself" >&2
  exit 1
fi
