#!/usr/bin/env bash
# One-command launch for the R6 echo/barge-in bench.
set -euo pipefail
cd "$(dirname "$0")"
[ -f assets/agent_ru.wav ] || ./make-assets.sh
python3 serve.py &
SRV=$!
sleep 1
open "http://localhost:${PORT:-8060}/" 2>/dev/null || true
trap 'kill $SRV 2>/dev/null || true' EXIT
wait $SRV
