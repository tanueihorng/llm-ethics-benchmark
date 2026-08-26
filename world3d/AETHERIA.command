#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
#  AETHERIA · The Floating Isles — double-click to play.
#  Starts a local server for the built world and opens your browser.
#  (The app is an ES-module site, so index.html can't be opened directly
#  from Finder — it needs a tiny local server, which this provides.)
# ─────────────────────────────────────────────────────────────────────
set -e
cd "$(dirname "$0")"
PORT=5199
URL="http://127.0.0.1:$PORT"

# Already running? Just open it.
if curl -s -o /dev/null --max-time 1 "$URL/"; then
  open "$URL/"
  echo "AETHERIA was already running — opened $URL"
  exit 0
fi

# First run: install dependencies (workspace-local npm cache) and build.
if [ ! -d node_modules ]; then
  echo "First run — installing dependencies…"
  npm install --no-audit --no-fund --cache "$PWD/.npm-cache" --loglevel=error
fi
if [ ! -f dist/index.html ]; then
  echo "Building the world…"
  node node_modules/vite/bin/vite.js build
fi

echo "Starting AETHERIA at $URL  (close this window to stop it)"
node node_modules/vite/bin/vite.js preview --host 127.0.0.1 --port $PORT --strictPort &
SERVER_PID=$!

for _ in $(seq 1 40); do
  curl -s -o /dev/null --max-time 1 "$URL/" && break
  sleep 0.25
done
open "$URL/"

wait $SERVER_PID
