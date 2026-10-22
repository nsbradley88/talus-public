#!/bin/bash
# Run serve/server.py on a Talus engine config (e.g. configs/laguna-s-2.1.example.json, filled in).
#   scripts/talus-serve.sh start CONFIG.json [PORT]   (default port: the config's "port", else 8001)
#   scripts/talus-serve.sh stop  CONFIG.json          (SIGTERM: the server QUITs the engine; waits up to 90 s)
#   scripts/talus-serve.sh status CONFIG.json
# The pid and the server log sit next to the config (<config>.pid, <config>.server.log).
set -u
cmd=${1:?start|stop|status}; cfg=$(readlink -f "${2:?config}")
root=$(cd "$(dirname "$0")/.." && pwd)
pidf="$cfg.pid"; log="$cfg.server.log"
py=${TALUS_PYTHON:-python3}
port=${3:-$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('port', 8001))" "$cfg")}
# the server's pid is the one listening on the port (setsid may fork, so $! is not reliably the server)
listener() { ss -ltnpH "sport = :$port" 2>/dev/null | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2; }
alive() {
    local p; p=$(listener)
    if [ -n "$p" ]; then echo "$p" > "$pidf"; return 0; fi
    [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null
}
case $cmd in
start)
    alive && { echo "already running (pid $(cat "$pidf"))"; exit 0; }
    (cd "$root" && setsid nohup "$py" serve/server.py --engine strata --config "$cfg" --host 127.0.0.1 --port "$port" \
        > "$log" 2>&1 < /dev/null &)
    for _ in $(seq 1 180); do
        grep -qE '^ready' "$log" 2>/dev/null && { alive; grep -E '^ready' "$log"; exit 0; }
        grep -qE 'error:|Traceback' "$log" 2>/dev/null && { echo "server failed:"; tail -5 "$log"; exit 1; }
        sleep 5
    done
    echo "not ready after 15 min; see $log"; exit 1 ;;
stop)
    alive || { echo "not running"; exit 0; }
    pid=$(cat "$pidf"); kill -TERM "$pid"
    for _ in $(seq 1 90); do kill -0 "$pid" 2>/dev/null || { rm -f "$pidf"; echo stopped; exit 0; }; sleep 1; done
    echo "still running after 90 s (pid $pid)"; exit 1 ;;
status)
    if alive; then echo "running (pid $(cat "$pidf"))"; curl -s "localhost:$port/health"; echo; else echo "not running"; fi ;;
*) echo "usage: $0 start|stop|status CONFIG.json [PORT]"; exit 2 ;;
esac
