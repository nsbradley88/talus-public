#!/bin/sh
# talus-server.service: Talus (the multi-model Strata fork) behind Strata's server on :8000, all interfaces, API key
# required (from the unit's EnvironmentFile, /etc/vllm/secrets.env, root only).  The run config comes from the active
# llm-model preset (TALUS_CONFIG=... in /etc/llama/active.env), else Laguna-S-2.1's.
cfg=$(sed -n 's/^TALUS_CONFIG=//p' /etc/llama/active.env 2>/dev/null)
[ -n "$cfg" ] || cfg=/path/to/laguna-s-2.1/talus-service.json
export STRATA_API_KEY="${STRATA_API_KEY:-${LLAMA_API_KEY:-${VLLM_API_KEY:-}}}"
[ -n "$STRATA_API_KEY" ] || { echo "talus-server: no API key in /etc/vllm/secrets.env" >&2; exit 1; }
cd /path/to/talus
exec python3 serve/server.py --engine strata --config "$cfg" --host 0.0.0.0 --port 8000
