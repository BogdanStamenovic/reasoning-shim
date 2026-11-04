#!/usr/bin/env bash
# ownbox lifecycle for reasoning-shim: setup | update | remove. Runs with the checkout as cwd.
# Idempotent: setup and update can be rerun; update never touches the config file.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
CONF_DIR=${XDG_CONFIG_HOME:-$HOME/.config}/reasoning-shim
CONF=$CONF_DIR/env
STATE=${XDG_STATE_HOME:-$HOME/.local/state}/reasoning-shim
UNIT_DIR=${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user
UNIT=$UNIT_DIR/reasoning-shim.service
tty() { [ -t 0 ]; }

deps() {
  if command -v uv >/dev/null; then
    [ -x "$HERE/.venv/bin/python" ] || uv venv -q "$HERE/.venv"
    uv pip install -q --python "$HERE/.venv" -e "$HERE"
  else
    [ -x "$HERE/.venv/bin/python" ] || python3 -m venv "$HERE/.venv"
    "$HERE/.venv/bin/pip" install -q -e "$HERE"
  fi
}

config() {
  mkdir -p "$CONF_DIR" "$STATE"
  if [ -f "$CONF" ]; then
    echo "reasoning-shim: keeping existing config $CONF"
    return
  fi
  local up=${RSHIM_UPSTREAM:-http://127.0.0.1:11434}
  if tty; then
    read -r -p "Upstream OpenAI-compatible server [$up]: " ans || true
    up=${ans:-$up}
  else
    echo "reasoning-shim: no TTY; upstream = $up (set RSHIM_UPSTREAM or edit $CONF)"
  fi
  cat > "$CONF" <<EOF
# reasoning-shim config (read by the systemd unit). Restart after editing:
#   systemctl --user restart reasoning-shim
SHIM_UPSTREAM=$up
SHIM_HOST=127.0.0.1
SHIM_PORT=11450
# Reasoning budgets in tokens, -1 = unlimited. reasoning_effort from the client wins:
# none/minimal=0, low=256, medium=1024; high and unset fall through to these.
SHIM_BUDGET_TRIVIAL=96
SHIM_BUDGET_TOOL=768
SHIM_BUDGET_DEFAULT=-1
SHIM_BUDGETS=1
SHIM_REPAIR=1
# One JSON line of metadata per request (no prompt or answer text).
SHIM_LOG=$STATE/shim.jsonl
EOF
  echo "reasoning-shim: wrote $CONF"
}

unit() {
  mkdir -p "$UNIT_DIR"
  cat > "$UNIT" <<EOF
[Unit]
Description=reasoning-shim: reasoning budgets + stranded tool-call repair in front of llama-server
After=network-online.target

[Service]
EnvironmentFile=$CONF
ExecStart=$HERE/.venv/bin/reasoning-shim serve
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
EOF
  systemctl --user daemon-reload
}

case "${1:-}" in
  setup)
    deps; config; unit
    systemctl --user enable --now reasoning-shim.service
    systemctl --user restart reasoning-shim.service
    echo "reasoning-shim: listening on $(. "$CONF"; echo "http://$SHIM_HOST:$SHIM_PORT/v1") -> $(. "$CONF"; echo "$SHIM_UPSTREAM")"
    ;;
  update)
    deps
    [ -f "$CONF" ] || config
    unit
    systemctl --user restart reasoning-shim.service
    echo "reasoning-shim: updated and restarted"
    ;;
  remove)
    systemctl --user disable --now reasoning-shim.service 2>/dev/null || true
    rm -f "$UNIT"; systemctl --user daemon-reload
    del=n
    if tty; then read -r -p "Also delete config $CONF_DIR and logs $STATE? [y/N]: " del || true; fi
    if [ "${del:-n}" = y ] || [ "${del:-n}" = Y ]; then rm -rf "$CONF_DIR" "$STATE"; echo "reasoning-shim: config and logs deleted"
    else echo "reasoning-shim: kept $CONF_DIR and $STATE"; fi
    ;;
  *) echo "usage: lifecycle.sh setup|update|remove" >&2; exit 2 ;;
esac
