#!/bin/bash
set -Eeuo pipefail

OUTPUT_DIR="/workspace/ovon/output"
LOG_FILE="$OUTPUT_DIR/helper.log"
REPO_DIR="/workspace/ovon"

log() {
  local ts
  ts="$(date '+%F %T%z')"
  printf "[%s] %s\n" "$ts" "$*" >> "$LOG_FILE"
}

log "finish"