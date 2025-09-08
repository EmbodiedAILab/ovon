#!/bin/bash]
OUTPUT_DIR="/workspace/ovon/output"
LOG_FILE="$OUTPUT_DIR/helper.log"

log_and_execute() {
  local ts cmd_line rc start end dur
  : "${LOG_FILE:=/workspace/ovon/experiments_log}"   # 默认日志文件
  mkdir -p "$(dirname "$LOG_FILE")"

  if [[ $# -eq 0 ]]; then
    echo "log: need a command" >&2
    return 2
  fi

  ts="$(date '+%F %T%z')"

  # 生成可读的命令行文本
  if [[ $# -eq 1 ]]; then
    cmd_line="$1"                               # 单字符串（可含管道/重定向）
  else
    cmd_line="$(printf '%q ' "$@")"; cmd_line="${cmd_line% }"  # 逐参数转义
  fi

  printf "[%s] $ %s\n" "$ts" "$cmd_line" >> "$LOG_FILE"

  start=$(date +%s)
  if [[ $# -eq 1 ]]; then
    bash -lc "$1"                               # 用 shell 解析整条命令
  else
    "$@"                                        # 直接执行参数形式
  fi
  rc=$?
  end=$(date +%s); dur=$((end - start))

  ts="$(date '+%F %T%z')"
  printf "[%s] (exit=%d, %ss)\n" "$ts" "$rc" "$dur" >> "$LOG_FILE"
  return "$rc"
}