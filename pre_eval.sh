#!/bin/bash
set -Eeuo pipefail

OUTPUT_DIR="/workspace/ovon/output"
LOG_FILE="$OUTPUT_DIR/helper.log"
REPO_DIR="/workspace/ovon"

# 确保日志目录存在
mkdir -p "$(dirname "$LOG_FILE")"

log() {
  local ts
  ts="$(date '+%F %T%z')"
  printf "[%s] %s\n" "$ts" "$*" >> "$LOG_FILE"
}

# 从环境变量获取{task_name}环境变量，写入/workspace/ovon/output/experiments_log
WORKFLOW_NAME="${workflow_name:-${WORKFLOW_NAME:-}}"
if [[ -z "${WORKFLOW_NAME}" ]]; then
  log "WARN: 环境变量 workflow_name / WORKFLOW_NAME 未设置。"
else
  log "WORKFLOW_NAME=${WORKFLOW_NAME}"
fi

# 获取/workspace/ovon github仓库的分支 写/workspace/ovon/output/experiments_log
# === 读取 Git 分支与提交信息 ===
if [[ -d "$REPO_DIR/.git" ]]; then
  # 当前分支名（若处于分离头指针，会返回 HEAD）
  BRANCH="$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'UNKNOWN')"
  # 当前提交短哈希
  COMMIT_SHORT="$(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo 'UNKNOWN')"

  log "repo=${REPO_DIR}"
  log "branch=${BRANCH}"
  log "commit=${COMMIT_SHORT}"
else
  log "ERROR: ${REPO_DIR} 不是一个 Git 仓库（未找到 .git 目录）。"
fi

# 从环境变了获取config目录，写入/workspace/ovon/experiments_log
EXP_CONFIG_PATH="${exp_config:-${EXP_CONFIG:-}}"
if [[ -z "${EXP_CONFIG}" ]]; then
  log "WARN: 环境变量 exp_config / EXP_CONFIG 未设置。"
else
  log "exp_config_path=${EXP_CONFIG_PATH}"
fi

EVAL_SPLIT="${eval_split:-${EVAL_SPLIT:-}}"
if [[ -z "${EVAL_SPLIT}" ]]; then
  log "WARN: 环境变量 eval_split/ EVAL_SPLIT未设置。"
else
  log "eval_split=${EVAL_SPLIT}"
fi

TASK_DATASET_PATH="${task_dataset_path:-${TASK_DATASET_PATH:-}}"
if [[ -z "${TASK_DATASET_PATH}" ]]; then
  log "WARN: 环境变量 task_dataset_path/ TASK_DATASET_PATH"
else
  log "task_dataset_path=${TASK_DATASET_PATH}"
fi


EVAL_CKPT_PATH_DIR="${eval_ckpt_path_dir:-${EVAL_CKPT_PATH_DIR:-}}"
if [[ -z "${EVAL_CKPT_PATH_DIR}" ]]; then
  log "WARN: 环境变量 eval_ckpt_path_dir/ EVAL_CKPT_PATH_DIR"
else
  log "eval_ckpt_path_dir=${EVAL_CKPT_PATH_DIR}"
fi


cp ${EXP_CONFIG_PATH} $OUTPUT_DIR/config.yaml

# 启动训练，（容易变动的参数通过命令行参数带入，不怎么动的就在config文件中，然后代码里面获取最后的config内容也写入日志中）训练中有些关键的日志可以保存到workspace/ovon/experiments_log