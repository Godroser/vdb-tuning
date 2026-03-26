#!/usr/bin/env bash
#
# 等待本机进程结束后再启动 vdtuner（默认等待 PID 3399333：
#   python -u main_ottertune.py …）
#
# 用法（推荐：整条命令也用 nohup 包一层，等待期间断开 SSH 也不会中断）:
#
#   cd /talas-pool/home/z78ding/vdb-tuning/auto-configure/vdtuner
#   chmod +x wait_job_3399333_and_run_main_tuner.sh
#   nohup bash wait_job_3399333_and_run_main_tuner.sh > wait_job_3399333_nohup.log 2>&1 & disown
#
# 说明:
# - 脚本内已对 python 使用 nohup … & disown，本地终端关闭后 tuner 仍继续在机器上跑。
# - 「关机 / 重启」本机后所有进程都会停止；仅「断 SSH / 登出」不会打断已 disown 的后台进程。
# - 若 otter 重启后换了新 PID，请改下方 WAIT_PID。
#
set -euo pipefail

# 要等待的进程：ps 里第二列，对应 main_ottertune.py
WAIT_PID=3399333
VENV="/talas-pool/home/z78ding/venv/bin/activate"
WORKDIR="/talas-pool/home/z78ding/vdb-tuning/auto-configure/vdtuner"
LOG="random-match-int-2048-angular-no-filters.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

wait_for_pid() {
  local pid=$1
  if ! kill -0 "${pid}" 2>/dev/null; then
    log "进程 PID ${pid} 不存在（可能已结束），直接启动 tuner。"
    return 0
  fi

  log "等待 PID ${pid} 结束（期望: python -u main_ottertune.py）…"
  while kill -0 "${pid}" 2>/dev/null; do
    sleep 30
  done
  log "PID ${pid} 已退出。"
}

wait_for_pid "${WAIT_PID}"

# shellcheck source=/dev/null
source "${VENV}"
cd "${WORKDIR}"

nohup python -u main_tuner.py > "${LOG}" 2>&1 &
disown || true

log "已在后台启动: python -u main_tuner.py"
log "日志文件: ${WORKDIR}/${LOG}"
