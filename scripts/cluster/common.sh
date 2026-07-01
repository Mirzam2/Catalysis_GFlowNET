#!/usr/bin/env bash
# Общая преамбула кластерных скриптов PDH-GFlowNet (тернарный пайплайн).
# Подключается через `source`. Задаёт RUNNER, корень репозитория, логи.
#
# Раннер по умолчанию — uv (валидированный стек, см. INSTALL_an01.md §3b).
# Переопределить:  RUNNER="python" ./scripts/cluster/xxx.sh
#   или conda:     RUNNER="conda run -n pdhgfn python" ...
set -euo pipefail

# Корень репозитория (скрипт лежит в scripts/cluster/)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Раннер: `uv run python` по умолчанию, любой можно передать через $RUNNER
RUNNER="${RUNNER:-uv run python}"

# Директория логов
LOG_DIR="${LOG_DIR:-logs/cluster}"
mkdir -p "$LOG_DIR"

# Устройство UMA
UMA_DEVICE="${UMA_DEVICE:-cuda}"

_ts() { date +%Y%m%d_%H%M%S; }

echo "=== PDH-GFlowNet cluster ==="
echo "  repo:       $REPO_ROOT"
echo "  runner:     $RUNNER"
echo "  uma_device: $UMA_DEVICE"
echo "  log_dir:    $LOG_DIR"
echo "==========================="
