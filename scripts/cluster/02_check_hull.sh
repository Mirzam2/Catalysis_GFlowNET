#!/usr/bin/env bash
# Шаг 2 (тернар): проверка собранного hull — счётчики фаз + санити PdZn.
# Не требует GPU/MP. Запускать после 01_rebuild_hull_ternary.sh.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
$RUNNER scripts/check_hull.py "$@"
