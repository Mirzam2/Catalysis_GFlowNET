#!/usr/bin/env bash
# Шаг 3 (тернар): обучение GFlowNet на тройном search-space.
#
# Тернар включается через config/run.yaml (секция search_space: max_diff_elem=3).
# Отдельно ничего в constants.py править не надо.
#
# Warm-start с бинарного чекпоинта: max_diff_elem 2->3 НЕ меняет action space
# (голова политики та же), поэтому бинарный .ckpt грузится напрямую. Дать путь:
#   WARMSTART_CKPT=logs/pdh_gfn/ckpts/xxx.ckpt ./scripts/cluster/03_train_ternary.sh
# Без него — обучение с нуля.
#
# Опции (env):
#   N_STEPS=3000       число шагов (иначе из run.yaml)
#   CONFIG=config/run.yaml
#   EXTRA="--random-action-prob 0.05"   любые доп. флаги train.py
#
# Запуск в фоне:  nohup ./scripts/cluster/03_train_ternary.sh &
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

if [[ ! -f data/mp_pdm_entries.json ]]; then
  echo "ОШИБКА: нет data/mp_pdm_entries.json — сначала 01_rebuild_hull_ternary.sh" >&2
  exit 1
fi

LOG="$LOG_DIR/train_ternary_$(_ts).log"
echo "лог: $LOG"

ARGS=(--config "${CONFIG:-config/run.yaml}" --uma-device "$UMA_DEVICE"
      --curriculum --profile)
[[ -n "${N_STEPS:-}" ]] && ARGS+=(--n-steps "$N_STEPS")
[[ -n "${WARMSTART_CKPT:-}" ]] && ARGS+=(--resume-from "$WARMSTART_CKPT")
[[ -n "${EXTRA:-}" ]] && ARGS+=($EXTRA)

echo "команда: $RUNNER scripts/train.py ${ARGS[*]}"
$RUNNER scripts/train.py "${ARGS[@]}" 2>&1 | tee "$LOG"
