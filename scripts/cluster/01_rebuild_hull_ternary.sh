#!/usr/bin/env bash
# Шаг 1 (тернар): пересчёт выпуклой оболочки в oc20-шкале для Pd-M-M' систем.
#
# Тянет структуры всех Pd-M + M-M' + Pd-M-M' chemsys (~625) из MP, релаксирует
# в UMA oc20, пишет data/mp_pdm_entries.json. ТЯЖЁЛЫЙ шаг: тысячи структур,
# часы на GPU. Прогон ВОЗОБНОВЛЯЕМЫЙ (сохранение после каждой записи, дедуп по
# mp_id) — если упало/прервалось, просто запусти скрипт снова.
#
# Нужен MP_API_KEY:  export MP_API_KEY=...
# Запуск в фоне:     nohup ./scripts/cluster/01_rebuild_hull_ternary.sh &
#
# Опции (env-переменные):
#   FMAX=0.05     порог сил при релаксации
#   LIMIT=10      ограничить число структур (для отладки)
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

if [[ -z "${MP_API_KEY:-}" ]]; then
  echo "ОШИБКА: нужен MP_API_KEY (export MP_API_KEY=...)" >&2
  exit 1
fi

LOG="$LOG_DIR/rebuild_hull_ternary_$(_ts).log"
echo "лог: $LOG"

ARGS=(--uma-device "$UMA_DEVICE" --fmax "${FMAX:-0.05}")
[[ -n "${LIMIT:-}" ]] && ARGS+=(--limit "$LIMIT")

$RUNNER scripts/rebuild_hull_oc20.py "${ARGS[@]}" 2>&1 | tee "$LOG"

echo
echo "готово. hull: data/mp_pdm_entries.json"
echo "проверь, что для известной фазы (напр. PdZn) e_above_hull ~ 0 —"
echo "запусти:  $RUNNER scripts/cluster/02_check_hull.sh"
