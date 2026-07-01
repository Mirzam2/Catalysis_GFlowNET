# Кластерный запуск — тернарные системы Pd-M-M'

Набор скриптов для перевода пайплайна на тройные интерметаллиды. Тернар
включён в `config/run.yaml` (секция `search_space: max_diff_elem=3`) — править
`constants.py` не нужно.

Раннер по умолчанию — `uv run python` (валидированный стек). Переопределить:
```bash
RUNNER="python" ./scripts/cluster/03_train_ternary.sh          # conda-активированный python
RUNNER="conda run -n pdhgfn python" ./scripts/cluster/03_train_ternary.sh
```
Общие переменные: `UMA_DEVICE` (дефолт `cuda`), `LOG_DIR` (дефолт `logs/cluster`).

## Порядок

### 1. Пересобрать hull в oc20 для тройных систем  — `01_rebuild_hull_ternary.sh`
Тянет Pd-M + M-M' + Pd-M-M' chemsys (~625) из MP, релаксирует в UMA oc20.
**Тяжёлый шаг: тысячи структур, часы на GPU.** Возобновляемый — при обрыве
просто запусти снова.
```bash
export MP_API_KEY=...
nohup ./scripts/cluster/01_rebuild_hull_ternary.sh &
tail -f logs/cluster/rebuild_hull_ternary_*.log
# отладка на 10 структурах:  LIMIT=10 ./scripts/cluster/01_rebuild_hull_ternary.sh
```

### 2. Проверить hull  — `02_check_hull.sh`
Счётчики фаз (должны быть унарные/бинарные/**тернарные**) + санити PdZn
(`e_above_hull ~ 0`). GPU/сеть не нужны.
```bash
./scripts/cluster/02_check_hull.sh
```

### 3. (опц.) Warmup кэша эталонами
Бинарные эталоны остаются валидны (дескрипторы element-agnostic) — полезны
как якоря. Команда без изменений:
```bash
uv run python scripts/warmup_cache.py --uma-device cuda --fmax 0.03 \
  --structures data/ref_structures/PdZn.cif data/ref_structures/PdGa.cif \
               data/ref_structures/PdIn.cif data/ref_structures/Pd3Sn_mp_718.cif
```

### 4. Обучение  — `03_train_ternary.sh`
```bash
# с нуля:
nohup ./scripts/cluster/03_train_ternary.sh &

# warm-start с бинарного чекпоинта (action space тот же, .ckpt грузится):
WARMSTART_CKPT=logs/pdh_gfn/ckpts/<last>.ckpt N_STEPS=3000 \
  nohup ./scripts/cluster/03_train_ternary.sh &

tail -f logs/cluster/train_ternary_*.log
```

## Заметки
- **Чистый тернар** (без бинаров): в `config/run.yaml` поставь
  `search_space.min_diff_elem: 3`. Тогда бинарные seed-затравки станут
  невалидны — либо отключи seed buffer, либо добавь тройные затравки.
- **Warm-start работает** только пока не менялись `space_groups` / `miller_set`
  / `m_elements` (они меняют размер action space → чекпоинт несовместим,
  учить с нуля).
- Если `02_check_hull` показал записи без `mp_id` — это остатки GGA-шкалы,
  нужен полный пересчёт шагом 1.
