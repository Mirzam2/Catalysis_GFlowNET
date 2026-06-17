# Установка PDH-GFlowNet на кластер an01

Гайд предполагает: рабочее conda-окружение с **проверенным UMA** (тем самым,
на котором гонялся `run_uma.py` со сверкой против DFT), файловая система
beegfs, вычислительные узлы без интернета (всё сетевое — на логин-узле).

---

## 0. Стратегия окружения — прочитать перед установкой

Каркас Mila gflownet пинит `torch ==2.5.1` и `pymatgen ==2024.3.1`.
fairchem-core (UMA) живёт на более свежем стеке. **Не даём pip ломать
окружение, в котором UMA уже валидирован против DFT** — это наш главный
актив. Поэтому порядок такой:

1. клонируем валидированное UMA-окружение (оригинал не трогаем вообще);
2. в клоне ослабляем пины gflownet локально и ставим его `--no-deps`-стилем
   с ручным контролем;
3. если конфликт всё же всплывёт — запасной план в разделе 7.

```bash
conda create --name pdhgfn --clone <имя_вашего_uma_окружения>
conda activate pdhgfn
python -c "from fairchem.core import pretrained_mlip; print('fairchem OK')"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Если клонирование падает по квоте home — окружения в beegfs:
```bash
conda create --prefix /beegfs/home/georgii.bychkov/envs/pdhgfn --clone <uma_env>
```

## 1. Кэши и переменные окружения (beegfs, офлайн-узлы)

Веса UMA качаются с HuggingFace на логин-узле один раз; на вычислительных
узлах включаем офлайн-режим, чтобы fairchem не лез в сеть.

```bash
# в ~/.bashrc или в sbatch-скрипте
export HF_HOME=/beegfs/home/georgii.bychkov/.cache/huggingface
export WANDB_MODE=offline    # логи wandb локально, синк руками с логин-узла

# один раз на логин-узле:
huggingface-cli login        # токен с доступом к facebook/UMA
python -c "
from fairchem.core import pretrained_mlip
pretrained_mlip.get_predict_unit('uma-s-1p2', device='cpu')
print('веса uma-s-1p2 в кэше')
"
# на вычислительных узлах (в sbatch):
export HF_HUB_OFFLINE=1
```

Веса уже в кэше после прогонов `run_uma.py` — тогда шаг сводится к
проверке, что `HF_HOME` указывает на тот же путь.

## 2. Каркас Mila gflownet

```bash
cd /beegfs/home/georgii.bychkov/projects
git clone https://github.com/alexhernandezgarcia/gflownet mila-gflownet
cd mila-gflownet
```

Ослабляем пины в локальном клоне (это нормальная практика для editable-
установки, upstream не трогаем):

```bash
sed -i 's/torch ==2.5.1/torch >=2.5/' pyproject.toml
sed -i 's/pymatgen ==2024.3.1/pymatgen >=2024.3.1/' pyproject.toml
```

Ставим **без** extras `materials` (его зависимость `dave` — прокси-MLP
Crystal-GFN — нам не нужна, у нас свой физический прокси; pymatgen/pyxtal
поставим сами):

```bash
pip install -e . --no-build-isolation
pip install pyxtal "pymatgen>=2024.6"   # >=2024.6: filter_out_sym_slabs в get_slabs
                                         # (в коде есть фолбэк и для более старых)
```

Контроль, что torch не переехал:
```bash
python -c "import torch; print(torch.__version__)"   # та же версия, что в шаге 0!
python -c "from fairchem.core import pretrained_mlip; print('UMA жив')"
python -c "from gflownet.envs.crystals.crystal import Crystal; print('gflownet жив')"
```

Если pip при установке gflownet попытался даунгрейдить torch — прервать
(`Ctrl+C`), поставить `pip install -e . --no-deps`, затем доставить руками
недостающее: `pip install botorch hydra-core torchtyping wandb plotly`.

## 3. pdh-gfn

```bash
cd /beegfs/home/georgii.bychkov/projects
unzip pdh-gfn.zip && cd pdh-gfn
pip install -e .
```

Дымовой тест без GPU (механика конвейера на EMT, ~2-3 мин на CPU):
```bash
python tests/test_pipeline.py
# ожидаем: ВСЕ ТЕСТЫ ПРОШЛИ
```

Первый тест torch-части (вложенный Stack — главный кандидат на сюрпризы):
```bash
python -c "
from pdh_gfn.envs.catalyst import PdMCatalyst
env = PdMCatalyst()
print('action_space_dim:', env.action_space_dim)
s = env.reset()
print('source state OK')
"
```

## 4. Записи Materials Project для оболочки (логин-узел, один раз)

```bash
pip install mp-api
export MP_API_KEY=<ключ>
python scripts/fetch_mp_entries.py --out data/mp_pdm_entries.json
```

## 5. ОБЯЗАТЕЛЬНО перед боевым запуском

1. **`pdh_gfn/constants.py`: BEP_CH_*, BEP_CC_*, MAMUN_* — плейсхолдеры.**
   Подставить регрессионные коэффициенты из рабочих таблиц проекта.
2. **Go/no-go по Purdy**: наш конвейер использует ту же связку
   (uma-s-1p2 + task=oc20 + LBFGS/FIRE), что и валидированный `run_uma.py`,
   но порог RMSE < 0.2 эВ должен быть подтверждён именно на 6 точках Purdy
   до запуска генерации.
3. `E_ACT_CH_MAX` в constants.py привязать к E_act(C-H) Pd(111) из бенчмарка.
4. `e_adsorbate_ref` для CH (аргумент RewardPipeline) — согласовать схему
   отсчёта с Purdy/Mamun (газофазный CH4 - 3/2 H2 или как в их таблицах).

## 6. Запуск

Интерактивная проверка на GPU-узле:
```bash
srun --gres=gpu:1 --time=0:30:00 --pty bash
conda activate pdhgfn
export HF_HUB_OFFLINE=1
python scripts/train.py --device cuda --n-steps 1   # сборка env+proxy
```

Шаблон sbatch (`sbatch/train_pdhgfn.sh`):
```bash
#!/bin/bash
#SBATCH --job-name=pdhgfn
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out

source ~/.bashrc
conda activate pdhgfn
export HF_HOME=/beegfs/home/georgii.bychkov/.cache/huggingface
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

cd /beegfs/home/georgii.bychkov/projects/pdh-gfn
python scripts/train.py --device cuda \
    --hull-entries data/mp_pdm_entries.json \
    --cache data/reward_cache.jsonl \
    --fmax 0.10
```

Кэш наград (`reward_cache.jsonl`) персистентен — перезапуски джоба
переиспользуют все уже посчитанные UMA-оценки. Не удалять между запусками.

## 7. Известные грабли

| Симптом | Причина | Лечение |
|---|---|---|
| pip даунгрейдит torch при установке gflownet | пин `torch==2.5.1` | шаг 2: sed по pyproject; или `--no-deps` + ручная доустановка |
| `TypeError: get_slabs() ... filter_out_sym_slabs` | старый pymatgen | в коде есть фолбэк; либо `pip install -U pymatgen` |
| fairchem лезет в сеть на вычислительном узле | нет `HF_HUB_OFFLINE=1` | экспортировать в sbatch |
| `torchtyping` ругается на новый torch | известная хрупкость | `pip install torchtyping==0.1.5`; если не помогает — написать мне, заменим аннотации |
| Квота home переполнена кэшем HF | дефолтный `~/.cache` | `HF_HOME` на beegfs (шаг 1) |
| Конфликт окружений неразрешим | несовместимые стеки | запасной план: два окружения + прокси как подпроцесс. RewardPipeline вынести в скрипт-воркер в UMA-окружении, общение через файлы/сокет. Каркас под это уже готов (Potential — интерфейс), доделка ~полдня |

## 8. Порядок первых экспериментов

1. `tests/test_pipeline.py` (CPU) — механика.
2. Smoke UMA: один вызов `RewardPipeline` на PdZn L1_0 (111) вручную,
   сверить BE(CH*) с ожиданиями из скрининга 418 фаз.
3. Go/no-go Purdy (раздел 5.2).
4. Затравка буфера (`pdh_gfn/seeding.py`) + 10 траекторий с крошечным
   бюджетом — проверка стыковки с GFlowNetAgent.
5. Полноценный запуск с fmax=0.10, после выхода на плато — дообучение
   с fmax=0.03.

## 9. Ускорение пайплайна

Профиль награды: **84% времени — адсорбционные релаксации** (3 адсорбата × N сайтов). Оптимизации, по убыванию выигрыша:

| # | Оптимизация | Выигрыш | Статус | Флаг |
|---|---|---|---|---|
| 1 | Батч-релаксация UMA на GPU | ×5–20 | каркас (`potential/batch.py`), доделать на кластере по fairchem v2 API | — |
| 2 | Предфильтр сайтов (single-point → топ-K) | ×2–4 | готово | `--prefilter-keep N` (откл. `--no-prefilter`) |
| 3 | Двухступенчатый fmax (грубо→точно) | ×2–3 | флаги есть (`--fmax`), логика переключения — в train | `--fmax 0.2` затем дообучение `0.03` |
| 4 | Ранний выход по E_hull | ×1.5–3 | готово | `--e-hull-cutoff 0.15` |
| 7 | Меньше сайтов для H* | ×1.3 | готово | `--max-sites-light 3` |

Готовые (#2,#4,#7) проверены на EMT — E_sel не меняется, ранжирование сохраняется. Включены по умолчанию с безопасными значениями.

**Главный рычаг — #1 (батчинг).** Каркас и две стратегии реализации (fairchem batch-relax / ручной FIRE по пакету) описаны в `pdh_gfn/potential/batch.py`. Это даёт основное ускорение, но требует проверки batch-API fairchem на реальном GPU — поэтому оставлен как следующий шаг, а не зашит вслепую.

**Рекомендация по порядку:** сначала прогнать с готовыми #2/#4/#7 (уже ×2–4 суммарно на боевых slab'ах с десятками сайтов), затем профилировать и внедрять #1, если пропускной способности не хватает для бюджета 1e4–1e5 оценок.
