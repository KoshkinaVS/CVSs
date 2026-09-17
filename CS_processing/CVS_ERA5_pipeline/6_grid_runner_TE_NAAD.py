"""
Grid-раннер (NAAD, LoRes/HiRes) — сквозной перебор сетки

    eps x size_filter x extr_type x maxgap x mintime x prioritize

поверх Stage B/C/D пайплайна NAAD, вызываемых как обычные Python-функции (без
Snakemake, как и у ERA5-версии). Аналог 6_grid_runner_TE_ERA5.py, но урезан
под текущий скоуп задачи NAAD (2026-09-05): "только до треков" — то есть
только Stage B (1_create_Nodes_from_DBSCAN_NAAD.py) + Stage C
(2_run_StitchNodes_by_year.py, переиспользуется от ERA5 как есть) + Stage D
(3_create_csv_tracks_from_StitchNodes.py, тоже переиспользуется как есть).

Чего ЗДЕСЬ НЕТ и почему
--------------------------
Stage E' (расчёт доп. параметров вроде SLP/ветра на узлах,
4_compute_era5_params_for_nodes.py) и Stage E (джойн этих параметров в треки,
5_join_params_into_tracks.py) сознательно НЕ перенесены — у ERA5-версии они
завязаны на add_params_v2_fast_2026-08-24.py (ERA5DataLoader/
CycloneProcessorERA5, читает ERA5-файлы по конкретным именам переменных).
Для NAAD нужен WRF-эквивалент этого расчётного ядра — отдельная задача, пока
не в скоупе (см. обсуждение 2026-09-05: "только до треков"). Если он
понадобится позже, --stage eprime/e из 6_grid_runner_TE_ERA5.py можно
перенести сюда по той же схеме, что и tracking/csv ниже — сообщите.

Ключевое отличие от ERA5-версии
-----------------------------------
- data_type (LoRes/HiRes) — параметр запуска (--data-type), а не константа
  файла: пути/dist_m различаются (см. naad_config.py), но сама логика Stage
  B/C/D одна на оба. Один запуск этого раннера = один data_type (запускайте
  дважды для LoRes и для HiRes, как и Stage B/C-скрипты по отдельности).
- extr_type: только "global"/"local" (у NAAD нет geom-варианта, в отличие от
  ERA5 - там id="geom" использует отдельный критерий по геометрии кластера,
  для NAAD такой Stage A не считался).
- eps/min_samples фиксированы под то, с чем реально посчитан Stage A NAAD
  (config_NAAD.json: eps=2, min_samples=4) - в отличие от ERA5, где по eps
  есть сетка [1, 2]. Если посчитаете Stage A NAAD для другого eps -
  добавьте в EPS_VALUES.
- MAXGAP_VALUES/MINTIME_VALUES ниже - ПЕРЕНЕСЕНЫ ПО АНАЛОГИИ с ERA5-версией
  (не пересчитаны/не перепроверены для NAAD) и округлены до кратных 3ч -
  нативному шагу NAAD (см. naad_config.DEFAULT_TIMESTEP_HOURS) - maxgap
  меньше 6ч (= 1 нативный шаг) не имеет смысла на 3-часовых данных.

Пример запуска
--------------
python 6_grid_runner_TE_NAAD.py --data-type LoRes --dry-run
python 6_grid_runner_TE_NAAD.py --data-type LoRes --years 2010
python 6_grid_runner_TE_NAAD.py --data-type LoRes --years 2010 --stage tracking
python 6_grid_runner_TE_NAAD.py --data-type LoRes --years 2010 --stage csv
python 6_grid_runner_TE_NAAD.py --data-type HiRes --years 2010

--stage tracking (Stage B на всех size_filter + Stage C) и --stage csv
(только Stage D, по уже готовому tracks_txt) можно запускать раздельно и
повторно — та же логика, что в 6_grid_runner_TE_ERA5.py (skip-if-exists на
каждом шаге, отдельная строка лога на лист сетки со статусом
tracking_ok/tracking_error/csv_ok/csv_error/pending_tracking). --stage all
(по умолчанию) — tracking, потом csv, последовательно в одном процессе.

2026-09-05: НЕ прогонялось на реальных данных (нет доступа к серверу из этой
сессии) — см. TODO в 1_create_Nodes_from_DBSCAN_NAAD.py и naad_config.py.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import sys
import traceback
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import naad_config as cfg

# Модули пронумерованы (1_xxx.py, 2_xxx.py, 3_xxx.py) - имя не годится для
# обычного `import 1_xxx`, поэтому importlib (тот же приём, что и в
# 6_grid_runner_TE_ERA5.py).
stage_b = importlib.import_module("1_create_Nodes_from_DBSCAN_NAAD")
stage_c = importlib.import_module("2_run_StitchNodes_by_year")  # ERA5-модуль, переиспользуется как есть
stage_d = importlib.import_module("3_create_csv_tracks_from_StitchNodes")  # тоже переиспользуется как есть


# ============================================================================
# CONFIG — сетка перебора и фиксированные параметры. Правьте здесь (как и в
# 6_grid_runner_TE_ERA5.py), а не через CLI, кроме --data-type/--region/
# --years/--stage/--dry-run (см. main()).
# ============================================================================

SIGMA = cfg.DEFAULT_SIGMA
MIN_SAMPLES = cfg.DEFAULT_MIN_SAMPLES

EPS_VALUES = [cfg.DEFAULT_EPS]  # Stage A NAAD посчитан только для eps=2 (config_NAAD.json)
SIZE_FILTER_VALUES = [10, 25, 49]  # 10 обязателен - это и есть SOURCE_SIZE_FILTER (см. naad_config.py)
EXTR_TYPES = ["global", "local"]  # без "geom" - для NAAD такой Stage A не считался

SEARCH_RANGE = cfg.DEFAULT_SEARCH_RANGE_DEG  # фиксирован (не перебирается), как у ERA5
# 2026-09-05, по явной просьбе пользователя: точно такие же значения, как в
# сетке ERA5 (6_grid_runner_TE_ERA5.py), а не урезанные/округлённые под
# 3-часовой нативный шаг NAAD (более ранняя версия этого файла ограничивала
# снизу maxgap=6ч/mintime=18ч, чтобы не оказаться меньше одного нативного
# шага - см. историю правок). Держим сопоставимость с ERA5-сеткой важнее
# такой "оптимизации": maxgap=3ч и 4ч на 3-часовых данных, скорее всего, дадут
# одинаковый результат (реальные разрывы между срезами кратны 3ч), это не
# баг - оба значения оставлены специально.
MAXGAP_VALUES = [3, 4, 6, 9, 12]
MINTIME_VALUES = [9, 12, 18, 24]
PRIORITIZE_VALUES = [False, True]

STAGE_D_VARIABLE_NAMES = ["rad", "r2d", "wspd"]
STAGE_D_FAST = True  # см. докстринг convert_year() в 3_create_csv_tracks_from_StitchNodes.py

DATA_TYPE = None  # переопределяется в main() из --data-type ДО первого использования
REGION_NAME = cfg.DEFAULT_REGION_NAME  # переопределяется в main() из --region
COMBO_LOG_PATH = None  # переопределяется в main() (зависит от DATA_TYPE/REGION_NAME)

COMBO_LOG_FIELDS = [
    "timestamp", "eps", "size_filter", "extr_type", "search_range", "mintime", "maxgap",
    "prioritize", "year", "combo_dir", "postfix", "tracks_txt", "csv_tracks_dir",
    "status", "error",
]
# Схема лога сознательно совпадает по духу с ERA5-версией (grid_run_log_*.csv),
# но БЕЗ eprime/e-специфичных полей (params_parquet, final_dir,
# n_missing_params) - этих стадий здесь нет (см. докстринг модуля выше).


def log_row(row: dict) -> None:
    COMBO_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new = not COMBO_LOG_PATH.exists()
    with open(COMBO_LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COMBO_LOG_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


# ============================================================================
# Stage B
# ============================================================================

def run_stage_b(eps: int, size_filter: int, extr_type: str, year: int) -> None:
    """extr_type здесь без подчёркивания ('global'/'local') - Stage B сам
    добавляет '_' внутри (см. --extr-type в 1_create_Nodes_from_DBSCAN_NAAD.py
    после правки 2026-09-05)."""
    stage_b.run_for_combo(
        data_type=DATA_TYPE, year=year, size_filter=size_filter,
        extr_type=f"_{extr_type}", sigma=SIGMA, eps=eps, min_samples=MIN_SAMPLES,
        region_name=REGION_NAME, path_init=cfg.PATH_INIT,
    )


def combo_dir_path(eps: int, size_filter: int, extr_type: str) -> Path:
    """Та же формула, что использует Stage B (run_for_combo) и Stage C
    (build_dirs) для комбо-папки - единая точка правды, чтобы не разъезжаться
    со стадиями при правках. extr_type - без подчёркивания."""
    return Path(
        f"{cfg.PATH_INIT}/TempestExtremes/{DATA_TYPE}/R2D_{DATA_TYPE}_{REGION_NAME}_sigma_{SIGMA}/"
        f"{eps:02d}-{MIN_SAMPLES:02d}-{size_filter:02d}_{extr_type}"
    )


# ============================================================================
# Stage C + D (по одному листу сетки)
# ============================================================================

def run_stage_tracking_leaf(eps, size_filter, extr_type, maxgap, mintime, prioritize, year):
    """ТОЛЬКО Stage C (StitchNodes) для одного листа - см. run_stage_tracking_leaf
    в 6_grid_runner_TE_ERA5.py, здесь то же самое, только stage_c.
    run_stitchnodes_for_combo() вызывается с NAAD-путями (path_init/data_type)."""
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "eps": eps, "size_filter": size_filter, "extr_type": extr_type,
        "search_range": SEARCH_RANGE, "mintime": mintime, "maxgap": maxgap,
        "prioritize": prioritize, "year": year,
        "combo_dir": "", "postfix": "", "tracks_txt": "", "csv_tracks_dir": "",
        "status": "", "error": "",
    }
    try:
        combo_dir, postfix = stage_c.run_stitchnodes_for_combo(
            path_init=f"{cfg.PATH_INIT}/TempestExtremes",
            data_type=DATA_TYPE, region_name=REGION_NAME, sigma=SIGMA,
            eps=eps, min_samples=MIN_SAMPLES, size_filter=size_filter, extr_type=extr_type,
            years=[year], search_range=SEARCH_RANGE, mintime=mintime, maxgap=maxgap,
            prioritize=prioritize, skip_existing=True,
        )
        combo_dir = Path(combo_dir)
        row["combo_dir"] = str(combo_dir)
        row["postfix"] = postfix

        tracks_txt = combo_dir / f"Tracks_R2D_txt_files{postfix}" / f"{DATA_TYPE}_TC_tracks_{year}.txt"
        row["tracks_txt"] = str(tracks_txt)
        if not tracks_txt.exists():
            raise FileNotFoundError(f"StitchNodes не создал {tracks_txt} (см. вывод выше)")

        row["status"] = "tracking_ok"
    except Exception as e:
        row["status"] = "tracking_error"
        row["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        log_row(row)
    return row


def run_stage_csv_leaf(eps, size_filter, extr_type, maxgap, mintime, prioritize, year):
    """ТОЛЬКО Stage D (tracks_txt -> CSV на трек) для одного листа - независимо
    от того, когда/где отработал --stage tracking. combo_dir/postfix
    пересчитываются той же формулой, что и в run_stage_tracking_leaf."""
    combo_dir = combo_dir_path(eps, size_filter, extr_type)
    postfix = stage_c.build_stitch_postfix(SEARCH_RANGE, mintime, maxgap, prioritize)
    tracks_txt = combo_dir / f"Tracks_R2D_txt_files{postfix}" / f"{DATA_TYPE}_TC_tracks_{year}.txt"
    csv_tracks_dir = combo_dir / f"csv_Tracks{postfix}"

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "eps": eps, "size_filter": size_filter, "extr_type": extr_type,
        "search_range": SEARCH_RANGE, "mintime": mintime, "maxgap": maxgap,
        "prioritize": prioritize, "year": year,
        "combo_dir": str(combo_dir), "postfix": postfix,
        "tracks_txt": str(tracks_txt), "csv_tracks_dir": str(csv_tracks_dir),
        "status": "", "error": "",
    }
    try:
        if not tracks_txt.exists():
            row["status"] = "pending_tracking"
            print(
                f"[Stage D] {tracks_txt} ещё нет - Stage C для этого листа (--stage tracking) "
                f"ещё не отработал, конверсия в CSV отложена до повторного запуска --stage csv."
            )
            return row

        stage_d.convert_year(tracks_txt, csv_tracks_dir, STAGE_D_VARIABLE_NAMES, fast=STAGE_D_FAST)
        row["status"] = "csv_ok"
    except Exception as e:
        row["status"] = "csv_error"
        row["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        log_row(row)
    return row


def iter_grid():
    for eps in EPS_VALUES:
        for extr_type in EXTR_TYPES:
            for size_filter in SIZE_FILTER_VALUES:
                for maxgap in MAXGAP_VALUES:
                    for mintime in MINTIME_VALUES:
                        for prioritize in PRIORITIZE_VALUES:
                            yield eps, size_filter, extr_type, maxgap, mintime, prioritize


def main():
    global DATA_TYPE, REGION_NAME, COMBO_LOG_PATH

    parser = argparse.ArgumentParser(
        description="Grid-раннер Stage B/C/D (NAAD) по сетке eps x size_filter x extr_type x maxgap x mintime x prioritize"
    )
    parser.add_argument("--data-type", choices=["LoRes", "HiRes"], required=True)
    parser.add_argument("--years", type=int, nargs="+", default=[2010])
    parser.add_argument(
        "--region", default=cfg.DEFAULT_REGION_NAME,
        help=f"Имя папки-региона (по умолчанию '{cfg.DEFAULT_REGION_NAME}') - у NAAD нет реального "
             "деления на регионы (домен фиксирован WRF-сеткой), это только сегмент пути на диске, "
             "как REGION_NAME в ERA5-версии.",
    )
    parser.add_argument(
        "--stage", choices=["all", "tracking", "csv"], default="all",
        help="all = tracking, затем csv, последовательно (по умолчанию); tracking = ТОЛЬКО "
             "Stage B(все size_filter)+Stage C (быстро, вся сетка); csv = ТОЛЬКО Stage D "
             "по уже готовому tracks_txt (листья без готового tracking помечаются "
             "pending_tracking, запустите --stage csv повторно позже). Stage E'/E здесь нет "
             "вовсе - см. докстринг модуля.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Только показать размер сетки и пути, ничего не считать")
    args = parser.parse_args()

    DATA_TYPE = args.data_type
    REGION_NAME = args.region
    COMBO_LOG_PATH = Path(cfg.PATH_INIT) / "TempestExtremes" / DATA_TYPE / f"grid_run_log_{REGION_NAME}.csv"

    leaves = list(iter_grid())
    n_stage_b_combos = len(EPS_VALUES) * len(EXTR_TYPES) * len(SIZE_FILTER_VALUES)
    print(
        f"data_type={DATA_TYPE} регион={REGION_NAME} (лог: {COMBO_LOG_PATH})\n"
        f"Сетка: {len(leaves)} листьев (Stage C/D) x {len(args.years)} год(а/лет); "
        f"Stage B - {n_stage_b_combos} прогонов. --stage={args.stage}"
    )
    if args.dry_run:
        for eps, size_filter, extr_type, maxgap, mintime, prioritize in leaves[:5]:
            print(f"  например: eps={eps} size_filter={size_filter} extr_type={extr_type} "
                  f"maxgap={maxgap}h mintime={mintime}h prioritize={prioritize} "
                  f"-> {combo_dir_path(eps, size_filter, extr_type)}")
        print("  ... (--dry-run: выполнение пропущено)")
        return

    for year in args.years:
        if args.stage in ("all", "tracking"):
            for eps in EPS_VALUES:
                for extr_type in EXTR_TYPES:
                    for size_filter in SIZE_FILTER_VALUES:
                        print(f"\n[Stage B] eps={eps} size_filter={size_filter} extr_type={extr_type} year={year}")
                        run_stage_b(eps, size_filter, extr_type, year)

            for eps, size_filter, extr_type, maxgap, mintime, prioritize in leaves:
                print(
                    f"\n[Stage C] eps={eps} size_filter={size_filter} extr_type={extr_type} "
                    f"maxgap={maxgap}h mintime={mintime}h prioritize={prioritize} year={year}"
                )
                run_stage_tracking_leaf(eps, size_filter, extr_type, maxgap, mintime, prioritize, year)

        if args.stage in ("all", "csv"):
            for eps, size_filter, extr_type, maxgap, mintime, prioritize in leaves:
                print(
                    f"\n[Stage D] eps={eps} size_filter={size_filter} extr_type={extr_type} "
                    f"maxgap={maxgap}h mintime={mintime}h prioritize={prioritize} year={year}"
                )
                run_stage_csv_leaf(eps, size_filter, extr_type, maxgap, mintime, prioritize, year)

    print(f"\nГотово. Лог по каждой комбинации: {COMBO_LOG_PATH}")


if __name__ == "__main__":
    main()
