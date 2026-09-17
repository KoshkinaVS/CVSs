"""
Stage C (NAAD, LoRes/HiRes) — StitchNodes, тонкая обёртка над
run_stitchnodes_for_combo() из 2_run_StitchNodes_by_year.py (ERA5). Эта
функция уже полностью параметризована по data_type/region_name/sigma/eps/...
и ничего не знает о происхождении узлов (ERA5 или NAAD) - ей важен только уже
готовый Stage B (см. 1_create_Nodes_from_DBSCAN_NAAD.py), лежащий в ожидаемых
папках. Отдельный файл - чтобы задать NAAD-specific значения по умолчанию
(data_type, mintime/maxgap под 3-часовой шаг NAAD, см. naad_config.py) не
трогая и не рискуя сломать поведение для ERA5.

Требование к размещению файла
------------------------------
Должен лежать в той же папке, что 2_run_StitchNodes_by_year.py и
naad_config.py (импортирует оба). 2_run_StitchNodes_by_year.py импортируется
по пути через importlib (имя файла начинается с цифры, обычный import не
сработает) - тот же приём, что уже используется в
4_compute_era5_params_for_nodes.py для add_params_v2_fast_2026-08-24.py.

Пример запуска
--------------
python 2_run_StitchNodes_by_year_NAAD.py --data-type LoRes --years 2010
python 2_run_StitchNodes_by_year_NAAD.py --data-type HiRes --years 2010 \
    --search-range 1.5 --mintime 18 --maxgap 6

После этого Stage D — 3_create_csv_tracks_from_StitchNodes.py (ERA5-скрипт,
БЕЗ ИЗМЕНЕНИЙ, он уже общий) с --sigma-dir/--postfix/--data-type, которые
печатает этот скрипт в конце.

2026-09-05: не прогонялось на реальных данных (нет доступа к серверу из этой
сессии) - см. TODO в 1_create_Nodes_from_DBSCAN_NAAD.py и докстринг
naad_config.py про непроверенные предположения (XTIME/XLAT/XLONG, значения
mintime/maxgap под 3ч).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import naad_config as cfg

SCRIPT_DIR = Path(__file__).resolve().parent


def _load_stage_c_module():
    """Импорт 2_run_StitchNodes_by_year.py (ERA5) по пути файла - имя
    начинается с цифры, обычный `import` не сработает. См. аналогичный приём
    в 4_compute_era5_params_for_nodes.py (load_add_params_module)."""
    path = SCRIPT_DIR / "2_run_StitchNodes_by_year.py"
    if not path.exists():
        raise FileNotFoundError(
            f"Не найден {path} - этот скрипт должен лежать рядом с "
            "2_run_StitchNodes_by_year.py (ERA5-версией), т.к. переиспользует "
            "её run_stitchnodes_for_combo()."
        )
    spec = importlib.util.spec_from_file_location("stage_c_stitchnodes_era5", str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules["stage_c_stitchnodes_era5"] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(
        description="Stage C (NAAD): StitchNodes для LoRes/HiRes - обёртка над "
                     "run_stitchnodes_for_combo() из 2_run_StitchNodes_by_year.py"
    )
    parser.add_argument("--data-type", choices=["LoRes", "HiRes"], required=True)
    parser.add_argument("--region-name", default=cfg.DEFAULT_REGION_NAME)
    parser.add_argument("--sigma", type=int, default=cfg.DEFAULT_SIGMA)
    parser.add_argument("--eps", type=int, default=cfg.DEFAULT_EPS)
    parser.add_argument("--min-samples", type=int, default=cfg.DEFAULT_MIN_SAMPLES)
    parser.add_argument(
        "--size-filter", type=int, default=cfg.SOURCE_SIZE_FILTER,
        help="Тот же size_filter, с которым запускался Stage B (1_create_Nodes_from_DBSCAN_NAAD.py) "
             f"для этой комбинации (по умолчанию SOURCE_SIZE_FILTER={cfg.SOURCE_SIZE_FILTER})",
    )
    parser.add_argument("--extr-type", choices=["global", "local", "geom"], default="local")
    parser.add_argument("--years", type=int, nargs="+", required=True)
    parser.add_argument("--search-range", type=float, default=cfg.DEFAULT_SEARCH_RANGE_DEG,
                         help=f"Градусы (см. naad_config.py) - по умолчанию {cfg.DEFAULT_SEARCH_RANGE_DEG}, "
                              "перенесено по аналогии с ERA5, не перепроверено для NAAD")
    parser.add_argument("--mintime", type=int, default=cfg.DEFAULT_MINTIME_HOURS)
    parser.add_argument("--maxgap", type=int, default=cfg.DEFAULT_MAXGAP_HOURS,
                         help=f"Часы - по умолчанию {cfg.DEFAULT_MAXGAP_HOURS} "
                              f"(= 2 нативных шага NAAD x {cfg.DEFAULT_TIMESTEP_HOURS}ч, "
                              "не 1 шаг, как получилось бы при переносе ERA5-значения 3ч как есть)")
    parser.add_argument("--prioritize", action="store_true")
    parser.add_argument("--path-init", default=None,
                         help=f"По умолчанию {cfg.PATH_INIT}/TempestExtremes (как у ERA5-версии этого скрипта)")
    parser.add_argument("--stitchnodes-cmd", default="StitchNodes",
                         help="Убедитесь, что бинарник StitchNodes (TempestExtremes) доступен в PATH на сервере")
    args = parser.parse_args()

    stage_c = _load_stage_c_module()

    path_init = args.path_init or f"{cfg.PATH_INIT}/TempestExtremes"

    combo_dir, postfix = stage_c.run_stitchnodes_for_combo(
        path_init=path_init,
        data_type=args.data_type,
        region_name=args.region_name,
        sigma=args.sigma,
        eps=args.eps,
        min_samples=args.min_samples,
        size_filter=args.size_filter,
        extr_type=args.extr_type,
        years=args.years,
        search_range=args.search_range,
        mintime=args.mintime,
        maxgap=args.maxgap,
        prioritize=args.prioritize,
        stitchnodes_cmd=args.stitchnodes_cmd,
    )

    print(f"\nДля Stage D (3_create_csv_tracks_from_StitchNodes.py, БЕЗ ИЗМЕНЕНИЙ) передайте:")
    print(f"  --sigma-dir {combo_dir}")
    print(f"  --postfix   {postfix}")
    print(f"  --data-type {args.data_type}")
    print(f"  --years     {' '.join(str(y) for y in args.years)}")


if __name__ == "__main__":
    main()
