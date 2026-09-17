"""
Прогон refill_pbl_trop_frac.py сразу по ВСЕЙ сетке (eps x extr_type) и по
всем годам из --years, одним проходом.

Конфигурация сетки (EPS_VALUES, EXTR_TYPES, MIN_SAMPLES, PATH_INIT,
DATA_TYPE, REGION_NAME, SIGMA, combo_dir_path) берётся НАПРЯМУЮ из
6_grid_runner_TE_ERA5.py (импортом), а не задаётся заново здесь — единая
точка правды, чтобы этот скрипт не разъехался с реальными путями пайплайна
(ровно та же причина, по которой возникла исходная ошибка с сеткой
tropopause: две копии одной и той же логики, которые тихо разошлись).

Помните (см. докстринг run_stage_e_prime() в 6_grid_runner_TE_ERA5.py):
Stage E' считается ОДИН РАЗ на (eps, extr_type), на пуле size_filter=10 —
НЕ на каждый size_filter (10/25/49 в Stage E джойнятся из одного и того же
parquet). Поэтому "все конфигурации" здесь = все (eps, extr_type): для
EPS_VALUES=[1, 2] и EXTR_TYPES=["global", "local", "geom"] это 6 комбинаций,
а не 18.

Комбинации, для которых Stage E' ещё не отработал (нет помесячных
чекпоинтов), пропускаются с предупреждением — не валят весь прогон.

Все точки со всех комбинаций/лет объединяются в один DataFrame и
пересчитываются ОДНИМ проходом (один Pool, одна группировка по календарным
дням) — тогда воркеры реально переиспользуют LRU-кэш открытых blh/tropopause
файлов МЕЖДУ комбинациями (один и тот же месяц ERA5 нужен eps=1/global И
eps=2/geom одновременно), а не открывают одни и те же NetCDF заново под
каждую комбинацию по отдельности.

Пример запуска
--------------
python refill_pbl_trop_frac_grid.py --years 2010 --workers 8
python refill_pbl_trop_frac_grid.py --years 2010 2011 --eps 1 2 --extr-type global local geom --workers 8
"""

from __future__ import annotations

import argparse
import importlib
import multiprocessing as mp
import sys
from glob import glob
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# Импортируем как модули (не через subprocess) — те же приёмы, что в самом
# 6_grid_runner_TE_ERA5.py (importlib, т.к. "6_..." не годится для `import`).
grid_runner = importlib.import_module("6_grid_runner_TE_ERA5")
refill_mod = importlib.import_module("refill_pbl_trop_frac")


def _iter_combo_months(eps_values, extr_types, years):
    """Yield (eps, extr_type, year, month_file_path) для всех уже посчитанных
    Stage E'-чекпоинтов в сетке."""
    found_any_combo = False
    for eps in eps_values:
        for extr_type in extr_types:
            combo_dir_10 = grid_runner.combo_dir_path(eps, size_filter=10, extr_type=extr_type)
            for year in years:
                monthly_dir = combo_dir_10 / f"params_by_node_{year}_monthly"
                month_files = sorted(glob(str(monthly_dir / "*.parquet")))
                if not month_files:
                    print(f"[skip] eps={eps} extr_type={extr_type} year={year}: "
                          f"нет чекпоинтов в {monthly_dir} (Stage E' ещё не считался?)")
                    continue
                found_any_combo = True
                for month_file in month_files:
                    yield eps, extr_type, year, Path(month_file)
    if not found_any_combo:
        print("Ни для одной (eps, extr_type, year) не нашлось чекпоинтов Stage E'.")


def main():
    parser = argparse.ArgumentParser(
        description="Пересчёт pbl_trop_frac по всей сетке (eps x extr_type) одним проходом"
    )
    parser.add_argument("--years", type=int, nargs="+", required=True)
    parser.add_argument("--eps", type=int, nargs="+", default=grid_runner.EPS_VALUES)
    parser.add_argument("--extr-type", dest="extr_types", nargs="+", default=grid_runner.EXTR_TYPES)
    parser.add_argument("--add-params-module", default=grid_runner.ADD_PARAMS_MODULE_PATH)
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument("--max-open-datasets", type=int, default=8)
    parser.add_argument("--radius-multiplier", type=float, default=4.0)
    args = parser.parse_args()

    combos = list(_iter_combo_months(args.eps, args.extr_types, args.years))
    if not combos:
        return

    frames = []
    for eps, extr_type, year, month_file in combos:
        df = pd.read_parquet(month_file)
        df["_combo_eps"] = eps
        df["_combo_extr_type"] = extr_type
        df["_combo_year"] = year
        df["_combo_month_file"] = str(month_file)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    n_combos = len({(c[0], c[1]) for c in combos})
    n_before = int(combined["pbl_trop_frac"].notna().sum())
    print(f"Всего строк по сетке: {len(combined)} (из {len(combos)} месячных чекпоинтов, "
          f"{n_combos} комбинаций eps/extr_type); pbl_trop_frac непустых сейчас: {n_before}")

    add_params_module = refill_mod.load_add_params_module(args.add_params_module)
    data_paths = add_params_module.get_data_paths()

    combined = refill_mod.refill_pbl_trop_frac(
        combined, args.add_params_module, data_paths,
        n_workers=args.workers, max_open_datasets=args.max_open_datasets,
        radius_multiplier=args.radius_multiplier,
    )

    n_after = int(combined["pbl_trop_frac"].notna().sum())
    print(f"pbl_trop_frac непустых после пересчёта: {n_after} из {len(combined)}")

    # Разбираем обратно по месячным чекпоинтам и перезаписываем их атомарно.
    group_cols = ["_combo_eps", "_combo_extr_type", "_combo_year", "_combo_month_file"]
    for (eps, extr_type, year, month_file_str), part in combined.groupby(group_cols):
        part = part.drop(columns=group_cols)
        refill_mod._atomic_write_parquet(part, Path(month_file_str))
        print(f"  записан {month_file_str} ({len(part)} строк)")

    # Пересобираем итоговый params_by_node_{year}.parquet для каждой (eps, extr_type, year).
    for eps in args.eps:
        for extr_type in args.extr_types:
            combo_dir_10 = grid_runner.combo_dir_path(eps, size_filter=10, extr_type=extr_type)
            for year in args.years:
                monthly_dir = combo_dir_10 / f"params_by_node_{year}_monthly"
                output_parquet = combo_dir_10 / f"params_by_node_{year}.parquet"
                month_files = sorted(glob(str(monthly_dir / "*.parquet")))
                if not month_files:
                    continue
                year_df = pd.concat([pd.read_parquet(f) for f in month_files], ignore_index=True)
                refill_mod._atomic_write_parquet(year_df, output_parquet)
                print(f"[итог] {output_parquet} пересобран ({len(year_df)} строк)")

    print("Готово.")


if __name__ == "__main__":
    main()
