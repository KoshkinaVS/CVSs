"""
Точечный пересчёт ТОЛЬКО pbl_trop_frac после починки бага с несовпадающей
сеткой tropopause (dyn_z) в add_params_v2_fast_2026-08-24.py (fix 2026-09-02).

Не трогает остальные 11 параметров из RESULT_PARAMS — грузит на каждый
timestamp только blh + tropopause (а не весь стек mslp/uv10m/t2/precip/
GRIB 850&500hPa/pv-omega), поэтому на порядок дешевле полного пересчёта
через 4_compute_era5_params_for_nodes.py.

ВАЖНО: пересчитывает pbl_trop_frac для ВСЕХ строк, а не только там, где
сейчас NaN. Баг с сеткой (индексы, посчитанные для сетки mslp/blh 0.25°
721x1440, применялись напрямую к массиву tropopause 0.3° 601x1201) мог
давать не только NaN, но и правдоподобные, но геометрически неверные
значения — там, где "чужой" индекс случайно попадал в границы массива
tropopause. Заполнение только NaN такие случаи не поймает, поэтому колонка
пересчитывается целиком.

Рассчитан на слой Stage E' (4_compute_era5_params_for_nodes.py) —
помесячные parquet-чекпоинты вида <output_stem>_monthly/{year}-{month:02d}.parquet
и/или итоговый params_by_node_{year}.parquet с колонками
["time", "lon_idx", "lat_idx", "lon", "lat", "rad", "crit", "wspd"] + RESULT_PARAMS.

Пример запуска
--------------
python refill_pbl_trop_frac.py \
    --monthly-dir /storage/thalassa/users/vkoshkina/data/TempestExtremes/ERA5/R2D_ERA5_NA_for_TC_850hPa_sigma_2/01-04-25_global/params_by_node_2010_monthly \
    --output /storage/thalassa/users/vkoshkina/data/TempestExtremes/ERA5/R2D_ERA5_NA_for_TC_850hPa_sigma_2/01-04-25_global/params_by_node_2010.parquet \
    --add-params-module /storage/thalassa/users/vkoshkina/scripts/ERA5/add_params_v2_fast_2026-08-24.py \
    --workers 8
"""

from __future__ import annotations

import argparse
import importlib.util
import multiprocessing as mp
import os
import sys
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

ADD_PARAMS_MODULE_NAME = "add_params_v2_fast"


def load_add_params_module(path: str):
    spec = importlib.util.spec_from_file_location(ADD_PARAMS_MODULE_NAME, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[ADD_PARAMS_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


# ============================================================================
# Воркеры под spawn (та же схема обёрток, что в 4_compute_era5_params_for_nodes.py:
# add_params_v2_fast грузится динамически по пути файла с дефисами/датой,
# поэтому функции внутри него непиклябельны под spawn напрямую — воркер сам
# подгружает модуль по пути из initargs один раз при старте).
# ============================================================================

_worker_module = None
_worker_state: Dict[str, object] = {}


def _spawn_worker_init(add_params_module_path: str, data_paths: dict, radius_multiplier: float, max_open_datasets: int):
    global _worker_module
    _worker_module = load_add_params_module(add_params_module_path)
    _worker_state["loader"] = _worker_module.ERA5DataLoader(data_paths, max_open_datasets=max_open_datasets)
    _worker_state["radius_multiplier"] = radius_multiplier


def _spawn_worker_process_day(day_points):
    """day_points: список (time, refill_point_id, lat, lon, rad)."""
    module = _worker_module
    loader = _worker_state["loader"]
    radius_multiplier = _worker_state["radius_multiplier"]

    AggregationMethod = module.AggregationMethod
    SpatialAggregatorERA5 = module.SpatialAggregatorERA5

    by_time: Dict[pd.Timestamp, list] = defaultdict(list)
    for time, point_id, lat, lon, rad in day_points:
        by_time[time].append((point_id, lat, lon, rad))

    out: List[Tuple[int, float]] = []

    for current_time, items in by_time.items():
        try:
            era5 = loader.load(current_time)
        except Exception:
            out.extend((point_id, np.nan) for point_id, *_ in items)
            continue

        blh = era5["blh"]
        tropopause = era5["tropopause"]

        if blh is None or tropopause is None:
            out.extend((point_id, np.nan) for point_id, *_ in items)
            continue

        blh_lat_name = "latitude" if "latitude" in blh.coords else "lat"
        blh_lon_name = "longitude" if "longitude" in blh.coords else "lon"
        blh_grid_lat = blh[blh_lat_name].values
        blh_grid_lon = blh[blh_lon_name].values

        trop_lat_name = "latitude" if "latitude" in tropopause.coords else "lat"
        trop_lon_name = "longitude" if "longitude" in tropopause.coords else "lon"
        trop_grid_lat = tropopause[trop_lat_name].values
        trop_grid_lon = tropopause[trop_lon_name].values

        for point_id, lat, lon, rad in items:
            radius_deg = float(rad) * radius_multiplier

            blh_agg = SpatialAggregatorERA5(
                center_lat=lat, center_lon=lon, radius_deg=radius_deg,
                grid_lat=blh_grid_lat, grid_lon=blh_grid_lon,
            )
            trop_agg = SpatialAggregatorERA5(
                center_lat=lat, center_lon=lon, radius_deg=radius_deg,
                grid_lat=trop_grid_lat, grid_lon=trop_grid_lon,
            )

            blh_val = blh_agg.aggregate(blh, AggregationMethod.MEDIAN)
            trop_val = trop_agg.aggregate(tropopause, AggregationMethod.MEDIAN)

            value = (
                blh_val * 1000.0 / trop_val
                if np.isfinite(trop_val) and trop_val > 0
                else np.nan
            )
            out.append((point_id, value))

    return out


def refill_pbl_trop_frac(
    df: pd.DataFrame,
    add_params_module_path: str,
    data_paths: dict,
    n_workers: int,
    max_open_datasets: int,
    radius_multiplier: float = 4.0,
) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    df["refill_point_id"] = range(len(df))

    day_buckets: Dict[Tuple[int, int, int], list] = defaultdict(list)
    for row in df.itertuples():
        t = row.time
        day_buckets[(t.year, t.month, t.day)].append(
            (t, row.refill_point_id, row.lat, row.lon, row.rad)
        )
    day_batches = list(day_buckets.values())

    ctx = mp.get_context("spawn")
    results: List[Tuple[int, float]] = []

    if n_workers > 1:
        pool = ctx.Pool(
            processes=n_workers,
            initializer=_spawn_worker_init,
            initargs=(add_params_module_path, data_paths, radius_multiplier, max_open_datasets),
        )
        try:
            for r in tqdm(
                pool.imap_unordered(_spawn_worker_process_day, day_batches),
                total=len(day_batches),
                desc="Дни",
            ):
                results.extend(r)
        finally:
            pool.close()
            pool.join()
    else:
        _spawn_worker_init(add_params_module_path, data_paths, radius_multiplier, max_open_datasets)
        for db in tqdm(day_batches, desc="Дни"):
            results.extend(_spawn_worker_process_day(db))

    value_by_id = dict(results)
    df["pbl_trop_frac"] = df["refill_point_id"].map(value_by_id)
    return df.drop(columns=["refill_point_id"])


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + f".partial.pid{os.getpid()}")
    try:
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Точечный пересчёт только pbl_trop_frac после починки сетки tropopause"
    )
    parser.add_argument("--monthly-dir", required=True,
                         help="Папка с помесячными parquet из Stage E' (params_by_node_{year}_monthly)")
    parser.add_argument("--output", required=True,
                         help="Путь к финальному parquet (params_by_node_{year}.parquet) — будет пересобран и перезаписан")
    parser.add_argument("--add-params-module", required=True,
                         help="Путь к (уже исправленному) add_params_v2_fast_2026-08-24.py")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument("--max-open-datasets", type=int, default=8)
    parser.add_argument("--radius-multiplier", type=float, default=4.0)
    args = parser.parse_args()

    add_params_module = load_add_params_module(args.add_params_module)
    data_paths = add_params_module.get_data_paths()

    monthly_dir = Path(args.monthly_dir)
    month_files = sorted(glob(str(monthly_dir / "*.parquet")))
    if not month_files:
        raise FileNotFoundError(f"Не найдено parquet-файлов в {monthly_dir}")

    all_months = []
    for month_file in month_files:
        print(f"[refill] {month_file}")
        df = pd.read_parquet(month_file)
        n_before = int(df["pbl_trop_frac"].notna().sum())

        df = refill_pbl_trop_frac(
            df, args.add_params_module, data_paths,
            n_workers=args.workers, max_open_datasets=args.max_open_datasets,
            radius_multiplier=args.radius_multiplier,
        )

        n_after = int(df["pbl_trop_frac"].notna().sum())
        print(f"  pbl_trop_frac: {n_before} -> {n_after} непустых из {len(df)}")

        _atomic_write_parquet(df, Path(month_file))
        all_months.append(df)

    result_df = pd.concat(all_months, ignore_index=True)
    _atomic_write_parquet(result_df, Path(args.output))
    print(f"Готово: {args.output}")


if __name__ == "__main__":
    main()
