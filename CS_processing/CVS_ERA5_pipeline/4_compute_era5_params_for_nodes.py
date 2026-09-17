"""
Stage E' — расчёт ERA5-параметров ОДИН РАЗ на весь пул узлов (Nodes) одной
Stage-B комбинации (eps x size_filter x extr_type), а не на каждую из до 40
комбинаций StitchNodes (maxgap x mintime x prioritize) в отдельности.

Почему так можно: сами ERA5-параметры (SLP_diff_cent_95, U10_mean, ...,
RAIN_HOURLY_sum) зависят только от (time, lat, lon, rad) конкретного узла —
StitchNodes лишь группирует уже существующие узлы в треки по-разному, не
меняя ни одну характеристику самого узла. Поэтому параметры можно посчитать
на входном пуле узлов (том самом txt, что идёт В StitchNodes), а потом при
сборке треков для любой комбинации maxgap/mintime/prioritize просто
подставлять готовое значение по ключу (time, lon_idx, lat_idx) —
см. 5_join_params_into_tracks.py (Stage E).

Расчётное ядро (ERA5DataLoader, SpatialAggregatorERA5, CycloneProcessorERA5,
RESULT_PARAMS, батчинг по дням с persistent multiprocessing.Pool) — это без
изменений add_params_v2_fast_2026-08-24.py, просто импортированный по пути
(имя файла с дефисами и датой нельзя импортировать как обычный модуль).
Единственное, что меняется — что подаётся на вход: не CSV на трек, а сырой
пул узлов (Nodes) за год, до StitchNodes.

Пример запуска
--------------
python 4_compute_era5_params_for_nodes.py \
    --nodes-dir /storage/thalassa/users/vkoshkina/data/TempestExtremes/ERA5/R2D_ERA5_NA_for_TC_850hPa_sigma_2/01-04-25_global/R2D_txt_files_2010 \
    --year 2010 \
    --output /storage/thalassa/users/vkoshkina/data/TempestExtremes/ERA5/R2D_ERA5_NA_for_TC_850hPa_sigma_2/01-04-25_global/params_by_node_2010.parquet \
    --workers 8
"""

from __future__ import annotations

import argparse
import importlib.util
import multiprocessing as mp
import os
import re
import sys
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm


# ============================================================================
# Импорт расчётного ядра из add_params_v2_fast_2026-08-24.py по пути файла
# (имя модуля с дефисами/датой не годится для обычного import)
# ============================================================================

DEFAULT_ADD_PARAMS_PATH = (
    "/storage/thalassa/users/vkoshkina/scripts/ERA5/add_params_v2_fast_2026-08-24.py"
)

ADD_PARAMS_MODULE_NAME = "add_params_v2_fast"


def load_add_params_module(path: str):
    spec = importlib.util.spec_from_file_location(ADD_PARAMS_MODULE_NAME, path)
    module = importlib.util.module_from_spec(spec)
    # ВАЖНО: регистрируем в sys.modules ДО exec_module. Без этого модуль
    # существует только как "осиротевший" объект - его не найти повторным
    # import_module("add_params_v2_fast") в этом же процессе (нужно, например,
    # если pickle попытается проверить его происхождение). См. докстринг
    # _spawn_worker_init ниже про то, почему этого одного шага всё равно не
    # хватает для spawn-дочерних процессов и что сделано вместо этого.
    sys.modules[ADD_PARAMS_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


# ============================================================================
# Парсер формата Nodes-txt (тот, что пишут 1_create_Nodes_from_DBSCAN.py и
# 1_create_Nodes_from_DBSCAN_geom.py, он же вход StitchNodes):
#
#   {year}\t{month}\t{day}\t{n_extrema}\t{hour}
#   \t{lon_idx}\t{lat_idx}\t{lon}\t{lat}\t{rad}\t{crit}\t{wspd}
#   ... (n_extrema строк) ...
# ============================================================================

def read_nodes_txt_file(path: Path) -> List[dict]:
    points = []
    with open(path, "r") as f:
        current_time = None
        for line in f:
            if not line.strip():
                continue
            if not line.startswith("\t"):
                # строка метаданных часа: year month day n_extrema hour
                year, month, day, _n, hour = line.strip().split()
                current_time = pd.Timestamp(
                    year=int(year), month=int(month), day=int(day), hour=int(hour)
                )
            else:
                lon_idx, lat_idx, lon, lat, rad, crit, wspd = line.strip().split("\t")
                points.append({
                    "time": current_time,
                    "lon_idx": int(lon_idx),
                    "lat_idx": int(lat_idx),
                    "lon": float(lon),
                    "lat": float(lat),
                    "rad": float(rad),
                    "crit": float(crit),
                    "wspd": float(wspd),
                })
    return points


def load_all_nodes(nodes_dir: Path, year: int) -> pd.DataFrame:
    txt_files = sorted(glob(str(nodes_dir / f"*_{year}-*.txt")))
    if not txt_files:
        raise FileNotFoundError(f"Не найдено ни одного txt-файла узлов в {nodes_dir} за {year}")

    all_points = []
    for txt_file in tqdm(txt_files, desc="Чтение Nodes txt"):
        all_points.extend(read_nodes_txt_file(Path(txt_file)))

    df = pd.DataFrame(all_points)
    df["point_id"] = range(len(df))
    return df


# ============================================================================
# Обёртки для Pool под spawn-контекст.
#
# ПОЧЕМУ ЭТО НУЖНО: add_params_v2_fast_2026-08-24.py сам загружается
# динамически (spec_from_file_location — имя файла с дефисами/датой нельзя
# импортировать обычным import). Значит функции _worker_init/_worker_process_day
# ВНУТРИ него физически непиклябельны под spawn: при старте дочернего
# процесса pickle должен суметь заново сделать import_module("add_params_v2_fast")
# УЖЕ В ДОЧЕРНЕМ процессе — а это свежий интерпретатор, который ничего не
# знает о нашем динамическом импорте в родителе (регистрация в sys.modules
# родителя в дочерний spawn-процесс не передаётся). Отсюда и падение:
# "Can't pickle <function _worker_init ...>: import of module
# 'add_params_v2_fast' failed".
#
# Решение: Pool получает initializer/task-функцию НЕ из динамического модуля
# напрямую, а вот эти две обёртки ниже — они лежат в этом самом файле,
# который импортируется штатно (реальный файл на sys.path), значит
# пиклябелен нормально что при запуске напрямую (python
# 4_compute_era5_params_for_nodes.py — тогда это __main__, и multiprocessing
# сам умеет переисполнять __main__ в потомке), что при импорте из
# 6_grid_runner_TE_ERA5.py (тогда это обычный модуль "4_compute_era5_params_for_nodes",
# который потомок находит через унаследованный sys.path). Каждый воркер сам
# подгружает add_params_v2_fast (по пути из initargs) один раз при старте.
# ============================================================================

_worker_add_params_module = None  # per-process global, выставляется _spawn_worker_init


def _spawn_worker_init(add_params_module_path: str, data_paths: dict, radius_multiplier: float, max_open_datasets: int) -> None:
    global _worker_add_params_module
    _worker_add_params_module = load_add_params_module(add_params_module_path)
    _worker_add_params_module._worker_init(data_paths, radius_multiplier, max_open_datasets)


def _spawn_worker_process_day(day_points):
    return _worker_add_params_module._worker_process_day(day_points)


# ============================================================================
# Расчёт (батчинг по календарным суткам + persistent Pool — как в v2 fast)
# ============================================================================

def compute_params_for_nodes(
    nodes_df: pd.DataFrame,
    add_params_module,
    add_params_module_path: str,
    n_workers: int = 4,
    max_open_datasets: int = 8,
) -> pd.DataFrame:
    RESULT_PARAMS = add_params_module.RESULT_PARAMS
    CycloneProcessorERA5 = add_params_module.CycloneProcessorERA5
    data_paths = add_params_module.get_data_paths()

    day_buckets: Dict[Tuple[int, int, int], list] = defaultdict(list)
    for row in nodes_df.itertuples():
        t = row.time
        day_buckets[(t.year, t.month, t.day)].append(
            (t, row.point_id, 0, row.lat, row.lon, row.rad)
        )
    day_batches = list(day_buckets.values())

    ctx = mp.get_context("spawn")
    results: List[Tuple[int, int, dict]] = []

    if n_workers > 1:
        pool = ctx.Pool(
            processes=n_workers,
            initializer=_spawn_worker_init,
            initargs=(add_params_module_path, data_paths, 4.0, max_open_datasets),
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
        processor = CycloneProcessorERA5(data_paths, radius_multiplier=4.0, max_open_datasets=max_open_datasets)
        for db in tqdm(day_batches, desc="Дни"):
            results.extend(processor.process_day_points(db))

    params_by_point_id = {point_id: res for point_id, _row_idx, res in results}

    params_df = pd.DataFrame.from_dict(params_by_point_id, orient="index")
    params_df.index.name = "point_id"
    params_df = params_df.reset_index()

    out = nodes_df.merge(params_df, on="point_id", how="left")
    return out[["time", "lon_idx", "lat_idx", "lon", "lat", "rad", "crit", "wspd"] + RESULT_PARAMS]


# ============================================================================
# Помесячные чекпоинты (2026-08-30)
#
# compute_params_for_nodes() выше копит результаты по ВСЕМ ~365 дням года в
# памяти и пишет один parquet только в самом конце — если процесс упадёт
# (OOM/kill/сбой на каком-то одном дне), весь прогресс за год теряется, и
# повторный запуск считает заново все дни, включая уже посчитанные.
#
# compute_params_for_nodes_monthly() ниже — та же логика, но по месяцам:
# на каждый месяц отдельный вызов compute_params_for_nodes() (Pool
# создаётся заново на каждый месяц - да, это лишние ~12 переоткрытий ERA5
# датасетов в _spawn_worker_init за год вместо одного, но на фоне ~10ч
# счёта это малая цена) и atomic-запись результата в monthly_dir. Если
# упадёт на середине года — при повторном запуске уже посчитанные месяцы
# просто читаются с диска (pd.read_parquet), пересчитывается только
# оставшееся. Итоговый годовой DataFrame собирается конкатенацией месяцев
# и возвращается — вызывающий код (run_stage_e_prime в
# 6_grid_runner_TE_ERA5.py, main() ниже) сохраняет его как обычно, тем же
# atomic tmp+rename, каким уже сохраняется финальный params_by_node_{year}.parquet.
# ============================================================================

def compute_params_for_nodes_monthly(
    nodes_df: pd.DataFrame,
    add_params_module,
    add_params_module_path: str,
    monthly_dir: Path,
    year: int,
    n_workers: int = 4,
    max_open_datasets: int = 8,
) -> pd.DataFrame:
    monthly_dir = Path(monthly_dir)
    monthly_dir.mkdir(parents=True, exist_ok=True)

    months_present = sorted(nodes_df["time"].dt.month.unique())
    monthly_results = []

    for month in months_present:
        month_parquet = monthly_dir / f"{year}-{month:02d}.parquet"

        if month_parquet.exists():
            print(f"[Stage E'] месяц {year}-{month:02d} уже посчитан ({month_parquet}) - пропуск")
            monthly_results.append(pd.read_parquet(month_parquet))
            continue

        month_nodes_df = nodes_df[nodes_df["time"].dt.month == month].reset_index(drop=True)
        print(f"[Stage E'] месяц {year}-{month:02d}: {len(month_nodes_df)} узлов - считаем...")

        month_result_df = compute_params_for_nodes(
            month_nodes_df, add_params_module, add_params_module_path,
            n_workers=n_workers, max_open_datasets=max_open_datasets,
        )

        tmp_path = month_parquet.with_suffix(month_parquet.suffix + f".partial.pid{os.getpid()}")
        try:
            month_result_df.to_parquet(tmp_path, index=False)
            os.replace(tmp_path, month_parquet)  # публикуем месяц только после успешной записи целиком
        except BaseException:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

        print(f"[Stage E'] месяц {year}-{month:02d} сохранён: {month_parquet}")
        monthly_results.append(month_result_df)

    return pd.concat(monthly_results, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(
        description="Stage E': расчёт ERA5-параметров один раз на весь пул узлов (кэш по time,lon_idx,lat_idx)"
    )
    parser.add_argument("--nodes-dir", required=True, help="Папка R2D_txt_files_{year} с помесячными Nodes-файлами")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--output", required=True, help="Путь к выходному parquet (params_by_node_{year}.parquet)")
    parser.add_argument("--add-params-module", default=DEFAULT_ADD_PARAMS_PATH,
                         help="Путь к add_params_v2_fast_2026-08-24.py (расчётное ядро)")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument("--max-open-datasets", type=int, default=8)
    parser.add_argument("--monthly-dir", default=None,
                         help="Папка для помесячных чекпоинтов parquet (по умолчанию "
                              "<output_dir>/<output_stem>_monthly/)")
    args = parser.parse_args()

    add_params_module = load_add_params_module(args.add_params_module)

    nodes_df = load_all_nodes(Path(args.nodes_dir), args.year)
    print(f"Загружено {len(nodes_df)} узлов за {args.year} год из {args.nodes_dir}")

    output_path = Path(args.output)
    monthly_dir = Path(args.monthly_dir) if args.monthly_dir else output_path.parent / f"{output_path.stem}_monthly"

    result_df = compute_params_for_nodes_monthly(
        nodes_df, add_params_module, args.add_params_module, monthly_dir, args.year,
        n_workers=args.workers, max_open_datasets=args.max_open_datasets,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_suffix(output_path.suffix + f".partial.pid{os.getpid()}")
    try:
        result_df.to_parquet(tmp_output_path, index=False)
        os.replace(tmp_output_path, output_path)
    except BaseException:
        if tmp_output_path.exists():
            tmp_output_path.unlink()
        raise

    print(f"Сохранено {len(result_df)} узлов с параметрами в {output_path}")
    n_nan = result_df[add_params_module.RESULT_PARAMS[0]].isna().sum()
    if n_nan:
        print(f"Внимание: у {n_nan} узлов не удалось посчитать параметры (нет данных ERA5 на этот момент?)")


if __name__ == "__main__":
    main()
