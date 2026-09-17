"""
Stage E — сборка финальных треков (аналог старого плоского hourly_data) из:
  1) Stage D: CSV на трек из csv_Tracks{postfix} (3_create_csv_tracks_from_StitchNodes.py,
     колонки как минимум record, track_id, i, j, lon, lat, rad, r2d, [wspd], time)
  2) Stage E': parquet-кэш параметров ERA5 на весь пул узлов, params_by_node_{year}.parquet
     (4_compute_era5_params_for_nodes.py, колонки time, lon_idx, lat_idx, ..., RESULT_PARAMS)

Джойн — по (time, i, j) трека == (time, lon_idx, lat_idx) узла. Это тот же
ключ, что и в Stage E' (см. её докстринг: точное совпадение по целочисленным
индексам сетки надёжнее, чем по float lon/lat после прохода через
TE-txt -> StitchNodes -> huracanpy). i/j в CSV трека — это те же lon_idx/lat_idx,
что писали create_Nodes_from_DBSCAN*.py в узел (первые два поля строки узла),
StitchNodes и huracanpy их не трогают, только группируют в треки.

Почему это отдельный скрипт, а не часть Stage D
--------------------------------------------------
Один и тот же parquet из Stage E' можно джойнить в треки ЛЮБОЙ комбинации
StitchNodes (maxgap x mintime x prioritize) на одном пуле узлов — параметры
одинаковые, треки — разные. Поэтому расчёт (Stage E', медленный) и джойн
(Stage E, быстрый merge) разделены: пересчитывать параметры при переборе
StitchNodes не нужно, только перечитывать один и тот же parquet.

Результат по схеме совместим с тем, что ожидают get_max_crit_df_for_ocean_TE_ERA5.py
и get_clusters_with_metrics_TE_ERA5.py (данные из monthly-подобных наборов CSV на
трек с i/j/r2d/rad + 12 параметров).

Пример запуска
--------------
python 5_join_params_into_tracks.py \
    --tracks-dir /storage/.../R2D_ERA5_NA_for_TC_850hPa_sigma_2/01-04-25_global/csv_Tracks_range_1_5_18h_12h_2010 \
    --params-parquet /storage/.../R2D_ERA5_NA_for_TC_850hPa_sigma_2/01-04-25_global/params_by_node_2010.parquet \
    --output-dir /storage/.../R2D_ERA5_NA_for_TC_850hPa_sigma_2/01-04-25_global/csv_Tracks_range_1_5_18h_12h_2010_params

2026-08-31: join_combo() ускорен (fast=True, по умолчанию) — см. её докстринг.
Формат/содержимое выходных CSV не меняется, только то, как быстро они
получаются (плюс попутно убран лишний повторный проход по каждому файлу
только ради подсчёта строк для диагностики — см. join_combo).
"""

from __future__ import annotations

import argparse
import os
import shutil
from glob import glob
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Поля узла, которые Stage E' (4_compute_era5_params_for_nodes.py) переносит из
# исходного Nodes-txt в parquet как есть (lon/lat/rad/crit/wspd) - они уже
# есть в CSV трека (под теми же или похожими именами, полученными через
# StitchNodes/huracanpy) и физически совпадают для одного и того же узла, см.
# докстринг Stage E'. Джойним из parquet только НОВОЕ - собственно 12
# ERA5-параметров, а не эти raw-поля, иначе в треке появляются дублирующие
# колонки (lon_node/rad_node/crit и т.п.).
_NODE_RAW_FIELDS = {"time", "lon_idx", "lat_idx", "lon", "lat", "rad", "crit", "wspd"}


def join_one_track(
    track_csv_path: Path,
    params_df: pd.DataFrame,
    output_path: Path,
    i_col: str = "i",
    j_col: str = "j",
    time_col: str = "time",
) -> int:
    """Джойнит параметры в один CSV трека. Возвращает число точек трека без
    найденного совпадения в params_df (для диагностики)."""
    track_df = pd.read_csv(track_csv_path)
    track_df[time_col] = pd.to_datetime(track_df[time_col])

    result_cols = [c for c in params_df.columns if c not in _NODE_RAW_FIELDS]
    join_cols = ["time", "lon_idx", "lat_idx"] + result_cols

    merged = track_df.merge(
        params_df[join_cols],
        left_on=[time_col, i_col, j_col],
        right_on=["time", "lon_idx", "lat_idx"],
        how="left",
    )
    # 'time' в left_on/right_on совпадает по имени -> pandas не дублирует эту
    # колонку (проверено отдельно); lon_idx/lat_idx из params_df остаются как
    # отдельные колонки (NaN там = нет совпадения в params_df) - используем их
    # для диагностики ДО удаления, затем убираем как дубли i/j трека.
    n_missing = int(merged["lon_idx"].isna().sum())

    merged = merged.drop(columns=["lon_idx", "lat_idx"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    return n_missing


def join_combo(
    tracks_dir: Path,
    params_parquet: Path,
    output_dir: Path,
    i_col: str = "i",
    j_col: str = "j",
    time_col: str = "time",
    fast: bool = True,
) -> None:
    """Джойнит ERA5-параметры во ВСЕ CSV-треки одного combo/postfix.

    2026-08-31: fast=True (по умолчанию) — что изменилось
    -----------------------------------------------------
    Старый код (fast=False, ниже — _join_combo_legacy) на каждый из N файлов
    треков делал: (1) открывал файл ЕЩЁ РАЗ только чтобы посчитать строки для
    диагностического сообщения (`sum(1 for _ in open(track_csv))`), (2) читал
    его снова через pd.read_csv, (3) делал отдельный pd.merge, (4) писал
    результат — то есть 2 открытия файла на чтение + N маленьких merge вместо
    одного большого. При N в десятки тысяч файлов (как в Stage D после
    ускорения — см. 3_create_csv_tracks_from_StitchNodes.py) накладные
    расходы на файловые операции и per-call overhead pandas.merge заметны.

    fast=True читает каждый файл РОВНО один раз, склеивает все треки в один
    DataFrame (с колонкой-меткой исходного файла), делает ОДИН merge на все
    точки сразу, затем возвращает результат обратно по файлам через
    groupby(source_file) - то же самое количество файлов на выходе, тот же
    формат и содержимое (см. тест сверки), просто один проход вместо N.

    2026-08-31: атомарная публикация output_dir (тот же приём, что и в
    convert_year() - Stage D, см. её докстринг) — пишем в
    output_dir.partial.pid{os.getpid()} и переименовываем в output_dir только
    если ВСЕ треки джойнились без исключений. Иначе, при kill посреди работы,
    6_grid_runner_TE_ERA5.py::_dir_has_files(final_dir) увидела бы частично
    заполненную final_dir и молча пропустила бы Stage E на следующем запуске.
    Если output_dir уже существует и не пуста - уже опубликовано раньше,
    ничего не делаем.
    """
    if output_dir.is_dir() and any(output_dir.iterdir()):
        print(f"{output_dir} уже не пуста (опубликована ранее) - пропуск")
        return

    params_df = pd.read_parquet(params_parquet)
    params_df["time"] = pd.to_datetime(params_df["time"])

    track_files = sorted(glob(str(tracks_dir / "*.csv")))
    if not track_files:
        print(f"CSV-треки не найдены в {tracks_dir}")
        return

    tmp_dir = output_dir.with_name(output_dir.name + f".partial.pid{os.getpid()}")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    if fast:
        total_missing, total_rows = _join_combo_fast(
            track_files, params_df, tmp_dir, i_col=i_col, j_col=j_col, time_col=time_col,
        )
    else:
        total_missing, total_rows = _join_combo_legacy(
            track_files, params_df, tmp_dir, i_col=i_col, j_col=j_col, time_col=time_col,
        )

    tmp_dir.rename(output_dir)  # атомарная публикация - только полностью собранный combo

    print(f"\nГотово: {len(track_files)} треков сохранено в {output_dir}")
    if total_missing:
        print(
            f"Внимание: {total_missing} из {total_rows} точек треков не нашли совпадения "
            f"в {params_parquet} по ключу (time, {i_col}, {j_col}) <-> (time, lon_idx, lat_idx). "
            "Возможные причины: params-parquet посчитан для другого пула узлов "
            "(другая комбинация eps/size_filter/extr_type), либо расхождение в точности "
            "time (huracanpy сдвинул час?) - стоит проверить вручную на паре точек."
        )


def _join_combo_fast(track_files, params_df, output_dir: Path, i_col: str, j_col: str, time_col: str):
    result_cols = [c for c in params_df.columns if c not in _NODE_RAW_FIELDS]
    join_cols = ["time", "lon_idx", "lat_idx"] + result_cols

    frames = []
    for f in tqdm(track_files, desc="Чтение CSV-треков"):
        track_df = pd.read_csv(f)
        track_df["_src_file"] = Path(f).name
        frames.append(track_df)
    all_tracks = pd.concat(frames, ignore_index=True)
    all_tracks[time_col] = pd.to_datetime(all_tracks[time_col])

    merged = all_tracks.merge(
        params_df[join_cols], left_on=[time_col, i_col, j_col],
        right_on=["time", "lon_idx", "lat_idx"], how="left",
    )
    total_missing = int(merged["lon_idx"].isna().sum())
    total_rows = len(merged)
    merged = merged.drop(columns=["lon_idx", "lat_idx"])

    for src_file, group in tqdm(merged.groupby("_src_file", sort=False), desc="Запись обогащённых CSV"):
        group.drop(columns=["_src_file"]).to_csv(output_dir / src_file, index=False)

    return total_missing, total_rows


def _join_combo_legacy(track_files, params_df, output_dir: Path, i_col: str, j_col: str, time_col: str):
    """Старый способ (файл на файл) - см. докстринг join_combo() про fast=True."""
    total_missing = 0
    total_rows = 0
    for track_csv in tqdm(track_files, desc="Джойн параметров в треки (legacy)"):
        track_csv = Path(track_csv)
        output_path = output_dir / track_csv.name
        n_rows = sum(1 for _ in open(track_csv)) - 1  # без заголовка, приблизительно
        n_missing = join_one_track(
            track_csv, params_df, output_path,
            i_col=i_col, j_col=j_col, time_col=time_col,
        )
        total_missing += n_missing
        total_rows += n_rows
    return total_missing, total_rows


def main():
    parser = argparse.ArgumentParser(
        description="Stage E: джойн параметров ERA5 (Stage E' parquet) в CSV-треки (Stage D) по (time, i, j)"
    )
    parser.add_argument("--tracks-dir", required=True, help="Папка csv_Tracks{postfix} (Stage D, CSV на трек)")
    parser.add_argument("--params-parquet", required=True, help="params_by_node_{year}.parquet (Stage E')")
    parser.add_argument("--output-dir", required=True, help="Куда сохранить обогащённые CSV на трек")
    parser.add_argument("--i-col", default="i", help="Колонка с lon_idx в CSV трека (по умолчанию 'i')")
    parser.add_argument("--j-col", default="j", help="Колонка с lat_idx в CSV трека (по умолчанию 'j')")
    parser.add_argument("--time-col", default="time")
    parser.add_argument(
        "--legacy-loop", action="store_true",
        help="Старый способ (файл-за-файлом: 2 чтения + отдельный merge на каждый трек) - "
             "медленнее, но так было до 2026-08-31. Для сверки с fast=True или как откат.",
    )
    args = parser.parse_args()

    join_combo(
        tracks_dir=Path(args.tracks_dir),
        params_parquet=Path(args.params_parquet),
        output_dir=Path(args.output_dir),
        i_col=args.i_col,
        j_col=args.j_col,
        time_col=args.time_col,
        fast=not args.legacy_loop,
    )


if __name__ == "__main__":
    main()
