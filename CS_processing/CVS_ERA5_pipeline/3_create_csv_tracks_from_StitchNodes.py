"""
Конвертация текстового вывода StitchNodes (TempestExtremes) в отдельные CSV
на трек: NNNNNN_track_YYYY-MM-DDTHH.csv.

Написан по образцу вашего черновика add_params_to_TE_tracks_ERA5.ipynb
(ячейки с huracanpy.load(..., source="tempestextremes") + разбивка по
track_id) — сама загрузка/разбивка там уже рабочая и проверенная, здесь она
только вынесена в отдельный параметризуемый .py-скрипт (без сломанной
ячейки add_param — быстрый расчёт параметров делает отдельно
add_params_v2_fast_2026-08-24.py, это Stage E, не этот скрипт).

Место в пайплайне
------------------
Stage C (2_run_StitchNodes_by_year.py) кладёт годовой txt в:
    {sigma_dir}/Tracks_R2D_txt_files{postfix}/{data_type}_TC_tracks_{year}.txt
Этот скрипт (Stage D) читает этот txt и раскладывает его на CSV по трекам в
СОСЕДНЮЮ папку (тот же sigma_dir, тот же postfix, только префикс другой):
    {sigma_dir}/csv_Tracks{postfix}/NNNNNN_track_YYYY-MM-DDTHH.csv
— это ровно то место, которое ожидает add_params_v2_fast_2026-08-24.py
(TRACKS_PATH) и которое мы уже использовали в get_max_crit_df_for_ocean_TE_ERA5.py.

Формат колонок на выходе (проверено на вашем черновике-ноутбуке, на
реальном ERA5_TC_tracks_2010.txt): record, track_id, i, j, lon, lat, rad,
r2d, time — то есть 'r2d' и 'i'/'j', как и предполагалось в адаптированных
скриптах кластеризации.

2026-08-31: ускорение (fast=True, по умолчанию) — см. докстринг convert_year()
ниже. Формат/имена выходных файлов НЕ меняются, только то, как быстро они
получаются — существующие потребители (add_params_v2_fast_2026-08-24.py,
Stage E, get_max_crit_df_for_ocean_TE_ERA5.py, get_clusters_with_metrics_TE_ERA5.py)
не нужно трогать.

Пример запуска
--------------
python 3_create_csv_tracks_from_StitchNodes.py \
    --sigma-dir /storage/thalassa/users/vkoshkina/data/TempestExtremes/ERA5/R2D_ERA5_NA_for_TC_850hPa_sigma_2 \
    --postfix _range_1_5_18h_12h_2010_25points_global \
    --data-type ERA5 \
    --years 2010
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import huracanpy


def convert_year(
    txt_file: Path,
    output_folder: Path,
    variable_names: list[str],
    fast: bool = True,
) -> int:
    """Читает один годовой txt StitchNodes и сохраняет по CSV на трек.

    Возвращает число сохранённых треков (0, если файл не найден/пуст).

    2026-08-31: почему это было медленно и что изменилось (fast=True, по
    умолчанию)
    ------------------------------------------------------------------------
    Старый код (fast=False, оставлен ниже как --legacy-loop / для сверки)
    вызывал `tracks.hrcn.sel_id(track_id)` В ЦИКЛЕ по каждому уникальному
    track_id — это заново фильтрует ВЕСЬ Dataset на каждой итерации. На
    58118 треках (реальный прогон eps=1/sf=10/global/2010, ~8760 часовых
    срезов) это O(число_треков x число_точек) и давало ~16 треков/сек — то
    есть ~55 минут на один год ОДНОГО листа сетки, хотя сам StitchNodes
    (Stage C) отрабатывает за секунды. При переборе сетки eps x size_filter x
    extr_type x maxgap x mintime x prioritize это умножается на число листьев.

    fast=True делает то же самое (тот же формат, те же файлы), но за ОДИН
    проход: `tracks.to_dataframe()` вызывается один раз на весь Dataset
    (векторизованно, без пересканирования на каждый трек), затем разбивка по
    трекам — это `pandas.DataFrame.groupby('track_id')`, которая проходит по
    уже готовому DataFrame один раз, а не пересчитывает выборку для каждого
    трека заново.

    Внимание — не проверено на реальном huracanpy (в песочнице его нет)
    -----------------------------------------------------------------------
    `sel_id()` — метод accessor'а `.hrcn` (huracanpy), его реализация здесь не
    видна; предполагается, что это чистая фильтрация по track_id без побочных
    пересчётов, и что `Dataset.to_dataframe()` целиком даёт те же колонки/
    значения, что `sel_id(track_id).to_dataframe()` по кусочкам — это разумное
    предположение (сама сборка Dataset не зависит от id), но НЕ проверено
    прогоном. Перед тем как полагаться на fast=True для боевой сетки —
    сравните на одном небольшом (уже готовом) txt старый и новый способ:

        python -c "
        from pathlib import Path
        import importlib
        m = importlib.import_module('3_create_csv_tracks_from_StitchNodes')
        m.convert_year(Path('.../ERA5_TC_tracks_2010.txt'), Path('/tmp/fast'), ['rad','r2d','wspd'], fast=True)
        m.convert_year(Path('.../ERA5_TC_tracks_2010.txt'), Path('/tmp/legacy'), ['rad','r2d','wspd'], fast=False)
        "
        diff -rq /tmp/fast /tmp/legacy   # пусто = идентично

    Если diff пуст на паре реальных файлов — можно спокойно доверять fast=True
    для всей сетки.

    2026-08-31: атомарная публикация папки (важно для перезапуска после kill)
    ---------------------------------------------------------------------------
    Раньше файлы писались НАПРЯМУЮ в output_folder по одному в цикле. Если
    процесс убить посреди конвертации (типичная ситуация - именно этот шаг
    самый долгий), в output_folder оставалась ЧАСТЬ треков (например, 3275 из
    58118). Проверка "готово ли" в 6_grid_runner_TE_ERA5.py (`_dir_has_files`)
    смотрит только "непустая ли папка" - на такую недописанную папку она бы
    ответила "да, готово" и на следующем запуске Stage D молча пропустился бы,
    оставив навсегда усечённый набор треков без единой ошибки или предупреждения.
    Тот же класс проблемы, что уже был решён для Stage B/Stage C (см. их
    докстринги про tmp-файл + os.replace/rename) - здесь то же самое, но на
    уровне ЦЕЛОЙ ПАПКИ: пишем в output_folder.partial.pid{os.getpid()} и
    переименовываем в output_folder ТОЛЬКО если весь год отконвертирован без
    исключений. Если output_folder уже существует и не пуста - конвертация уже
    была опубликована атомарно раньше, повторно делать нечего.
    """
    if not txt_file.exists():
        print(f"Файл не найден, пропуск: {txt_file}")
        return 0

    if output_folder.is_dir() and any(output_folder.iterdir()):
        print(f"{output_folder} уже не пуста (опубликована ранее) - пропуск")
        return len(list(output_folder.glob("*.csv")))

    tracks = huracanpy.load(
        str(txt_file),
        source="tempestextremes",
        variable_names=variable_names,
    )

    if len(tracks.record) == 0:
        print(f"Пустой файл (0 точек), пропуск: {txt_file}")
        return 0

    tmp_folder = output_folder.with_name(output_folder.name + f".partial.pid{os.getpid()}")
    if tmp_folder.exists():
        shutil.rmtree(tmp_folder)  # огрызок от предыдущего убитого прогона с тем же pid (маловероятно, но не мешаем)
    tmp_folder.mkdir(parents=True)

    if fast:
        n_saved = _convert_year_fast(tracks, tmp_folder, txt_file.name)
    else:
        n_saved = _convert_year_legacy(tracks, tmp_folder, txt_file.name)

    # Атомарная публикация - только если весь год успешно отконвертирован
    # (исключение из _convert_year_fast/_legacy выше просто пробросится наружу,
    # tmp_folder останется на диске как частичный результат для диагностики,
    # output_folder не будет создана - следующий запуск пересоберёт год заново).
    tmp_folder.rename(output_folder)
    return n_saved


def _convert_year_fast(tracks, output_folder: Path, txt_name: str) -> int:
    df = tracks.to_dataframe().reset_index(drop=True)
    n_saved = 0
    groups = df.groupby("track_id", sort=False)
    for track_id, track_df in tqdm(groups, total=groups.ngroups, desc=f"Saving tracks from {txt_name}"):
        start_time = pd.to_datetime(track_df["time"].min())
        start_datetime = start_time.strftime('%Y-%m-%dT%H')
        filename = output_folder / f'{int(track_id):06d}_track_{start_datetime}.csv'
        track_df.reset_index(drop=True).to_csv(filename, index=False)
        n_saved += 1
    return n_saved


def _convert_year_legacy(tracks, output_folder: Path, txt_name: str) -> int:
    """Старый, медленный, но проверенный на черновике способ — см. докстринг
    convert_year() выше про O(N^2) и как сверить с fast=True."""
    unique_tracks = np.unique(tracks.track_id.values)
    for track_id in tqdm(unique_tracks, desc=f"Saving tracks from {txt_name} (legacy loop)"):
        track_data = tracks.hrcn.sel_id(track_id)
        start_time = pd.to_datetime(track_data.time.min().values)
        start_datetime = start_time.strftime('%Y-%m-%dT%H')
        filename = output_folder / f'{int(track_id):06d}_track_{start_datetime}.csv'
        track_df = track_data.to_dataframe().reset_index(drop=True)
        track_df.to_csv(filename, index=False)
    return len(unique_tracks)


def main():
    parser = argparse.ArgumentParser(
        description="StitchNodes txt -> CSV на трек (Stage D пайплайна TE-трекинга)"
    )
    parser.add_argument(
        "--sigma-dir", required=True,
        help="Папка конфигурации, например .../R2D_ERA5_NA_for_TC_850hPa_sigma_2 "
             "(родитель и для Tracks_R2D_txt_files{postfix}, и для csv_Tracks{postfix})",
    )
    parser.add_argument(
        "--postfix", required=True,
        help="Тот же postfix, что использован в 2_run_StitchNodes_by_year.py, "
             "например _range_1_5_18h_12h_2010_25points_global "
             "(range/mintime/maxgap/year/size_filter/extr_type[/prioritize])",
    )
    parser.add_argument("--data-type", default="ERA5", help="Как в имени файла: {data_type}_TC_tracks_{year}.txt")
    parser.add_argument("--years", type=int, nargs="+", default=[2010])
    parser.add_argument(
        "--variable-names", nargs="+", default=["rad", "r2d", "wspd"],
        help="Имена для доп. полей StitchNodes (после lon,lat), по порядку --in_fmt "
             "в 2_run_StitchNodes_by_year.py; сейчас там \"lon,lat,rad,r2d,wind\", "
             "поэтому порядок значений должен быть rad,r2d,<что-то>",
    )
    parser.add_argument(
        "--legacy-loop", action="store_true",
        help="Старый способ (tracks.hrcn.sel_id() в цикле по каждому треку) - медленно, "
             "но так было до 2026-08-31. Использовать только для сверки с fast=True "
             "(см. докстринг convert_year) или как аварийный откат.",
    )
    args = parser.parse_args()

    sigma_dir = Path(args.sigma_dir)
    tracks_txt_dir = sigma_dir / f"Tracks_R2D_txt_files{args.postfix}"
    csv_tracks_dir = sigma_dir / f"csv_Tracks{args.postfix}"

    total_saved = 0
    for year in args.years:
        txt_file = tracks_txt_dir / f"{args.data_type}_TC_tracks_{year}.txt"
        n_saved = convert_year(txt_file, csv_tracks_dir, args.variable_names, fast=not args.legacy_loop)
        total_saved += n_saved
        print(f"{year}: сохранено {n_saved} треков в {csv_tracks_dir}")

    print(f"\nВсего сохранено {total_saved} треков в {csv_tracks_dir}")


if __name__ == "__main__":
    main()
