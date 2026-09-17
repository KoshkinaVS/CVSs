"""
Stage C — запуск StitchNodes (TempestExtremes) по годам, по одной комбинации
(eps, size_filter, extr_type) узлов Stage B, для одной комбинации параметров
трекинга (search_range, mintime, maxgap, prioritize).

2026-08-30: обновлено под новую вложенную схему папок, которую уже пишут
1_create_Nodes_from_DBSCAN.py / 1_create_Nodes_from_DBSCAN_geom.py (Stage B):

    {sigma_dir}/{eps:02d}-{min_samples:02d}-{size_filter:02d}_{extr_type}/
        R2D_txt_files_{year}/                              <- узлы (Stage B, уже готово)
        R2D_txt_files_list_for_tracking/                    <- сюда этот скрипт кладёт --in_list
        Tracks_R2D_txt_files{postfix}/                       <- сюда этот скрипт кладёт треки (--out)
            {data_type}_TC_tracks_{year}.txt

где sigma_dir = {path_init}/{data_type}/R2D_{data_type}_{region_name}_sigma_{sigma},
extr_type ∈ {'global', 'local', 'geom'} (без подчёркивания — как в имени
combo-папки у Stage B), postfix кодирует именно StitchNodes-параметры:
    _range_{search_range}_{mintime}h_{maxgap}h[_prioritize_r2d]

Название 'Tracks_R2D_txt_files{postfix}' совпадает с тем, что уже ожидает
Stage D (3_create_csv_tracks_from_StitchNodes.py, флаг --postfix) — то есть для
перехода к Stage D достаточно передать туда:
    --sigma-dir {sigma_dir}/{eps:02d}-{min_samples:02d}-{size_filter:02d}_{extr_type}
    --postfix   {postfix}
(имя параметра "--sigma-dir" там осталось прежним, но по факту туда нужно
передавать combo_dir, а не sigma_dir — раньше combo и sigma совпадали).

Из старого скрипта убраны ветки для LoRes/SMP/GPN/HiRes/GLORYS/ALT — под них
уже есть отдельный run_StitchNodes_ocean.py (для GLORYS/ALT/альтиметрии,
плоская схема с одним годовым файлом, без combo eps/size_filter/extr_type).
Если 2_run_StitchNodes_by_year.py всё ещё нужен для этих типов данных в старом
виде — скажите, верну веткование, просто оставлю ERA5-ветку с новой схемой.

Один запуск = одна комбинация (eps, size_filter, extr_type) x (search_range,
mintime, maxgap, prioritize). Перебор всех 5x4x2=40 комбинаций StitchNodes
(и eps x size_filter x extr_type сверху) — отдельный раннер, ещё не написан
(следующий шаг после этого скрипта).
"""

import os
from pathlib import Path

from tqdm import tqdm

PATH_INIT = "/storage/thalassa/users/vkoshkina/data/TempestExtremes"
DATA_TYPE = "ERA5"
STITCHNODES_CMD = "StitchNodes"  # убедитесь, что доступен в PATH


def build_dirs(path_init, data_type, region_name, sigma, eps, min_samples, size_filter, extr_type):
    """sigma_dir - общий для всех combo этого региона/sigma; combo_dir - конкретная
    (eps, size_filter, extr_type), та же, что строит Stage B в create_Nodes_from_DBSCAN*.py."""
    sigma_dir = f"{path_init}/{data_type}/R2D_{data_type}_{region_name}_sigma_{sigma}"
    combo_dir = f"{sigma_dir}/{eps:02d}-{min_samples:02d}-{size_filter:02d}_{extr_type}"
    return sigma_dir, combo_dir


def build_stitch_postfix(search_range: float, mintime: int, maxgap: int, prioritize: bool) -> str:
    """_range_{search_range}_{mintime}h_{maxgap}h[_prioritize_r2d] -
    кодирует только параметры StitchNodes (eps/size_filter/extr_type уже в combo_dir)."""
    range_str = str(search_range).replace(".", "_")
    postfix = f"_range_{range_str}_{mintime}h_{maxgap}h"
    if prioritize:
        postfix += "_prioritize_r2d"
    return postfix


def run_stitchnodes_for_combo(
    *,
    path_init: str = PATH_INIT,
    data_type: str = DATA_TYPE,
    region_name: str,
    sigma: int,
    eps: int,
    min_samples: int = 4,
    size_filter: int,
    extr_type: str,  # 'global' | 'local' | 'geom' - какую папку узлов Stage B брать
    years,
    search_range: float,
    mintime: int,
    maxgap: int,
    prioritize: bool = False,
    stitchnodes_cmd: str = STITCHNODES_CMD,
    skip_existing: bool = True,
):
    sigma_dir, combo_dir = build_dirs(
        path_init, data_type, region_name, sigma, eps, min_samples, size_filter, extr_type
    )
    postfix = build_stitch_postfix(search_range, mintime, maxgap, prioritize)

    input_list_dir = f"{combo_dir}/R2D_txt_files_list_for_tracking"
    tracks_dir = f"{combo_dir}/Tracks_R2D_txt_files{postfix}"

    os.makedirs(input_list_dir, exist_ok=True)
    os.makedirs(tracks_dir, exist_ok=True)

    desc = f"eps={eps:02d} sf={size_filter:02d} {extr_type} | range={search_range} mintime={mintime}h maxgap={maxgap}h"
    for year in tqdm(years, desc=desc):
        year_nodes_dir = f"{combo_dir}/R2D_txt_files_{year}"
        if not os.path.isdir(year_nodes_dir):
            print(f"Нет папки узлов {year_nodes_dir} (Stage B для этой комбинации не запускался?), пропуск {year}")
            continue

        output_list_file = f"{input_list_dir}/{data_type}_R2D_extr_{year}.txt"
        output_tracks_file = f"{tracks_dir}/{data_type}_TC_tracks_{year}.txt"
        # StitchNodes пишет напрямую в --out; если процесс упадёт/будет убит
        # (OOM/таймаут/kill) на середине года, файл по конечному имени уже
        # существует (частично) - на следующем запуске skip_existing принял
        # бы его за готовый. Поэтому StitchNodes пишет в tmp_tracks_file, и
        # только exit_code == 0 переименовывается в output_tracks_file -
        # тогда его существование действительно означает "год досчитан".
        tmp_tracks_file = f"{output_tracks_file}.partial"

        if skip_existing and os.path.exists(output_tracks_file):
            print(f"{output_tracks_file} уже существует, пропуск {year}")
            continue

        base_path_out = f"{year_nodes_dir}/{data_type}_R2D_extr"
        n_months_found = 0
        with open(output_list_file, "w") as fout:
            for month in range(1, 13):
                output_txt = f"{base_path_out}_{year}-{month:02d}.txt"
                if os.path.exists(output_txt):
                    fout.write(f"{output_txt}\n")
                    n_months_found += 1
                else:
                    print(f"Внимание: нет файла узлов {output_txt} (месяц пропущен)")

        if n_months_found == 0:
            print(f"Список файлов для {year} пуст (нет месячных txt в {year_nodes_dir}), пропуск")
            continue
        if n_months_found < 12:
            print(f"{year}: найдено только {n_months_found}/12 месяцев узлов - треки будут неполными")

        if os.path.exists(tmp_tracks_file):
            os.remove(tmp_tracks_file)  # огрызок от прерванного прошлого прогона - начинаем заново

        cmd = (
            f"{stitchnodes_cmd} "
            f"--in_list {output_list_file} "
            f"--out {tmp_tracks_file} "
            f'--in_fmt "lon,lat,rad,r2d,wind" '
            f"--range {search_range} "
            f'--mintime "{mintime}h" '
            f'--maxgap "{maxgap}h" '
        )
        if prioritize:
            cmd += "--prioritize -r2d "

        print(f"Выполняется: {cmd}")
        original_dir = os.getcwd()
        try:
            os.chdir(combo_dir)
            exit_code = os.system(cmd)
            if exit_code == 0 and os.path.exists(tmp_tracks_file):
                os.replace(tmp_tracks_file, output_tracks_file)  # атомарная публикация только успешного результата
                print(f"Успешно: треки {year} сохранены в {output_tracks_file}")
            else:
                print(
                    f"Ошибка StitchNodes для {year} (exit code {exit_code}) - итоговый файл НЕ создан, "
                    f"частичный результат (если есть) оставлен в {tmp_tracks_file}. "
                    f"Год будет пересобран заново при следующем запуске."
                )
        except Exception as e:
            print(f"Ошибка при выполнении команды для {year}: {e}")
        finally:
            os.chdir(original_dir)

    return combo_dir, postfix


if __name__ == "__main__":
    # === Stage B combo (какие узлы берём - должны быть уже созданы create_Nodes_from_DBSCAN*.py) ===
    eps = 1
    min_samples = 4
    size_filter = 25
    extr_type = "global"  # 'global' | 'local' | 'geom'

    sigma = 2
    level_hPa = 850
    region_name = f"NA_for_TC_{level_hPa}hPa"

    years = range(2010, 2011)

    # === Stage C (StitchNodes) параметры ===
    search_range = 1.5  # 2026-07-14 - скорость перемещения самых быстрых EX <= 140 км/ч (Bernhardt&DeGaetano,2012; Lodise et al.,2022)
    mintime = 18  # 2026-07-14 - как в Han, Y., & Ullrich, P. A. (2025)
    maxgap = 3  # 2026-08-25 - стараемся получить надёжные треки по часовым данным
    prioritize = False  # True -> --prioritize -r2d (соединение с узлом с макс. r2d)

    combo_dir, postfix = run_stitchnodes_for_combo(
        region_name=region_name,
        sigma=sigma,
        eps=eps,
        min_samples=min_samples,
        size_filter=size_filter,
        extr_type=extr_type,
        years=years,
        search_range=search_range,
        mintime=mintime,
        maxgap=maxgap,
        prioritize=prioritize,
    )

    print(f"\nДля Stage D (3_create_csv_tracks_from_StitchNodes.py) передайте:")
    print(f"  --sigma-dir {combo_dir}")
    print(f"  --postfix   {postfix}")

    # Пример перебора нескольких maxgap на одном пуле узлов (один Stage B, три Stage C):
    # for maxgap in (3, 4, 6, 9, 12):
    #     run_stitchnodes_for_combo(
    #         region_name=region_name, sigma=sigma, eps=eps, min_samples=min_samples,
    #         size_filter=size_filter, extr_type=extr_type, years=years,
    #         search_range=search_range, mintime=mintime, maxgap=maxgap, prioritize=prioritize,
    #     )
