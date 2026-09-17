"""
Grid-раннер — сквозной перебор всей сетки

    eps x size_filter x extr_type x maxgap x mintime x prioritize

поверх уже написанных стадий пайплайна, вызываемых как обычные Python-функции
(без Snakemake — по вашему выбору: обычный скрипт с циклами и ручным
skip-if-exists на каждом шаге):

    Stage B  1_create_Nodes_from_DBSCAN.py / 1_create_Nodes_from_DBSCAN_geom.py
    Stage C  2_run_StitchNodes_by_year.py
    Stage D  3_create_csv_tracks_from_StitchNodes.py
    Stage E' 4_compute_era5_params_for_nodes.py
    Stage E  5_join_params_into_tracks.py

Stage G (huracanpy-валидация) сюда пока не входит — ещё не написана, добавится
отдельно и будет проходить по combo_log.csv, который пишет этот раннер.

Экономия на Stage E' (самый дорогой шаг, ~10 ч/год на комбинацию)
--------------------------------------------------------------------
size_filter — пост-фильтр по числу точек в кластере поверх ОДНОГО и того же
DBSCAN(eps, min_samples) (см. докстринги 1_create_Nodes_from_DBSCAN.py и
1_create_Nodes_from_DBSCAN_geom.py после правки 2026-08-30). Узлы size_filter=
25/49 — строгое подмножество узлов size_filter=10 (тот же (time, lon_idx,
lat_idx), то же rad/crit/wspd) для одного (eps, extr_type). Поэтому:

  - Stage B запускается для ВСЕХ size_filter в сетке (10, 25, 49) — треки
    трекаются по-разному в зависимости от того, какие вихри отфильтрованы,
    это и есть смысл перебора size_filter;
  - Stage E' запускается только ОДИН РАЗ на (eps, extr_type), на пуле узлов
    size_filter=10 (самом полном) — а для size_filter=25/49 в Stage E
    переиспользуется тот же parquet (джойн по (time,i,j) находит совпадение
    для каждой точки, т.к. это подмножество).

Это сокращает число дорогих Stage E'-прогонов с 18 (eps x size_filter x
extr_type) до 6 (eps x extr_type) — в 3 раза дешевле по самому долгому шагу.

Требование к размещению файла
------------------------------
Скрипт импортирует остальные стадии как модули (не через subprocess), поэтому
должен лежать В ТОЙ ЖЕ папке, что и они (CS_processing/CVS_alt_tracking/
TempestExtremes/): 1_create_Nodes_from_DBSCAN.py, 1_create_Nodes_from_DBSCAN_geom.py,
2_run_StitchNodes_by_year.py, 3_create_csv_tracks_from_StitchNodes.py,
4_compute_era5_params_for_nodes.py, 5_join_params_into_tracks.py.

Четыре независимые фазы (--stage) - для прозрачности пайплайна
-----------------------------------------------------------------
2026-08-31: было три фазы (tracking/eprime/e), но tracking сама внутри была
слитной (Stage C + Stage D одним блокирующим вызовом на лист). Stage C
(StitchNodes) занимает секунды-десятки секунд на лист, Stage D (конвертация
годового txt в CSV на трек) даже после векторизации (2026-09-01, см.
докстринг convert_year() в 3_create_csv_tracks_from_StitchNodes.py) может
занимать заметно дольше на листьях с большим числом треков - слитно они
заставляли всю сетку ждать самый долгий лист Stage D, прежде чем StitchNodes
успевал "протрековать" всё остальное. Поэтому 2026-09-01: tracking и csv -
тоже отдельные независимые фазы. Теперь ЧЕТЫРЕ ЯВНО независимые фазы, каждая
- отдельный процесс, каждая пишет grid_run_log.csv СВОЕЙ строкой на лист
сетки со СВОИМ статусом (см. ниже) - в любой момент по логу видно, на каком
именно шаге застряла та или иная комбинация:

    --stage tracking   Stage B(все size_filter) + Stage C (StitchNodes -
                        построение самих треков, без CSV/ERA5-параметров).
                        Быстрый шаг - вся сетка проходится за разумное время.
                        Статус строки: tracking_ok / tracking_error.

    --stage csv         ТОЛЬКО Stage D (конвертация уже готового
                        Stage C-трека, tracks_txt, в CSV на трек). Если
                        tracks_txt для листа ещё нет (tracking не отработал) -
                        лист помечается pending_tracking и пропускается,
                        запустите --stage csv ещё раз позже.
                        Статус строки: csv_ok / pending_tracking / csv_error.

    --stage eprime      Stage B(size_filter=10) + Stage E'
                        (расчёт ERA5-параметров на узлах - см. "Экономия на
                        Stage E'" выше). В grid_run_log.csv НЕ пишет (эта
                        фаза не про листья сетки, а про (eps, extr_type) -
                        прогресс виден по помесячным parquet-чекпоинтам,
                        см. докстринг run_stage_e_prime()).

    --stage e           ТОЛЬКО Stage E (джойн уже готового Stage E'-parquet
                        в уже готовые Stage D-треки). Ничего не пересчитывает
                        выше - если для листа ещё нет Stage D (csv не
                        отработал) или Stage E'-parquet (eprime не отработал),
                        лист помечается pending_stage_d / pending_stage_e_prime
                        и пропускается - запустите --stage e ещё раз позже.
                        Статус строки: ok / pending_stage_d /
                        pending_stage_e_prime / error.

tracking и eprime зависят только от Stage A (общий для всех размеров сетки,
не считается этим раннером) - друг от друга не зависят вовсе, поэтому их
можно (и стоит) запускать как два независимых параллельных процесса. csv
зависит только от tracking (по конкретному листу - tracks_txt), не от eprime;
e зависит от csv И eprime. Типичный порядок для "быстро протрековать всё, а
CSV/джойн - потом":

    python 6_grid_runner_TE_ERA5.py --stage tracking --years 2010          # быстро, вся сетка
    python 6_grid_runner_TE_ERA5.py --stage eprime --years 2010 --stage-e-workers 8 &   # параллельно, не зависит от tracking
    python 6_grid_runner_TE_ERA5.py --stage csv --years 2010               # после tracking (или следом, по мере готовности отдельных листьев)
    python 6_grid_runner_TE_ERA5.py --stage e --years 2010                 # после csv И eprime

csv можно запускать и НЕ дожидаясь, пока --stage tracking полностью пройдёт
всю сетку - skip-check пропустит листья без готового tracks_txt как
pending_tracking, а повторный запуск --stage csv позже подхватит то, что
tracking успел досчитать за это время.

--stage all (по умолчанию) - прежнее слитное поведение: всё последовательно
в одном процессе (Stage B+C+D+E' +E), ОДНА строка лога на лист (status
ok/pending_stage_e_prime/error, как раньше) - оставлено для обратной
совместимости/простых прогонов, но не даёт независимости фаз друг от друга.

Пример запуска
--------------
python 6_grid_runner_TE_ERA5.py --dry-run          # только посчитать/показать сетку, ничего не считать
python 6_grid_runner_TE_ERA5.py --years 2010       # реальный прогон, всё последовательно (--stage all)
python 6_grid_runner_TE_ERA5.py --years 2010 --stage-e-workers 8
"""

from __future__ import annotations

import argparse
import csv
import importlib
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# Модули пронумерованы (1_xxx.py ... 5_xxx.py) для наглядности порядка запуска
# в списке файлов - но такое имя не годится для обычного `import 1_xxx`
# (Python-идентификатор не может начинаться с цифры). importlib.import_module
# работает с произвольной строкой-именем файла, поэтому используем его.
stage_b_global_local = importlib.import_module("1_create_Nodes_from_DBSCAN")
stage_b_geom = importlib.import_module("1_create_Nodes_from_DBSCAN_geom")
stage_c = importlib.import_module("2_run_StitchNodes_by_year")
stage_d = importlib.import_module("3_create_csv_tracks_from_StitchNodes")
stage_e_prime = importlib.import_module("4_compute_era5_params_for_nodes")
stage_e = importlib.import_module("5_join_params_into_tracks")


# ============================================================================
# CONFIG — сетка перебора и фиксированные параметры. Правьте здесь, как и в
# остальных скриптах пайплайна (не через CLI, кроме --years/--dry-run/...).
# ============================================================================

PATH_INIT = "/storage/thalassa/users/vkoshkina/data"
DATA_TYPE = "ERA5"
SIGMA = 2
LEVEL_HPA = 850
REGION_PREFIX = "NA_for_TC"  # значение по умолчанию - переопределяется флагом --region (см. main())
REGION_NAME = f"{REGION_PREFIX}_{LEVEL_HPA}hPa"
MIN_SAMPLES = 4

EPS_VALUES = [1, 2]
SIZE_FILTER_VALUES = [10, 25, 49]  # 10 обязателен в сетке - это ещё и источник для Stage E' (см. докстринг)
EXTR_TYPES = ["global", "local", "geom"]

SEARCH_RANGE = 1.5  # фиксирован (не перебирается) - см. 2_run_StitchNodes_by_year.py
MAXGAP_VALUES = [3, 4, 6, 9, 12]
MINTIME_VALUES = [9, 12, 18, 24]
PRIORITIZE_VALUES = [False, True]

ADD_PARAMS_MODULE_PATH = stage_e_prime.DEFAULT_ADD_PARAMS_PATH
STAGE_E_PRIME_WORKERS_DEFAULT = 4
STAGE_E_PRIME_MAX_OPEN_DATASETS = 8

STAGE_D_VARIABLE_NAMES = ["rad", "r2d", "wspd"]
# 2026-08-31: convert_year (txt -> CSV на трек) стал сильно быстрее -
# см. докстринг convert_year() в 3_create_csv_tracks_from_StitchNodes.py
# (векторизованный groupby вместо hrcn.sel_id() в цикле по каждому треку -
# именно это раньше было "долгим переводом txt в csv"). STAGE_D_FAST=False -
# аварийный откат на старое поведение без правки кода в двух местах ниже,
# если на боевой сетке всё-таки найдётся расхождение с fast=True.
STAGE_D_FAST = True
# Аналогично для join_combo (Stage E) - см. её докстринг в 5_join_params_into_tracks.py.
STAGE_E_FAST = True

COMBO_LOG_PATH = Path(PATH_INIT) / "TempestExtremes" / DATA_TYPE / f"grid_run_log_{REGION_NAME}.csv"
# 2026-08-31: раньше было "grid_run_log.csv" - ОДИН файл на PATH_INIT/DATA_TYPE,
# без REGION_NAME. Пока существовал только один регион (NA_for_TC_850hPa), это
# было не важно, но REGION_NAME/LEVEL_HPA - конфиг-константы этого файла (см.
# "Правьте здесь" выше), которые можно (и предполагается) менять между
# прогонами для другого региона (например Arctic_850hPa) - при этом ВСЕ
# файловые пути пайплайна (combo_dir_path, Stage A nc_dir) УЖЕ содержат
# REGION_NAME и не пересекаются между регионами, а вот старый общий
# grid_run_log.csv - пересекался бы: два региона писали бы строки в ОДИН файл,
# и find_combos() (tracks_comparison_lib.py, группировка по COMBO_DIMS - без
# region) молча смешивала бы их для одинаковых (eps, size_filter, extr_type,
# maxgap, mintime, prioritize, year) из РАЗНЫХ регионов - Stage G/H сравнение
# тихо взяло бы tracks_txt не того региона. Теперь у каждого региона - свой
# лог-файл, конфликт исключён по построению.
#
# Миграция уже накопленных данных (регион NA_for_TC_850hPa, единственный на
# момент этой правки): один раз на сервере переименуйте старый файл, чтобы он
# нашёлся под новым именем:
#     mv {PATH_INIT}/TempestExtremes/{DATA_TYPE}/grid_run_log.csv \
#        {PATH_INIT}/TempestExtremes/{DATA_TYPE}/grid_run_log_NA_for_TC_850hPa.csv
# (подставьте реальный PATH_INIT/DATA_TYPE - см. CONFIG выше). Без этого шага
# новый прогон для NA просто начнёт новый пустой лог - уже посчитанные Stage
# C/D/E результаты на диске не пострадают (скрипт их не трогает), но
# find_combos()/audit_nodes_day_coverage.py не увидят старую историю, пока
# файл не переименован.
COMBO_LOG_FIELDS = [
    "timestamp", "eps", "size_filter", "extr_type", "search_range", "mintime", "maxgap",
    "prioritize", "year", "combo_dir", "postfix", "params_parquet", "tracks_txt",
    "csv_tracks_dir", "final_dir", "n_missing_params", "status", "error",
]


def log_row(row: dict) -> None:
    COMBO_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new = not COMBO_LOG_PATH.exists()
    with open(COMBO_LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COMBO_LOG_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def _dir_has_files(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


# ============================================================================
# Stage B (диспетчер global/local vs geom - разные модули, разные сигнатуры)
# ============================================================================

def run_stage_b(eps: int, size_filter: int, extr_type: str, years) -> None:
    if extr_type == "geom":
        stage_b_geom.run_for_combo(
            eps=eps, size_filter=size_filter, sigma=SIGMA, data_type=DATA_TYPE,
            level_hPa=LEVEL_HPA, region_name=REGION_NAME, years=years,
            path_init=PATH_INIT, min_samples=MIN_SAMPLES,
        )
    else:
        internal_extr_type = f"_{extr_type}"  # 'global' -> '_global', 'local' -> '_local'
        stage_b_global_local.run_for_combo(
            eps=eps, size_filter=size_filter, extr_type=internal_extr_type,
            sigma=SIGMA, data_type=DATA_TYPE, level_hPa=LEVEL_HPA, region_name=REGION_NAME,
            years=years, path_init=PATH_INIT, min_samples=MIN_SAMPLES,
        )


def combo_dir_path(eps: int, size_filter: int, extr_type: str) -> Path:
    """Та же формула, что использует Stage B/Stage C для комбо-папки - единая точка
    правды здесь, чтобы не разъезжаться со стадиями при правках."""
    return Path(
        f"{PATH_INIT}/TempestExtremes/{DATA_TYPE}/R2D_{DATA_TYPE}_{REGION_NAME}_sigma_{SIGMA}/"
        f"{eps:02d}-{MIN_SAMPLES:02d}-{size_filter:02d}_{extr_type}"
    )


# ============================================================================
# Stage E' (один раз на (eps, extr_type), на пуле size_filter=10)
# ============================================================================

_add_params_module = None  # ленивая загрузка один раз на весь прогон


def _get_add_params_module():
    global _add_params_module
    if _add_params_module is None:
        print(f"Импорт расчётного ядра из {ADD_PARAMS_MODULE_PATH} ...")
        _add_params_module = stage_e_prime.load_add_params_module(ADD_PARAMS_MODULE_PATH)
    return _add_params_module


def run_stage_e_prime(eps: int, extr_type: str, year: int, n_workers: int) -> Path:
    """Возвращает путь к params_by_node_{year}.parquet на пуле size_filter=10.
    Если уже посчитан - не пересчитывает (самый дорогой шаг, поэтому skip
    здесь особенно важен, в отличие от более дешёвых Stage C/D/E).

    2026-08-30: запись сделана атомарной (tmp + os.replace), как в Stage
    B/C - раньше result_df.to_parquet() писал СРАЗУ в output_parquet, и
    если процесс убьют/упадёт посреди записи (а parquet - не построчный
    формат, там несколько row group'ов и футер в конце файла), на диске
    мог остаться "готовый на вид", но битый/усечённый файл - следующий
    запуск (или, что хуже, ПАРАЛЛЕЛЬНЫЙ процесс --stage tracking/all, который
    как раз проверяет Path(params_parquet).exists() перед Stage E, см.
    докстринг модуля про --stage) принял бы его за полностью посчитанный
    и либо тоже пропустил бы пересчёт, либо попытался бы прочитать битый
    parquet в Stage E. Теперь os.replace() публикует файл под финальным
    именем только после того, как to_parquet() полностью отработал.

    2026-08-30: сам расчёт теперь идёт через compute_params_for_nodes_monthly()
    (см. её докстринг в 4_compute_era5_params_for_nodes.py) - день за днём
    внутри Pool всё ещё считается за один проход по месяцу, но КАЖДЫЙ месяц
    сохраняется как отдельный чекпоинт-parquet сразу после расчёта. Раньше
    при сбое на середине года (например, на 24-м дне из 365) весь прогресс
    года терялся - нечего было переиспользовать, потому что до этой правки
    результаты копились в памяти и писались одним файлом только в конце.
    Теперь при повторном запуске уже посчитанные месяцы читаются с диска
    (доли секунды), пересчитывается только то, что не успело. Чекпоинты
    лежат в combo_dir_10/params_by_node_{year}_monthly/{year}-{MM}.parquet -
    сами по себе не нужны после того, как итоговый годовой parquet ниже
    успешно собран и опубликован, но их можно не удалять (места немного,
    а как диагностика/для другого года пригодятся)."""
    combo_dir_10 = combo_dir_path(eps, size_filter=10, extr_type=extr_type)
    nodes_dir = combo_dir_10 / f"R2D_txt_files_{year}"
    output_parquet = combo_dir_10 / f"params_by_node_{year}.parquet"
    monthly_dir = combo_dir_10 / f"params_by_node_{year}_monthly"
    tmp_parquet = output_parquet.with_suffix(output_parquet.suffix + f".partial.pid{os.getpid()}")

    if output_parquet.exists():
        print(f"[Stage E'] уже посчитан: {output_parquet} - пропуск")
        return output_parquet

    if not nodes_dir.is_dir():
        raise FileNotFoundError(
            f"[Stage E'] нет папки узлов {nodes_dir} - Stage B для (eps={eps}, "
            f"size_filter=10, extr_type={extr_type}) не отработал?"
        )

    add_params_module = _get_add_params_module()
    nodes_df = stage_e_prime.load_all_nodes(nodes_dir, year)
    print(f"[Stage E'] {len(nodes_df)} узлов за {year}, eps={eps}, extr_type={extr_type} - считаем параметры "
          f"(помесячные чекпоинты в {monthly_dir})...")

    result_df = stage_e_prime.compute_params_for_nodes_monthly(
        nodes_df, add_params_module, ADD_PARAMS_MODULE_PATH, monthly_dir, year, n_workers=n_workers,
        max_open_datasets=STAGE_E_PRIME_MAX_OPEN_DATASETS,
    )
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    try:
        result_df.to_parquet(tmp_parquet, index=False)
        os.replace(tmp_parquet, output_parquet)  # атомарная публикация только полностью записанного файла
    except BaseException:
        # BaseException, а не Exception - чтобы .partial подчищался и после Ctrl+C
        # (KeyboardInterrupt). При kill -9/OOM-killer этот except вообще не
        # выполнится (сигнал не даёт Python доработать) - тогда .partial.pidNNN
        # просто останется на диске, но это не страшно: skip-check в начале
        # функции смотрит только на ФИНАЛЬНОЕ имя output_parquet, .partial-файл
        # ни на что не влияет и его можно спокойно удалить руками при уборке.
        if tmp_parquet.exists():
            tmp_parquet.unlink()
        raise
    print(f"[Stage E'] сохранено {len(result_df)} узлов в {output_parquet}")
    return output_parquet


# ============================================================================
# Один лист сетки: (eps, size_filter, extr_type, maxgap, mintime, prioritize)
# ============================================================================

def run_one_leaf(eps, size_filter, extr_type, maxgap, mintime, prioritize, year, params_parquet, stage_e_prime_workers):
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "eps": eps, "size_filter": size_filter, "extr_type": extr_type,
        "search_range": SEARCH_RANGE, "mintime": mintime, "maxgap": maxgap,
        "prioritize": prioritize, "year": year,
        "combo_dir": "", "postfix": "", "params_parquet": str(params_parquet),
        "tracks_txt": "", "csv_tracks_dir": "", "final_dir": "",
        "n_missing_params": "", "status": "", "error": "",
    }
    try:
        # --- Stage C: StitchNodes ---
        combo_dir, postfix = stage_c.run_stitchnodes_for_combo(
            path_init=f"{PATH_INIT}/TempestExtremes",
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

        # --- Stage D: CSV на трек ---
        csv_tracks_dir = combo_dir / f"csv_Tracks{postfix}"
        row["csv_tracks_dir"] = str(csv_tracks_dir)
        if not _dir_has_files(csv_tracks_dir):
            stage_d.convert_year(tracks_txt, csv_tracks_dir, STAGE_D_VARIABLE_NAMES, fast=STAGE_D_FAST)
        else:
            print(f"[Stage D] {csv_tracks_dir} уже не пуст - пропуск")

        # --- Stage E: джойн параметров (Stage E' parquet - общий на eps/extr_type) ---
        final_dir = combo_dir / f"csv_Tracks{postfix}_params"
        row["final_dir"] = str(final_dir)
        if not Path(params_parquet).exists():
            # Обычная ситуация при --stage all, запущенном отдельно от
            # --stage eprime: parquet для этого (eps, extr_type) ещё не
            # досчитан. Stage C/D уже сделаны (см. выше) - не пропадают,
            # только джойн параметров откладывается. Запустите --stage all
            # ещё раз позже - Stage C/D пропустятся как готовые (skip-check),
            # досчитается только Stage E. (Для раздельного запуска используйте
            # --stage tracking + --stage e - см. докстринг модуля.)
            row["status"] = "pending_stage_e_prime"
            print(
                f"[Stage E] {params_parquet} ещё не готов (Stage E' для eps={eps}, "
                f"extr_type={extr_type} не закончен) - Stage D сделан, джойн параметров "
                f"отложен до повторного запуска."
            )
            return row  # finally ниже сам залогирует row - явный log_row тут не нужен

        if not _dir_has_files(final_dir):
            stage_e.join_combo(
                tracks_dir=csv_tracks_dir, params_parquet=params_parquet, output_dir=final_dir,
                fast=STAGE_E_FAST,
            )
        else:
            print(f"[Stage E] {final_dir} уже не пуст - пропуск")

        row["status"] = "ok"
    except Exception as e:
        row["status"] = "error"
        row["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        log_row(row)

    return row


# ============================================================================
# Тот же лист, но раздельно на четыре независимые фазы (--stage tracking /
# --stage csv / --stage eprime / --stage e) - см. докстринг модуля "Четыре
# независимые фазы". run_one_leaf() выше не трогаем (используется только
# --stage all, старое слитное поведение).
# ============================================================================

def run_stage_tracking_leaf(eps, size_filter, extr_type, maxgap, mintime, prioritize, year):
    """ТОЛЬКО Stage C (StitchNodes) для одного листа сетки - БЕЗ Stage D (CSV на
    трек - отдельно, run_stage_csv_leaf/--stage csv) и БЕЗ Stage E (джойн
    параметров - отдельно, run_stage_e_leaf/--stage e). Пишет свою строку в
    grid_run_log.csv, статус относится ТОЛЬКО к этой фазе: tracking_ok/tracking_error.

    2026-09-01: раньше здесь же (в одном блокирующем вызове) шёл и Stage D -
    см. докстринг модуля "Четыре независимые фазы" про то, почему это
    заставляло быстрый Stage C ждать медленный Stage D на каждом листе."""
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "eps": eps, "size_filter": size_filter, "extr_type": extr_type,
        "search_range": SEARCH_RANGE, "mintime": mintime, "maxgap": maxgap,
        "prioritize": prioritize, "year": year,
        "combo_dir": "", "postfix": "", "params_parquet": "",
        "tracks_txt": "", "csv_tracks_dir": "", "final_dir": "",
        "n_missing_params": "", "status": "", "error": "",
    }
    try:
        combo_dir, postfix = stage_c.run_stitchnodes_for_combo(
            path_init=f"{PATH_INIT}/TempestExtremes",
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
    """ТОЛЬКО Stage D (StitchNodes tracks_txt -> CSV на трек) для одного листа -
    независимо от того, когда именно отработал --stage tracking для этого
    листа (в этом же процессе или в другом, только что или давно). combo_dir/
    postfix пересчитываются той же формулой, что и в run_stage_e_leaf - не
    полагаемся на порядок вызовов внутри одного процесса.

    Сам convert_year() уже атомарно публикует csv_tracks_dir (см. её докстринг
    в 3_create_csv_tracks_from_StitchNodes.py) и сам пропускает уже
    опубликованные - здесь дополнительный skip-check не нужен."""
    combo_dir = combo_dir_path(eps, size_filter, extr_type)
    postfix = stage_c.build_stitch_postfix(SEARCH_RANGE, mintime, maxgap, prioritize)
    tracks_txt = combo_dir / f"Tracks_R2D_txt_files{postfix}" / f"{DATA_TYPE}_TC_tracks_{year}.txt"
    csv_tracks_dir = combo_dir / f"csv_Tracks{postfix}"

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "eps": eps, "size_filter": size_filter, "extr_type": extr_type,
        "search_range": SEARCH_RANGE, "mintime": mintime, "maxgap": maxgap,
        "prioritize": prioritize, "year": year,
        "combo_dir": str(combo_dir), "postfix": postfix, "params_parquet": "",
        "tracks_txt": str(tracks_txt), "csv_tracks_dir": str(csv_tracks_dir), "final_dir": "",
        "n_missing_params": "", "status": "", "error": "",
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


def run_stage_e_leaf(eps, size_filter, extr_type, maxgap, mintime, prioritize, year):
    """Stage E (джойн Stage E'-параметров в Stage D-треки) для одного листа -
    полностью независимо от того, в этом же или в другом запуске отработали
    Stage B/C/D (--stage tracking) и Stage E' (--stage eprime). combo_dir/
    postfix/csv_tracks_dir вычисляются той же формулой, что и в Stage C
    (combo_dir_path + stage_c.build_stitch_postfix) - не полагаемся на то,
    что tracking только что отработал в этом же процессе."""
    combo_dir = combo_dir_path(eps, size_filter, extr_type)
    postfix = stage_c.build_stitch_postfix(SEARCH_RANGE, mintime, maxgap, prioritize)
    tracks_txt = combo_dir / f"Tracks_R2D_txt_files{postfix}" / f"{DATA_TYPE}_TC_tracks_{year}.txt"
    csv_tracks_dir = combo_dir / f"csv_Tracks{postfix}"
    params_parquet = combo_dir_path(eps, 10, extr_type) / f"params_by_node_{year}.parquet"
    final_dir = combo_dir / f"csv_Tracks{postfix}_params"

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "eps": eps, "size_filter": size_filter, "extr_type": extr_type,
        "search_range": SEARCH_RANGE, "mintime": mintime, "maxgap": maxgap,
        "prioritize": prioritize, "year": year,
        "combo_dir": str(combo_dir), "postfix": postfix, "params_parquet": str(params_parquet),
        "tracks_txt": str(tracks_txt), "csv_tracks_dir": str(csv_tracks_dir), "final_dir": str(final_dir),
        "n_missing_params": "", "status": "", "error": "",
    }
    try:
        if not _dir_has_files(csv_tracks_dir):
            row["status"] = "pending_stage_d"
            print(
                f"[Stage E] {csv_tracks_dir} ещё пуст/не создан - Stage D для этого листа "
                f"(--stage csv) ещё не отработал, джойн отложен до повторного запуска --stage e."
            )
            return row

        if not params_parquet.exists():
            row["status"] = "pending_stage_e_prime"
            print(
                f"[Stage E] {params_parquet} ещё не готов (Stage E' для eps={eps}, extr_type={extr_type} "
                f"не закончен) - джойн отложен до повторного запуска --stage e."
            )
            return row

        if _dir_has_files(final_dir):
            print(f"[Stage E] {final_dir} уже не пуст - пропуск")
        else:
            stage_e.join_combo(
                tracks_dir=csv_tracks_dir, params_parquet=params_parquet, output_dir=final_dir,
                fast=STAGE_E_FAST,
            )

        row["status"] = "ok"
    except Exception as e:
        row["status"] = "error"
        row["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        log_row(row)
    return row


# ============================================================================
# main
# ============================================================================

def iter_grid():
    for eps in EPS_VALUES:
        for extr_type in EXTR_TYPES:
            for size_filter in SIZE_FILTER_VALUES:
                for maxgap in MAXGAP_VALUES:
                    for mintime in MINTIME_VALUES:
                        for prioritize in PRIORITIZE_VALUES:
                            yield eps, size_filter, extr_type, maxgap, mintime, prioritize


def main():
    global REGION_NAME, COMBO_LOG_PATH  # переопределяются ниже из --region (см. докстринг --region)

    parser = argparse.ArgumentParser(description="Grid-раннер Stage B-E по сетке eps x size_filter x extr_type x maxgap x mintime x prioritize")
    parser.add_argument("--years", type=int, nargs="+", default=[2010])
    parser.add_argument("--stage-e-workers", type=int, default=STAGE_E_PRIME_WORKERS_DEFAULT)
    parser.add_argument(
        "--region", default=REGION_PREFIX,
        help=f"Префикс региона (по умолчанию '{REGION_PREFIX}') - REGION_NAME собирается как "
             f"'{{region}}_{{LEVEL_HPA}}hPa' (LEVEL_HPA - CONFIG-константа, не CLI, см. верх файла). "
             f"Все пути пайплайна (combo_dir_path, Stage A nc_dir) и grid_run_log_{{REGION_NAME}}.csv "
             f"уже зависят от REGION_NAME, поэтому НА ОДНОМ PATH_INIT можно безопасно запускать разные "
             f"регионы параллельно (в разных сессиях/процессах) - файлы не пересекаются. Пример: "
             f"--region Arctic -> REGION_NAME=Arctic_850hPa. ВАЖНО: audit_nodes_day_coverage.py и "
             f"fix_stale_nodes_txt.py читают REGION_NAME из CONFIG этого модуля при импорте (это "
             f"ОТДЕЛЬНЫЕ процессы/запуски) - --region здесь на них не влияет, для другого региона в них "
             f"нужен свой способ указать регион (сейчас - --sigma-dir в audit; в fix_stale_nodes_txt.py "
             f"для Stage A ещё не реализовано - скажите, если нужно для Arctic)."
    )
    parser.add_argument(
        "--stage", choices=["all", "tracking", "csv", "eprime", "e"], default="all",
        help="all = Stage B+E'+C/D/E последовательно в одном процессе, одна строка лога на лист (по "
             "умолчанию, прежнее поведение); tracking = ТОЛЬКО Stage B(все size_filter)+Stage C "
             "(построение треков StitchNodes, без CSV/ERA5-параметров) - быстрый шаг, для запуска РЯДОМ "
             "с --stage eprime; csv = ТОЛЬКО Stage D (уже готовый tracks_txt -> CSV на трек) - листья без "
             "готового tracking помечаются pending_tracking (запустите --stage csv повторно позже); "
             "eprime = ТОЛЬКО Stage B(size_filter=10)+Stage E' (расчёт ERA5-параметров на узлах) - "
             "не пишет в grid_run_log.csv, прогресс - по помесячным parquet-чекпоинтам; "
             "e = ТОЛЬКО Stage E (джойн уже готового Stage E'-parquet в уже готовые Stage D-треки) - "
             "ничего не пересчитывает выше, листья без готового csv/eprime помечаются "
             "pending_stage_d/pending_stage_e_prime и пропускаются (запустите --stage e повторно позже). "
             "См. докстринг модуля 'Четыре независимые фазы'."
    )
    parser.add_argument("--dry-run", action="store_true", help="Только показать размер сетки и пути, ничего не считать")
    args = parser.parse_args()

    # --region переопределяет REGION_NAME (и, соответственно, COMBO_LOG_PATH) ДО того, как
    # что-либо этими значениями воспользуется - все функции ниже (combo_dir_path, log_row,
    # run_stage_b/run_stage_e_prime/...) читают их как globals в момент вызова, так что
    # переприсвоение здесь, в самом начале main(), распространяется на весь прогон.
    REGION_NAME = f"{args.region}_{LEVEL_HPA}hPa"
    COMBO_LOG_PATH = Path(PATH_INIT) / "TempestExtremes" / DATA_TYPE / f"grid_run_log_{REGION_NAME}.csv"

    leaves = list(iter_grid())
    n_stage_b_combos = len(EPS_VALUES) * len(EXTR_TYPES) * len(SIZE_FILTER_VALUES)
    n_stage_e_prime_combos = len(EPS_VALUES) * len(EXTR_TYPES)
    print(
        f"Регион: {REGION_NAME} (лог: {COMBO_LOG_PATH})\n"
        f"Сетка: {len(leaves)} листьев (Stage C/D/E) x {len(args.years)} год(а/лет); "
        f"Stage B - {n_stage_b_combos} прогонов; Stage E' (дорогой шаг) - только {n_stage_e_prime_combos} прогонов "
        f"(вместо {n_stage_b_combos} без переиспользования size_filter=10). --stage={args.stage}"
    )
    if args.dry_run:
        for eps, size_filter, extr_type, maxgap, mintime, prioritize in leaves[:5]:
            print(f"  например: eps={eps} size_filter={size_filter} extr_type={extr_type} "
                  f"maxgap={maxgap}h mintime={mintime}h prioritize={prioritize} "
                  f"-> {combo_dir_path(eps, size_filter, extr_type)}")
        print("  ... (--dry-run: выполнение пропущено)")
        return

    for year in args.years:
        if args.stage in ("all", "eprime"):
            # --- Stage B(size_filter=10) + Stage E' один раз на (eps, extr_type) ---
            for eps in EPS_VALUES:
                for extr_type in EXTR_TYPES:
                    print(f"\n[Stage B] eps={eps} size_filter=10 extr_type={extr_type} year={year} (источник для Stage E')")
                    run_stage_b(eps, 10, extr_type, years=(year, year))

                    print(f"\n[Stage E'] eps={eps} extr_type={extr_type} year={year} (общий для всех size_filter)")
                    run_stage_e_prime(eps, extr_type, year, n_workers=args.stage_e_workers)

            if args.stage == "eprime":
                continue  # tracking/e - забота отдельных процессов

        if args.stage in ("all", "tracking"):
            # --- Stage B на всех size_filter (10 - тот же вызов, что и выше;
            # skip-check делает повтор дешёвым и безопасным даже при параллельном
            # запуске с --stage eprime - см. докстринг модуля) ---
            for eps in EPS_VALUES:
                for extr_type in EXTR_TYPES:
                    for size_filter in SIZE_FILTER_VALUES:
                        print(f"\n[Stage B] eps={eps} size_filter={size_filter} extr_type={extr_type} year={year}")
                        run_stage_b(eps, size_filter, extr_type, years=(year, year))

            for eps, size_filter, extr_type, maxgap, mintime, prioritize in leaves:
                if args.stage == "tracking":
                    print(
                        f"\n[Stage C] eps={eps} size_filter={size_filter} extr_type={extr_type} "
                        f"maxgap={maxgap}h mintime={mintime}h prioritize={prioritize} year={year}"
                    )
                    run_stage_tracking_leaf(eps, size_filter, extr_type, maxgap, mintime, prioritize, year)
                else:  # "all" - прежнее слитное поведение, Stage E тоже здесь
                    # Путь к parquet считаем формулой напрямую (не из словаря,
                    # заполняемого run_stage_e_prime) - Stage E' мог посчитаться
                    # в СОВСЕМ ДРУГОМ процессе (--stage eprime), так что полагаться
                    # на то, что он выполнялся в этом же процессе, нельзя.
                    params_parquet = combo_dir_path(eps, 10, extr_type) / f"params_by_node_{year}.parquet"
                    print(
                        f"\n[Stage C/D/E] eps={eps} size_filter={size_filter} extr_type={extr_type} "
                        f"maxgap={maxgap}h mintime={mintime}h prioritize={prioritize} year={year}"
                    )
                    run_one_leaf(eps, size_filter, extr_type, maxgap, mintime, prioritize, year,
                                 params_parquet, args.stage_e_workers)

        if args.stage == "csv":
            # --- ТОЛЬКО Stage D: конвертация уже готового Stage C-трека
            # (tracks_txt) в CSV на трек (см. run_stage_csv_leaf) - листья без
            # готового tracking помечаются pending_tracking и пропускаются. ---
            for eps, size_filter, extr_type, maxgap, mintime, prioritize in leaves:
                print(
                    f"\n[Stage D] eps={eps} size_filter={size_filter} extr_type={extr_type} "
                    f"maxgap={maxgap}h mintime={mintime}h prioritize={prioritize} year={year}"
                )
                run_stage_csv_leaf(eps, size_filter, extr_type, maxgap, mintime, prioritize, year)

        if args.stage == "e":
            # --- ТОЛЬКО Stage E: джойн уже готового Stage E'-parquet в уже
            # готовые Stage D-треки (см. run_stage_e_leaf) ---
            for eps, size_filter, extr_type, maxgap, mintime, prioritize in leaves:
                print(
                    f"\n[Stage E] eps={eps} size_filter={size_filter} extr_type={extr_type} "
                    f"maxgap={maxgap}h mintime={mintime}h prioritize={prioritize} year={year}"
                )
                run_stage_e_leaf(eps, size_filter, extr_type, maxgap, mintime, prioritize, year)

    print(f"\nГотово. Лог по каждой комбинации: {COMBO_LOG_PATH}")


if __name__ == "__main__":
    main()
