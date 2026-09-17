"""
Общая библиотека для сравнения треков TempestExtremes (ERA5) с внешними
референсными треками — IBTrACS, SyCLoPS, EddyClicker (ручная разметка).

Используется:
    7_compare_TE_with_reference.py    — IBTrACS и SyCLoPS (глобальные базы)
    8_compare_TE_with_EddyClicker.py  — EddyClicker (ручная разметка, NAAD LoRes)

Идея разделения: то, что специфично для конкретного ИСТОЧНИКА (как читать
файлы, откуда брать lon/lat/id) — в load_*_tracks() ниже. Всё остальное
(huracanpy.assess.match/pod, трек-статистика, графики) работает только с
track_id/lon/lat/time и не знает, откуда взялись данные — один и тот же код
обслуживает IBTrACS, SyCLoPS и EddyClicker. Не вызывает эту библиотеку сама
по себе — это набор функций, а не CLI.

TE-треки (эта часть не менялась со Stage G, см. докстринг 6_grid_runner_TE_ERA5.py):
читаются НЕ по формуле путей, а из grid_run_log.csv, который построчно (лист
сетки x год) пишет 6_grid_runner_TE_ERA5.py — единственный источник правды о
том, что реально посчитано и где лежит.

Произвольный источник -> huracanpy
-----------------------------------
huracanpy умеет грузить произвольный CSV: huracanpy.load(path, source="csv"),
где каждая строка — точка трека, есть колонки track_id, year, month, day,
hour, lon, lat (+ любые дополнительные атрибуты сохраняются как есть) —
см. https://huracanpy.readthedocs.io/en/stable/examples/load_csv.html.
Это ТОТ ЖЕ путь построения xarray Dataset, которым huracanpy сам строит его
для tempestextremes/ibtracs, поэтому результат гарантированно совместим с
huracanpy.assess.match/pod — в отличие от ручной сборки xr.Dataset "на глаз".
dataframe_to_huracanpy() ниже поэтому не строит Dataset сама, а пишет
временный CSV в этом формате и отдаёт его huracanpy.load().

ВАЖНО (честно, а не молча): в этой песочнице нет доступа ни к huracanpy, ни к
серверу, поэтому маршрут source="csv" проверен только по официальной
документации huracanpy (ссылка выше), а не реальным запуском — перед первым
боевым прогоном 7_/8_ имеет смысл быстро прогнать load_syclops_tracks/
load_eddyclicker_tracks на маленьком куске данных и посмотреть на результат.
Вся остальная логика в этом файле (парсинг SyCLoPS CSV, чтение EddyClicker,
группировка grid_run_log.csv) протестирована синтетически без huracanpy.
"""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_rgba
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# Пути/CONFIG для TE — из грид-раннера, единая точка правды (см. combo_dir_path там же)
grid_runner = importlib.import_module("6_grid_runner_TE_ERA5")

PATH_INIT = grid_runner.PATH_INIT
DATA_TYPE = grid_runner.DATA_TYPE
COMBO_LOG_PATH = grid_runner.COMBO_LOG_PATH
REGION_NAME = grid_runner.REGION_NAME
SIGMA = grid_runner.SIGMA

# Папка конкретной комбинации sigma/региона - та же, что родитель всех combo_dir
# (см. combo_dir_path/SIGMA_DIR в 6_grid_runner_TE_ERA5.py и audit_nodes_day_coverage.py).
# 2026-08-31: результаты сравнения (7_*/8_*) раньше складывались в
# PATH_INIT/TempestExtremes/DATA_TYPE/ - на уровень выше, рядом со ВСЕМИ
# sigma/регионами сразу. Теперь - ВНУТРИ этой sigma/регион-папки, рядом с
# combo-папками (01-04-10_global и т.п.), чтобы результаты сравнения физически
# лежали там же, где все варианты, которые сравнивались.
SIGMA_DIR = Path(PATH_INIT) / "TempestExtremes" / DATA_TYPE / f"R2D_{DATA_TYPE}_{REGION_NAME}_sigma_{SIGMA}"

# Статусы grid_run_log.csv, для которых tracks_txt считается готовым.
# Для сравнения TE-треков с эталоном (IBTrACS/SyCLoPS/EddyClicker) нужен
# только Stage C (StitchNodes) - ERA5-параметры (Stage E'/Stage E) тут не
# участвуют вовсе. 2026-08-31: после разделения 6_grid_runner_TE_ERA5.py на
# независимые --stage tracking/eprime/e (см. его докстринг) строку с
# гарантированно готовым tracks_txt пишет ИМЕННО --stage tracking (статус
# "tracking_ok"), поэтому он добавлен в список. "ok"/"pending_stage_e_prime"
# оставлены для обратной совместимости со старыми запусками --stage all/tracks
# (слитная схема, где tracks_txt тоже гарантированно готов при этих статусах).
ELIGIBLE_STATUSES = {"tracking_ok", "ok", "pending_stage_e_prime"}

MAX_DIST = 200   # km, huracanpy.assess.match
MEAN_DIST = 120  # km
MIN_OVERLAP = 3  # минимум совпадающих точек

COMBO_DIMS = ["eps", "size_filter", "extr_type", "maxgap", "mintime", "prioritize", "search_range"]

SYCLOPS_ROOT = Path(PATH_INIT) / "SyCLoPS" / "tracks_types_csv"
# Актуальные коды типов из SyCLoPS/filter_type_for_year_and_region_*.py
# (TYPE_CONFIG: TC / SS / PL). СТАРЫЙ tracks_comparison/comparison_utils.py
# использовал "STLC"/"PTLC"/"MS" и путь TempestExtremes/SyCLoPS/{type}/*.txt —
# это рассинхронизировалось с текущим фильтрующим скриптом (он пишет
# TC/SS/PL как CSV в tracks_types_csv/), поэтому здесь используются коды и
# путь фильтрующего скрипта, а не старого comparison_utils.py.
SYCLOPS_TYPES = ["TC", "SS", "PL"]


# ============================================================================
# 1. TE: чтение grid_run_log.csv и группировка по комбинациям (как в Stage G)
# ============================================================================

def read_combo_log(log_path: Path = COMBO_LOG_PATH) -> pd.DataFrame:
    if not log_path.exists():
        raise FileNotFoundError(
            f"Не найден {log_path} — сначала запустите 6_grid_runner_TE_ERA5.py "
            f"(хотя бы --stage tracks)."
        )
    df = pd.read_csv(log_path)
    df["eps"] = df["eps"].astype(int)
    df["size_filter"] = df["size_filter"].astype(int)
    df["maxgap"] = df["maxgap"].astype(int)
    df["mintime"] = df["mintime"].astype(int)
    df["year"] = df["year"].astype(int)
    df["search_range"] = df["search_range"].astype(float)
    df["prioritize"] = df["prioritize"].astype(str).str.lower().isin(["true", "1"])
    return df


def combo_label(combo: dict) -> str:
    label = (
        f"eps{combo['eps']:02d}_sf{combo['size_filter']:02d}_{combo['extr_type']}"
        f"_mg{combo['maxgap']}h_mt{combo['mintime']}h"
    )
    if combo["prioritize"]:
        label += "_prioritize"
    return label


def stitch_sub_label(combo: dict) -> str:
    """Часть combo_label() БЕЗ eps/size_filter/extr_type - используется как
    подпапка ВНУТРИ конкретной combo-папки (см. combo_output_dir), где
    eps/size_filter/extr_type уже заданы самим путём - остаётся различать
    только StitchNodes-параметры (maxgap/mintime/prioritize)."""
    label = f"mg{combo['maxgap']}h_mt{combo['mintime']}h"
    if combo["prioritize"]:
        label += "_prioritize"
    return label


def combo_output_dir(combo: dict, reference_subpath) -> Path:
    """Папка результатов сравнения ДЛЯ ОДНОЙ комбинации, ВНУТРИ её собственной
    combo-папки (eps-min_samples-size_filter_extr_type) - рядом с
    Tracks_R2D_txt_files{postfix}/csv_Tracks{postfix}/params_by_node_*.parquet
    и т.д., а не в общей папке на уровне sigma_dir (см. докстринг SIGMA_DIR -
    там остаётся только СВОДКА по всей сетке сразу, summary_all_configurations.csv
    и графики - у неё нет "своей" combo-папки, т.к. она про много комбинаций
    одновременно).

    reference_subpath - что сравнивали, например 'IBTrACS', 'SyCLoPS/TC_NA',
    'EddyClicker' (Path или строка с '/').
    """
    combo_dir = grid_runner.combo_dir_path(combo["eps"], combo["size_filter"], combo["extr_type"])
    return combo_dir / "data_comparison_huracanpy" / Path(reference_subpath) / stitch_sub_label(combo)


def find_combos(
    log_df: pd.DataFrame,
    years=None, eps=None, size_filter=None, extr_type=None,
    maxgap=None, mintime=None, prioritize=None,
    statuses=ELIGIBLE_STATUSES,
) -> list[dict]:
    df = log_df[log_df["status"].isin(statuses)].copy()

    if years:
        df = df[df["year"].isin(years)]
    if eps is not None:
        df = df[df["eps"] == eps]
    if size_filter is not None:
        df = df[df["size_filter"] == size_filter]
    if extr_type is not None:
        df = df[df["extr_type"] == extr_type]
    if maxgap is not None:
        df = df[df["maxgap"] == maxgap]
    if mintime is not None:
        df = df[df["mintime"] == mintime]
    if prioritize is not None:
        df = df[df["prioritize"] == prioritize]

    df = df[df["tracks_txt"].apply(lambda p: isinstance(p, str) and Path(p).exists())]
    if df.empty:
        return []

    combos = []
    for combo_key, group in df.groupby(COMBO_DIMS, dropna=False):
        combo = dict(zip(COMBO_DIMS, combo_key))
        # grid_run_log.csv - append-only (6_grid_runner_TE_ERA5.py дописывает
        # строку на КАЖДЫЙ запуск, никогда не перезаписывает старую) - если
        # один и тот же (combo, year) лист был посчитан больше одного раза
        # (например, раньше прогоняли --stage all, потом отдельно --stage
        # tracks на те же годы), в логе будет несколько строк на один год.
        # dict(zip(...)) сам по себе уже схлопывает дубли по ключу year,
        # оставляя последнюю (т.к. group отсортирована по year с сохранением
        # исходного - хронологического - порядка строк при равенстве года),
        # но САМ список years ниже раньше строился из НЕотсортированной по
        # схлопыванию group_sorted["year"] и мог содержать один год
        # несколько раз подряд (это видно было в выводе --dry-run: "years=
        # [2010, 2010]"). Для TE (словарь по году) и IBTrACS (.isin()) дубль
        # в списке лет был безобиден, но для SyCLoPS load_syclops_tracks()
        # буквально грузит файл года в цикле `for year in years` - с
        # дублирующимся годом трек оказался бы посчитан в датасете ДВАЖДЫ
        # (задвоенные точки/id), искажая matching. Поэтому years теперь
        # строится из уже дедуплицированных ключей словаря, а не из сырых
        # строк лога.
        group_sorted = group.sort_values("year")
        tracks_txt_by_year = dict(zip(group_sorted["year"], group_sorted["tracks_txt"]))
        n_raw_rows = len(group_sorted)
        if n_raw_rows > len(tracks_txt_by_year):
            print(f"[find_combos] {combo_label(combo)}: в grid_run_log.csv {n_raw_rows} строк на "
                  f"{len(tracks_txt_by_year)} уникальных год(а/лет) - беру последнюю запись на каждый год.")
        combo["tracks_txt_by_year"] = tracks_txt_by_year
        combo["years"] = sorted(int(y) for y in tracks_txt_by_year.keys())
        combos.append(combo)
    return combos


def load_te_tracks_for_combo(combo: dict, variable_names=("rad", "r2d", "wspd")):
    import huracanpy
    all_tracks = []
    for year, tracks_txt in tqdm(
        combo["tracks_txt_by_year"].items(), desc=f"Loading TE tracks {combo_label(combo)}", leave=False,
    ):
        try:
            all_tracks.append(huracanpy.load(str(tracks_txt), source="tempestextremes", variable_names=list(variable_names)))
        except FileNotFoundError:
            print(f"  {tracks_txt} не найден, пропуск {year}")
            continue
    if not all_tracks:
        raise FileNotFoundError(f"Не удалось загрузить ни одного года треков для {combo_label(combo)}")
    return huracanpy.concat_tracks(all_tracks)


# ============================================================================
# 2. Произвольный табличный источник -> huracanpy (через CSV-загрузчик huracanpy)
# ============================================================================

def dataframe_to_huracanpy(df: pd.DataFrame, track_id_col: str, lon_col: str, lat_col: str, time_col: str,
                            extra_cols: list[str] | None = None):
    """Строит huracanpy-совместимый Dataset из произвольного pandas DataFrame
    через официальный CSV-загрузчик huracanpy (source="csv") — см. докстринг
    модуля. df должен быть уже "плоским" (одна строка = одна точка трека)."""
    import huracanpy

    cols = [track_id_col, time_col, lon_col, lat_col] + list(extra_cols or [])
    work = df[cols].copy()
    work = work.rename(columns={track_id_col: "track_id", lon_col: "lon", lat_col: "lat"})

    t = pd.to_datetime(work[time_col])
    work["year"] = t.dt.year
    work["month"] = t.dt.month
    work["day"] = t.dt.day
    work["hour"] = t.dt.hour
    work = work.drop(columns=[time_col])

    fd, tmp_path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    try:
        work.to_csv(tmp_path, index=False)
        return huracanpy.load(tmp_path, source="csv")
    finally:
        os.remove(tmp_path)


def build_csv_for_huracanpy(df: pd.DataFrame, track_id_col: str, lon_col: str, lat_col: str, time_col: str,
                             extra_cols: list[str] | None = None) -> pd.DataFrame:
    """Чистая (без huracanpy/IO) часть dataframe_to_huracanpy — то, что
    реально можно протестировать без huracanpy: строит DataFrame в точности
    того вида, который будет записан во временный CSV и отдан
    huracanpy.load(..., source='csv')."""
    cols = [track_id_col, time_col, lon_col, lat_col] + list(extra_cols or [])
    work = df[cols].copy()
    work = work.rename(columns={track_id_col: "track_id", lon_col: "lon", lat_col: "lat"})
    t = pd.to_datetime(work[time_col])
    work["year"] = t.dt.year
    work["month"] = t.dt.month
    work["day"] = t.dt.day
    work["hour"] = t.dt.hour
    return work.drop(columns=[time_col])


# ============================================================================
# 3. SyCLoPS — читает RAW CSV прямо из tracks_types_csv/ (без промежуточной
#    конвертации в TE-txt формат — тот путь, которым пользовался старый
#    comparison_utils.py, зависел от конвертации, которая, похоже, не
#    поддерживается в актуальном виде: filter_type_for_year_and_region*.py
#    пишет TC/SS/PL как CSV, а не как TempestExtremes .txt)
# ============================================================================

def normalize_lon_360_to_180(lon) -> float:
    """SyCLoPS хранит LON в [0, 360) — переводим в [-180, 180), как у ERA5/TE."""
    return ((np.asarray(lon, dtype=float) + 180.0) % 360.0) - 180.0


def find_syclops_files(syclops_type: str, year: int, region: str = "NA", root: Path = SYCLOPS_ROOT) -> list[Path]:
    """Имя файла — {stem}_tracks.csv, где stem кодирует ещё и границы региона
    (см. SyCLoPS/filter_type_for_year_and_region_*.py), поэтому ищем по
    маске с wildcard на месте границ, а не строим точное имя."""
    pattern = f"SyCLoPS_{syclops_type}_{region}_*_{year}_tracks.csv"
    return sorted(Path(root).glob(pattern))


def load_syclops_tracks(syclops_type: str, years: list[int], region: str = "NA", root: Path = SYCLOPS_ROOT,
                         on_multiple: str = "newest"):
    """on_multiple: если для (type, region, year) нашлось НЕСКОЛЬКО файлов
    (в repo несколько версий filter_type_for_year_and_region*.py с разными
    границами региона — на сервере в одной папке могли накопиться файлы от
    разных запусков) - 'newest' берёт файл с самым поздним mtime (по
    умолчанию, безопаснее тихого объединения дублей), 'all' объединяет все
    найденные (используйте, только если точно знаете, что дублей нет)."""
    frames = []
    for year in years:
        files = find_syclops_files(syclops_type, year, region, root)
        if not files:
            print(f"SyCLoPS {syclops_type}/{region}: {year} не найден в {root}, пропуск")
            continue
        if len(files) > 1:
            print(f"SyCLoPS {syclops_type}/{region}/{year}: найдено {len(files)} файлов: "
                  f"{[f.name for f in files]}")
            if on_multiple == "newest":
                files = [max(files, key=lambda f: f.stat().st_mtime)]
                print(f"  -> беру самый свежий: {files[0].name} (on_multiple='newest')")
        for f in files:
            df = pd.read_csv(f, parse_dates=["ISOTIME"])
            frames.append(df)

    if not frames:
        raise FileNotFoundError(f"Не найдено ни одного файла SyCLoPS {syclops_type}/{region} за {years} в {root}")

    combined = pd.concat(frames, ignore_index=True)
    combined["lon_180"] = normalize_lon_360_to_180(combined["LON"])
    combined = combined.rename(columns={"TID": "track_id_syclops"})

    return dataframe_to_huracanpy(
        combined, track_id_col="track_id_syclops", lon_col="lon_180", lat_col="LAT", time_col="ISOTIME",
        extra_cols=["MSLP", "WS"],
    )


# ============================================================================
# 4. EddyClicker — ручная разметка на сетке NAAD LoRes (Cartesian), но
#    сравнение делаем по lat/lon (см. add_latlon.py: latitude/longitude
#    получены по индексам x/y из XLAT/XLONG WRF-файла) - те же huracanpy
#    match/POD, что и для IBTrACS/SyCLoPS, просто источник - папка с одним
#    CSV на трек, а не единый файл.
# ============================================================================

_LAT_COL_CANDIDATES = ("lat", "latitude", "LAT")
_LON_COL_CANDIDATES = ("lon", "longitude", "LON")
_TIME_COL_CANDIDATES = ("datetime", "time", "ISOTIME")


def _first_present(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def load_eddyclicker_tracks(tracks_dir: Path, extra_cols: list[str] | None = None):
    """Читает ВСЕ *.csv в tracks_dir как ОТДЕЛЬНЫЕ треки (один файл = один
    трек — формат папок add_track_params_EddyClicker*.py). Колонки
    lat/lon/datetime определяются автоматически из нескольких возможных
    имён (в разных версиях EddyClicker-скриптов они называются по-разному —
    add_latlon.py пишет 'latitude'/'longitude', а add_track_params_*
    ожидает 'lat'/'lon'; здесь не гадаем какая версия использовалась,
    просто берём то, что реально есть в файле).

    Возвращает (huracanpy Dataset, file_map), где file_map: track_id (int,
    присвоен по порядку файлов) -> исходное имя файла (для отчётности/CSV)."""
    files = sorted(Path(tracks_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"Нет *.csv в {tracks_dir}")

    rows = []
    file_map = {}
    skipped = []

    for track_id, f in enumerate(files):
        df = pd.read_csv(f)
        lat_col = _first_present(df.columns, _LAT_COL_CANDIDATES)
        lon_col = _first_present(df.columns, _LON_COL_CANDIDATES)
        time_col = _first_present(df.columns, _TIME_COL_CANDIDATES)
        if lat_col is None or lon_col is None or time_col is None:
            skipped.append((f.name, lat_col, lon_col, time_col))
            continue

        cols = [time_col, lat_col, lon_col] + [c for c in (extra_cols or []) if c in df.columns]
        sub = df[cols].rename(columns={time_col: "time", lat_col: "lat", lon_col: "lon"})
        sub["track_id"] = track_id
        rows.append(sub)
        file_map[track_id] = f.name

    if skipped:
        print(f"EddyClicker: пропущено {len(skipped)} файлов без lat/lon/datetime "
              f"(искали среди {_LAT_COL_CANDIDATES}/{_LON_COL_CANDIDATES}/{_TIME_COL_CANDIDATES}):")
        for name, lat_col, lon_col, time_col in skipped[:10]:
            print(f"  {name}: lat={lat_col} lon={lon_col} time={time_col}")

    if not rows:
        raise FileNotFoundError(
            f"Ни в одном из {len(files)} файлов в {tracks_dir} не нашлось колонок lat/lon/datetime — "
            f"проверьте, что add_latlon.py уже отработал для этой папки."
        )

    combined = pd.concat(rows, ignore_index=True).dropna(subset=["lat", "lon", "time"])
    present_extra = [c for c in (extra_cols or []) if c in combined.columns]

    ds = dataframe_to_huracanpy(
        combined, track_id_col="track_id", lon_col="lon", lat_col="lat", time_col="time",
        extra_cols=present_extra,
    )
    return ds, file_map


# ============================================================================
# 5. Generic matching / POD / трек-статистика (не знает про источник)
# ============================================================================

def calculate_matches(reference_tracks, te_tracks, reference_name):
    import huracanpy
    return huracanpy.assess.match(
        [reference_tracks, te_tracks], names=[reference_name, "TE_NA"],
        max_dist=MAX_DIST, mean_dist=MEAN_DIST, min_overlap=MIN_OVERLAP, tracks1_is_ref=True,
    )


def calculate_pod(matches, reference_tracks, reference_name):
    import huracanpy
    return float(huracanpy.assess.pod(matches, ref=reference_tracks, ref_name=reference_name))


def select_best_matches(matches, reference_id_column):
    if len(matches) == 0:
        return matches.copy()
    return (
        matches.sort_values([reference_id_column, "temp", "dist"], ascending=[True, False, True])
        .drop_duplicates(subset=reference_id_column, keep="first")
        .reset_index(drop=True)
    )


def calculate_temporal_overlap(reference_track, te_track):
    ref = pd.to_datetime(reference_track.time.values)
    te = pd.to_datetime(te_track.time.values)
    if len(ref) == 0 or len(te) == 0:
        return {"overlap_start": pd.NaT, "overlap_end": pd.NaT, "overlap_duration_hours": 0, "overlap_duration_fraction": 0}
    ref_start, ref_end = ref.min(), ref.max()
    te_start, te_end = te.min(), te.max()
    duration = (ref_end - ref_start).total_seconds() / 3600
    start, end = max(ref_start, te_start), min(ref_end, te_end)
    overlap = 0 if start > end else (end - start).total_seconds() / 3600
    fraction = overlap / duration if duration > 0 else 0
    return {"overlap_start": start, "overlap_end": end, "overlap_duration_hours": overlap, "overlap_duration_fraction": fraction}


def calculate_track_statistics(reference_tracks, te_tracks, matches, reference_id_column, te_id_column="id_TE_NA"):
    reference_ids = np.unique(reference_tracks.track_id.values)
    best = select_best_matches(matches, reference_id_column)
    best_dict = {r[reference_id_column]: r for _, r in best.iterrows()}
    rows = []

    for ref_id in reference_ids:
        ref_track = reference_tracks.where(reference_tracks.track_id == ref_id, drop=True).sortby("time")
        times = pd.to_datetime(ref_track.time.values)
        n = len(times)
        duration = ((times.max() - times.min()).total_seconds() / 3600) if n > 1 else 0

        if ref_id not in best_dict:
            rows.append({
                "reference_ID": ref_id, "detected": False, "TE_ID_best": np.nan,
                "reference_n_points": n, "reference_duration_hours": duration,
                "matched_n_points": 0, "overlap_fraction": 0, "overlap_percent": 0,
                "mean_distance_km": np.nan, "overlap_start": pd.NaT, "overlap_end": pd.NaT,
                "overlap_duration_hours": 0, "overlap_duration_fraction": 0, "overlap_duration_percent": 0,
            })
            continue

        row = best_dict[ref_id]
        te_id = row[te_id_column]
        matched = row["temp"]
        dist = row["dist"]
        frac = matched / n if n else np.nan
        te_track = te_tracks.where(te_tracks.track_id == te_id, drop=True).sortby("time")
        temporal = calculate_temporal_overlap(ref_track, te_track)

        rows.append({
            "reference_ID": ref_id, "detected": True, "TE_ID_best": te_id,
            "reference_n_points": n, "reference_duration_hours": duration,
            "matched_n_points": matched, "overlap_fraction": frac, "overlap_percent": 100 * frac,
            "mean_distance_km": dist, "overlap_start": temporal["overlap_start"], "overlap_end": temporal["overlap_end"],
            "overlap_duration_hours": temporal["overlap_duration_hours"],
            "overlap_duration_fraction": temporal["overlap_duration_fraction"],
            "overlap_duration_percent": 100 * temporal["overlap_duration_fraction"],
        })

    return pd.DataFrame(rows)


def summarize_track_statistics(track_stats: pd.DataFrame, pod: float, n_reference: int, n_te) -> dict:
    """2026-09-02: добавлены mean/median_overlap_percent_1to1 - явно поименованная
    "1-к-1" метрика качества воспроизведения шторма ОДНИМ, наиболее долго
    совпадающим TE-треком (по запросу - "оцени отдельно качество по
    воспроизведению трека ТЦ одним, наиболее длительным совпадающим треком").

    Это НЕ новая формула - calculate_track_statistics() уже для каждого
    референсного трека выбирает РОВНО ОДИН TE-трек через select_best_matches()
    (сортировка по temp desc, dist asc - т.е. именно "самый долго совпадающий",
    дистанция только тай-брейк), поэтому пара (референсный трек, TE_ID_best)
    уже 1-к-1 по построению. overlap_duration_percent - это доля [genesis,
    lysis] референсного шторма, покрытая ЭТИМ единственным лучшим TE-треком
    (по времени, а не по числу точек) - то есть ровно "качество воспроизведения
    ОДНИМ треком". Числа здесь совпадают с mean/median_temporal_overlap_percent
    ниже (та же величина) - колонки добавлены отдельно и явно так названы,
    чтобы их было легко процитировать в таблице как "1-к-1"-метрику, не путая с
    mean/median_overlap_percent (тот - по числу точек, не по времени)."""
    d = track_stats[track_stats.detected] if len(track_stats) else track_stats
    if len(d):
        vals = {
            "mean_overlap_percent": d.overlap_percent.mean(), "median_overlap_percent": d.overlap_percent.median(),
            "mean_temporal_overlap_percent": d.overlap_duration_percent.mean(),
            "median_temporal_overlap_percent": d.overlap_duration_percent.median(),
            "mean_overlap_percent_1to1": d.overlap_duration_percent.mean(),
            "median_overlap_percent_1to1": d.overlap_duration_percent.median(),
            "mean_distance_km": d.mean_distance_km.mean(), "median_distance_km": d.mean_distance_km.median(),
        }
    else:
        vals = {k: np.nan for k in [
            "mean_overlap_percent", "median_overlap_percent", "mean_temporal_overlap_percent",
            "median_temporal_overlap_percent", "mean_overlap_percent_1to1", "median_overlap_percent_1to1",
            "mean_distance_km", "median_distance_km",
        ]}
    return {
        "n_reference_tracks": n_reference, "n_matched": len(d), "n_unmatched": n_reference - len(d),
        "n_TE_tracks": n_te, "POD": pod, **vals,
    }


COMBINED_SCORE_COLUMNS = ["POD", "median_overlap_percent", "median_overlap_percent_1to1"]


def sort_summary_by_combined_score(summary_df: pd.DataFrame, add_column: bool = True) -> pd.DataFrame:
    """2026-09-02: сортирует сводную таблицу (summary_all_configurations.csv)
    по убыванию composite-метрики POD + median_overlap_percent +
    median_overlap_percent_1to1 - по запросу "сортировка по убыванию
    POD+median_overlap_percent+median_overlap_percent_1_1".

    POD хранится как доля (0..1), а оба overlap - в процентах (0..100),
    поэтому перед суммированием POD домножается на 100 - иначе сумма была бы
    почти целиком определена одним POD (0..1 против 0..100+0..100), что не
    похоже на то, что имелось в виду под "суммой трёх метрик". Итоговая
    колонка combined_score = 100*POD + median_overlap_percent +
    median_overlap_percent_1to1 (в диапазоне примерно 0..300) добавляется в
    таблицу (add_column=True, по умолчанию) - так видно, ПОЧЕМУ строки
    отсортированы именно так, а не только результат.

    Строки с NaN в любой из трёх колонок (например, комбинация взята из
    ранее посчитанного кэша без --force - см. коммент у _summarize_row в
    7_compare_TE_with_reference.py) получают combined_score=NaN и уходят в
    конец таблицы (pandas.sort_values кладёт NaN в конец независимо от
    ascending) - не путаются с реально низким, но посчитанным score."""
    df = summary_df.copy()
    missing = [c for c in COMBINED_SCORE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"В summary_df не хватает колонок для combined_score: {missing} - похоже, это таблица "
            f"от версии скриптов до 2026-09-02 (median_overlap_percent_1to1 ещё не считалась)."
        )
    df["combined_score"] = 100 * df["POD"] + df["median_overlap_percent"] + df["median_overlap_percent_1to1"]
    df = df.sort_values("combined_score", ascending=False, na_position="last").reset_index(drop=True)
    if not add_column:
        df = df.drop(columns="combined_score")
    return df


MERGE_KEY_COLUMNS = COMBO_DIMS + ["years"]

# Колонки, по полноте которых сравниваются старая/новая версии строки при
# одинаковом ключе (см. merge_summary_tables) - чем меньше NaN среди них,
# тем "полнее" строка. Специально НЕ combined_score/POD одни - если позже
# появятся другие "тяжёлые" метрики, посчитанные только при --force, их
# стоит сюда добавлять.
COMPLETENESS_COLUMNS = ["n_TE_tracks", "POD", "median_overlap_percent", "median_overlap_percent_1to1"]


def merge_summary_tables(old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """2026-09-03: сливает новую порцию строк (new_df, результат ТЕКУЩЕГО
    запуска 7_/8_compare_*.py) со СТАРОЙ, уже сохранённой на диске таблицей
    (old_df, summary_all_configurations.csv от предыдущих запусков) -
    upsert по составному ключу MERGE_KEY_COLUMNS (все измерения комбинации +
    years), а не блинд-overwrite.

    Зачем это вообще нужно: main() раньше просто писал
    `pd.DataFrame(summaries).to_csv(summary_csv)` поверх старого файла - при
    запуске с узкими фильтрами (например, только одна конкретная
    конфигурация через --eps/--size-filter/... для отрисовки одной карты,
    см. plot_top_configs.py / ручной точечный --plot-tracks запуск) это
    БЕЗВОЗВРАТНО стирало все остальные, ранее посчитанные строки на полной
    сетке - именно так была потеряна полная таблица в одном из прогонов.

    Логика:
      - строки old_df, чей ключ НЕ встречается в new_df, сохраняются как есть;
      - строки new_df, чей ключ НЕ встречается в old_df, добавляются как есть;
      - для строк с ОДИНАКОВЫМ ключом в обеих таблицах - побеждает более
        "полная" версия (меньше NaN среди COMPLETENESS_COLUMNS). Это нужно
        отдельно от простого "новое всегда важнее старого": повторный запуск
        только с --plot-tracks (без --force) на уже посчитанной комбинации
        даёт строку с NaN в POD/n_TE_tracks (см. _summarize_row) - без этой
        проверки такой перезапуск мог бы ЗАТЕРЕТЬ ранее полностью
        посчитанную строку той же комбинации NaN-версией. При равной
        полноте побеждает новая строка (newest wins).

    Если в old_df нет одной или нескольких колонок MERGE_KEY_COLUMNS/
    COMPLETENESS_COLUMNS (например, это таблица от версии скриптов до
    добавления median_overlap_percent_1to1) - соответствующие колонки
    считаются отсутствующими/NaN только для целей сравнения полноты, слияние
    по ключу всё равно работает по тем колонкам, что есть в обеих таблицах."""
    if old_df is None or len(old_df) == 0:
        return new_df.reset_index(drop=True)
    if new_df is None or len(new_df) == 0:
        return old_df.reset_index(drop=True)

    key_cols = [c for c in MERGE_KEY_COLUMNS if c in old_df.columns and c in new_df.columns]
    missing_key_cols = [c for c in MERGE_KEY_COLUMNS if c not in key_cols]
    if missing_key_cols:
        raise ValueError(
            f"merge_summary_tables: не хватает колонок ключа {missing_key_cols} в старой или новой "
            f"таблице - похоже, одна из них от несовместимой версии скрипта. Слияние отменено, чтобы "
            f"не потерять данные - проверьте вручную."
        )

    def _make_key(df: pd.DataFrame) -> pd.Series:
        return df[key_cols].astype(str).agg("|".join, axis=1)

    def _completeness(df: pd.DataFrame) -> pd.Series:
        cols = [c for c in COMPLETENESS_COLUMNS if c in df.columns]
        if not cols:
            return pd.Series(0, index=df.index)
        return df[cols].notna().sum(axis=1)

    old = old_df.copy()
    new = new_df.copy()
    old["_merge_key"] = _make_key(old)
    new["_merge_key"] = _make_key(new)
    old["_completeness"] = _completeness(old)
    new["_completeness"] = _completeness(new)

    old_by_key = {k: i for i, k in enumerate(old["_merge_key"])}
    rows = []
    used_old_idx = set()
    for i, new_row in new.iterrows():
        key = new_row["_merge_key"]
        if key in old_by_key:
            old_idx = old_by_key[key]
            used_old_idx.add(old_idx)
            old_row = old.iloc[old_idx]
            # newest wins при равной полноте - берём новую строку, если её
            # полнота >= старой, иначе оставляем старую (более полную).
            rows.append(new_row if new_row["_completeness"] >= old_row["_completeness"] else old_row)
        else:
            rows.append(new_row)
    for old_idx in range(len(old)):
        if old_idx not in used_old_idx:
            rows.append(old.iloc[old_idx])

    merged = pd.DataFrame(rows).drop(columns=["_merge_key", "_completeness"]).reset_index(drop=True)
    return merged


# ============================================================================
# 6. Графики (generic — reference_label только для подписей)
# ============================================================================

def plot_reference_track(reference_track, te_tracks, reference_id, reference_label, output_filename, year=None, title_extra=""):
    fig = plt.figure(figsize=(12, 8), dpi=150)
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.set_extent([-110, 15, 0, 73], ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue", alpha=0.3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
    ax.gridlines(draw_labels=True, linestyle="--", alpha=0.5)

    ax.plot(reference_track.lon, reference_track.lat, color="black", linewidth=3, zorder=2,
            transform=ccrs.PlateCarree(), label=reference_label)
    ax.plot(reference_track.lon[0], reference_track.lat[0], "go", markersize=8, zorder=2,
            transform=ccrs.PlateCarree(), label="Genesis")
    ax.plot(reference_track.lon[-1], reference_track.lat[-1], "rs", markersize=8, zorder=2,
            transform=ccrs.PlateCarree(), label="Lysis")

    te_ids = [] if te_tracks is None else np.unique(te_tracks.track_id.values)
    te_colors = ["#FF1493", "#FF0000", "#CC00FF", "#FF4500", "#FFA500", "#FF69B4", "#AD1457", "#8B008B", "#FF6347", "#FFD700"]
    for i, te_id in enumerate(te_ids):
        te_track = te_tracks.where(te_tracks.track_id == te_id, drop=True).sortby("time")
        ax.plot(te_track.lon, te_track.lat, color=te_colors[i % len(te_colors)], linewidth=1.8, alpha=0.9, zorder=5,
                transform=ccrs.PlateCarree(), label=f"TE {te_id}")

    title = f"{reference_label} {reference_id}"
    if year is not None:
        title += f" - {year}"
    title += f"\nBlack: {reference_label} | Colored: TempestExtremes ({len(te_ids)} tracks)"
    if title_extra:
        title += f"\n{title_extra}"
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    plt.savefig(output_filename, dpi=150, bbox_inches="tight")
    plt.close()


def plot_all_reference_tracks(reference_tracks, te_tracks, output_dir, reference_label, reference_id_column, matches,
                               max_matched_tracks: int = 10):
    """Одна карта на референсный трек - референс чёрным + ВСЕ совпавшие с ним
    TE-треки (каждый своим цветом) на ОДНОЙ карте, 1 png = 1 ТЦ (см.
    plot_reference_track - она уже рисует весь te_for_ref на одних axes).

    2026-08-31: раньше вызывающий код (7_compare_TE_with_reference.py)
    разрешал --plot-tracks только когда фильтры сузили выборку до РОВНО ОДНОЙ
    комбинации - иначе комбинации x референсные треки грозили тысячами PNG.
    Теперь ограничение убрано на уровне комбинаций и заменено на ограничение
    ЗДЕСЬ, на уровне одного референсного трека: если у него >= max_matched_tracks
    совпавших TE-треков - это почти наверняка вырожденный матчинг (слишком
    мягкие пороги/артефакт), а не читаемая картина одного шторма - карта для
    такого референсного трека пропускается (с сообщением), а не рисуется
    нечитаемой кашей. Для нормального случая (обычно 0-2 совпадения на
    реальный шторм) поведение не изменилось - все совпавшие треки идут на
    одну картинку, как и раньше."""
    os.makedirs(output_dir, exist_ok=True)
    ids = np.unique(reference_tracks.track_id.values)
    matched_ids = set(matches[reference_id_column].values) if len(matches) else set()
    n_skipped_too_many = 0

    for i, ref_id in enumerate(ids, start=1):
        ref_track = reference_tracks.where(reference_tracks.track_id == ref_id, drop=True).sortby("time")
        year = int(pd.Timestamp(ref_track.time.values[0]).year)

        if ref_id in matched_ids:
            rows = matches[matches[reference_id_column] == ref_id]
            te_ids = np.unique(rows["id_TE_NA"].values)
            if len(te_ids) >= max_matched_tracks:
                print(f"  [{reference_label} {ref_id}] карта пропущена: {len(te_ids)} совпавших TE-треков "
                      f">= {max_matched_tracks} - похоже на вырожденный матчинг, картинка была бы нечитаемой.")
                n_skipped_too_many += 1
                continue
            te_for_ref = te_tracks.where(te_tracks.track_id.isin(list(te_ids)), drop=True)
        else:
            te_for_ref = None

        filename = f"{output_dir}/{reference_label}_{ref_id}_{year}_track{i}.png"
        plot_reference_track(ref_track, te_for_ref, ref_id, reference_label, filename, year=year)

    if n_skipped_too_many:
        print(f"  Итого пропущено карт (>= {max_matched_tracks} совпадений на референсный трек): {n_skipped_too_many}")


def plot_best_single_match_tracks(reference_tracks, te_tracks, stats_df: pd.DataFrame, reference_label: str,
                                   output_dir) -> int:
    """2026-09-02: отдельные "1-к-1" картинки - референс (чёрным) + РОВНО ОДИН
    (не все совпавшие) TE-трек: тот самый, что calculate_track_statistics()
    выбрала как TE_ID_best - наиболее долго совпадающий по времени (см.
    докстринг summarize_track_statistics про mean/median_overlap_percent_1to1
    - те же пары "референс-трек x TE_ID_best", только здесь ещё и нарисованы).

    В отличие от plot_all_reference_tracks() (которая рисует ВСЕ совпавшие
    TE-треки на одной карте - удобно, чтобы увидеть фрагментацию/склейку),
    здесь картинка ВСЕГДА ровно "1 против 1", даже если у шторма было
    несколько частичных совпадений - чтобы честно показать, насколько ОДИН
    трек воспроизводит весь жизненный цикл шторма, без визуального шума от
    других кандидатов. Сохраняется в отдельную подпапку (не путать с картами
    plot_all_reference_tracks), имя файла помечено '_1to1'.

    Возвращает число сохранённых картинок (0 - если детектированных
    треков нет)."""
    os.makedirs(output_dir, exist_ok=True)
    detected = stats_df[stats_df["detected"] & stats_df["TE_ID_best"].notna()]
    n_saved = 0

    for i, row in enumerate(detected.itertuples(index=False), start=1):
        ref_id = row.reference_ID
        te_id = row.TE_ID_best
        overlap_pct = row.overlap_duration_percent

        ref_track = reference_tracks.where(reference_tracks.track_id == ref_id, drop=True).sortby("time")
        te_track = te_tracks.where(te_tracks.track_id == te_id, drop=True).sortby("time")
        if len(ref_track.time) == 0 or len(te_track.time) == 0:
            continue
        year = int(pd.Timestamp(ref_track.time.values[0]).year)

        filename = f"{output_dir}/{reference_label}_{ref_id}_{year}_1to1_track{i}.png"
        plot_reference_track(
            ref_track, te_track, ref_id, reference_label, filename, year=year,
            title_extra=f"1-к-1: наиболее долго совпадающий TE-трек, temporal overlap = {overlap_pct:.1f}%",
        )
        n_saved += 1

    print(f"  Сохранено 1-к-1 картинок (единственный лучший TE-трек на референсный трек): {n_saved}")
    return n_saved


def plot_pod_bar(summary_df: pd.DataFrame, output_filename: Path, title: str, top_n: int = 30) -> None:
    df = summary_df.sort_values("POD", ascending=False).head(top_n)
    labels = [combo_label(row) for _, row in df.iterrows()]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(df))))
    ax.barh(labels[::-1], (100 * df["POD"])[::-1], color="tab:blue")
    ax.set_xlabel("POD (%)")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    plt.tight_layout()
    output_filename.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Сохранено: {output_filename}")


def plot_configuration_summary(summary_df: pd.DataFrame, output_filename: Path | None = None,
                                title: str = "TE vs reference: sensitivity to eps, size_filter, extr_type") -> pd.DataFrame:
    """Портировано (с адаптацией под реальную схему summary_all_configurations.csv —
    см. ниже) из plot_configuration_summary() в ERA5_plot_test.ipynb.

    Отличия от версии в ноутбуке
    ------------------------------
    Ноутбук строил её под ОДИН конкретный прогон (level_hPa/size_filter из
    {500,850}x{10,25}, только IBTrACS: требовал колонки level_hPa, n_IBTrACS).
    В нашем summary_all_configurations.csv (см. run_combo()/_summarize_row() в
    7_compare_TE_with_reference.py и 8_compare_TE_with_EddyClicker.py) этих
    колонок нет — level_hPa не варьируется внутри одного прогона (это
    константа LEVEL_HPA в 6_grid_runner_TE_ERA5.py, не измерение сетки), а
    референс общий (n_reference_tracks вместо n_IBTrACS, т.к. тот же формат
    используется для IBTrACS/SyCLoPS/EddyClicker). Поэтому здесь:
        Цвет   -> eps (вместо level_hPa)
        Hatch  -> extr_type: 'local' -> ///, иначе (global/geom) -> без hatch
        Альфа  -> size_filter (произвольный набор значений, не только {10,25})
        Подпись оси X -> combo_label(row) (eps/size_filter/extr_type/maxgap/
                          mintime/prioritize) - как в plot_pod_bar(), чтобы
                          не терять maxgap/mintime/prioritize при сортировке.
    4 панели: (a) POD, (b) Median temporal overlap (median_temporal_overlap_percent
    — notebook-версия называла колонку 'median_overlap_percent', но по
    докстрингу имела в виду именно временное перекрытие треков), (c) число
    TE-треков (лог-шкала), (d) 2026-09-02: median_overlap_percent_1to1 —
    качество воспроизведения шторма ОДНИМ, наиболее долго совпадающим
    TE-треком (см. докстринг summarize_track_statistics) — ЗАМЕНИЛА собой
    прежнюю панель (d) "медианная дистанция между сопоставленными треками"
    (median_distance_km) по прямому запросу - "убери [дистанцию] из всех
    [графиков], можно 4 графиком вместо среднего расстояния". Сама колонка
    median_distance_km никуда не делась (остаётся в summary_all_configurations.csv
    как сырые данные), просто больше не занимает панель на этом рисунке.

    2026-09-02: почему рисунок раньше был "очень тяжёлый и не открывался" и
    что изменилось
    -----------------------------------------------------------------------
    figsize ширина была `max(15, 1.1 * len(df))` дюймов при dpi=300 БЕЗ
    верхнего предела — при полном переборе сетки (eps x size_filter x
    extr_type x maxgap x mintime x prioritize = до 720 строк на один год)
    это давало ширину ~790 дюймов x 300 dpi = ~237000 пикселей в одном
    измерении - на грани/за пределом жёсткого лимита Agg-бэкенда matplotlib
    (2**16 = 65536 пикселей на измерение) и в любом случае неоткрываемый по
    размеру PNG (сотни МБ - десятки/сотни тысяч текстовых подписей, хатчей,
    баров). Теперь: ширина фигуры ограничена сверху MAX_FIG_WIDTH_IN
    дюймами, а dpi при большом len(df) снижается пропорционально (не ниже
    MIN_DPI), чтобы фактическая ширина в пикселях оставалась примерно
    постоянной независимо от того, сколько строк в сетке — вместо того чтобы
    расти без ограничений. При большом числе баров (> ANNOTATE_MAX_BARS)
    числовые подписи над барами и через один-два подписи по оси X
    дополнительно прорежаются - иначе на сотнях баров они всё равно
    накладываются друг на друга и только замедляют рендер, не добавляя
    читаемости. На типичных прогонах (одна-две комбинации eps/size_filter/
    extr_type, десятки строк) поведение визуально не отличается от прежнего.

    Parameters
    ----------
    summary_df : pandas.DataFrame
        Как из summary_all_configurations.csv (7_compare_TE_with_reference.py /
        8_compare_TE_with_EddyClicker.py).
    output_filename : Path or None
        Куда сохранить PNG (директория создаётся автоматически). None — не сохранять.

    Returns
    -------
    pandas.DataFrame
        Копия summary_df, отсортированная так же, как отображено на графике.
    """
    required = ["eps", "size_filter", "extr_type", "POD", "median_temporal_overlap_percent",
                "n_TE_tracks", "median_overlap_percent_1to1"]
    missing = [c for c in required if c not in summary_df.columns]
    if missing:
        raise ValueError(f"В summary_df не хватает колонок: {missing}")

    df = summary_df.copy()
    df["extr_order"] = df["extr_type"].map({"local": 0, "global": 1, "geom": 2}).fillna(3)
    df = df.sort_values(["eps", "size_filter", "extr_order"]).reset_index(drop=True)
    df = df.drop(columns="extr_order")

    x = np.arange(len(df))
    labels = [combo_label(row) for _, row in df.iterrows()]

    eps_values = sorted(df["eps"].unique())
    eps_colors = dict(zip(eps_values, plt.cm.tab10.colors))
    hatch_types = {"local": "///", "global": "", "geom": ".."}
    size_filters = sorted(df["size_filter"].unique())
    # Первый (обычно самый маленький, source_size_filter=10) - самый бледный, дальше насыщеннее
    size_alpha = {sf: 0.35 + 0.65 * i / max(len(size_filters) - 1, 1) for i, sf in enumerate(size_filters)}

    # --- размер фигуры: ограничен сверху, dpi компенсирует большие сетки (см. докстринг) ---
    MAX_FIG_WIDTH_IN = 55
    MIN_DPI = 90
    BASE_DPI = 300
    ANNOTATE_MAX_BARS = 60  # больше этого - числовые подписи над барами только мешают, отключаем
    TICK_LABEL_DENSITY_PER_IN = 1.3  # сколько подписей по X ещё читаемо на дюйм ширины

    desired_width_in = max(15, 1.1 * len(df))
    fig_width_in = min(desired_width_in, MAX_FIG_WIDTH_IN)
    dpi = BASE_DPI if desired_width_in <= MAX_FIG_WIDTH_IN else max(
        MIN_DPI, int(BASE_DPI * fig_width_in / desired_width_in)
    )
    show_annotations = len(df) <= ANNOTATE_MAX_BARS
    max_labels = max(1, int(fig_width_in * TICK_LABEL_DENSITY_PER_IN))
    tick_step = max(1, -(-len(df) // max_labels))  # ceil(len(df) / max_labels)

    fig, axes = plt.subplots(2, 2, figsize=(fig_width_in, 10))

    def draw_bars(ax, values, ylabel, panel_title, log_scale=False):
        # 2026-09-02: коэрсим в numpy float ДО цикла - matplotlib Rectangle умеет
        # рисовать NaN-высоту (бар просто не появляется), но падает с TypeError на
        # чистом python None (int + None). Источник None - комбинации, взятые из
        # уже посчитанного matching_statistics.csv (см. _summarize_row() в
        # 7_/8_compare_*.py) - там раньше n_te_tracks мог остаться None; сам источник
        # тоже исправлен, но здесь оставлена защита на случай любых будущих дыр.
        values = np.asarray(values, dtype=float)
        bars = []
        for i, (value, eps, sf, extr) in enumerate(zip(values, df["eps"], df["size_filter"], df["extr_type"])):
            bar = ax.bar(i, value, width=0.7, color=eps_colors[eps], alpha=size_alpha[sf],
                        hatch=hatch_types.get(extr, ""), edgecolor="black", linewidth=0.8)[0]
            bars.append(bar)
        ax.set_ylabel(ylabel)
        ax.set_title(panel_title, fontsize=12)
        shown_x = x[::tick_step]
        shown_labels = labels[::tick_step]
        ax.set_xticks(shown_x)
        ax.set_xticklabels(shown_labels, fontsize=7, rotation=60, ha="right")
        finite = values[np.isfinite(values)]
        finite_positive = finite[finite > 0]
        # 2026-09-02: set_yscale("log") ЗДЕСЬ безусловно валило весь рисунок (и
        # panel (a)/(b)/(d) вместе с ним - функция одна на все 4 панели, а
        # matplotlib падает ПОЗЖЕ, внутри tight_layout, когда лог-локатор пытается
        # расставить деления и не находит НИ ОДНОГО положительного значения) - это
        # ровно тот же сценарий, что и с n_TE_tracks=NaN: если ВСЕ комбинации в
        # прогоне взяты из кэша без --force, у panel (c) "Number of TE tracks"
        # получается всё NaN, лог-шкала невозможна в принципе. Теперь лог-шкала
        # включается, только если есть хоть одно положительное конечное значение -
        # иначе тихо остаёмся на линейной (сам факт "все NaN" уже виден по
        # пустой панели, отдельно предупреждать не нужно).
        if log_scale and len(finite_positive):
            ax.set_yscale("log")
        else:
            log_scale = False
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
        if len(finite):
            top = finite.max()
            if log_scale:
                ax.set_ylim(bottom=max(1, finite_positive.min() * 0.8), top=top * 1.5)
            else:
                ax.set_ylim(bottom=0, top=top * 1.20 if top > 0 else 1)
        return bars

    def annotate(ax, bars, values, fmt):
        if not show_annotations:
            return
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                ax.text(bar.get_x() + bar.get_width() / 2, value * 1.02 if value > 0 else value + 1,
                        fmt.format(value), ha="center", va="bottom", fontsize=8)

    ax = axes[0, 0]
    pod_percent = 100 * df["POD"]
    bars = draw_bars(ax, pod_percent, "POD (%)", "(a) Probability of Detection")
    annotate(ax, bars, pod_percent, "{:.1f}")

    ax = axes[0, 1]
    overlap = df["median_temporal_overlap_percent"]
    bars = draw_bars(ax, overlap, "Median temporal overlap (%)", "(b) Median Temporal Overlap")
    annotate(ax, bars, overlap, "{:.1f}")

    ax = axes[1, 0]
    n_te = df["n_TE_tracks"]
    bars = draw_bars(ax, n_te, "Number of TE tracks", "(c) Number of TempestExtremes Tracks", log_scale=True)
    annotate(ax, bars, n_te, "{:,.0f}")

    ax = axes[1, 1]
    overlap_1to1 = df["median_overlap_percent_1to1"]
    bars = draw_bars(ax, overlap_1to1, "Median 1-to-1 overlap (%)", "(d) Median 1-to-1 Track Overlap "
                                                                     "(single longest-matching TE track)")
    annotate(ax, bars, overlap_1to1, "{:.1f}")

    for ax in axes.flat:
        for i in range(1, len(df)):
            if df["eps"].iloc[i] != df["eps"].iloc[i - 1]:
                ax.axvline(i - 0.5, linestyle="--", linewidth=1, color="gray", alpha=0.6)

    legend_elements = [Patch(facecolor=eps_colors[eps], edgecolor="black", label=f"eps={eps}") for eps in eps_values]
    legend_elements += [
        Patch(facecolor="white", edgecolor="black", hatch=hatch, label=f"extr_type={extr}")
        for extr, hatch in hatch_types.items() if extr in df["extr_type"].unique()
    ]
    legend_elements += [
        Patch(facecolor="gray", edgecolor="black", alpha=size_alpha[sf], label=f"size_filter={sf}")
        for sf in size_filters
    ]
    fig.legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    if output_filename is not None:
        output_filename = Path(output_filename)
        output_filename.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_filename, dpi=dpi, bbox_inches="tight")
        print(f"Сохранено: {output_filename} (figsize={fig_width_in:.1f}x10in, dpi={dpi}, "
              f"{len(df)} комбинаций{'' if show_annotations else ', подписи над барами отключены - их > ' + str(ANNOTATE_MAX_BARS)})")
    plt.close(fig)

    return df


def plot_pod_vs_overlap_scatter(summary_df: pd.DataFrame, output_filename: Path, title: str) -> None:
    """2026-09-02: точки сделаны заметно прозрачнее (было alpha=0.7 на весь
    маркер целиком, включая чёрную обводку - при полностью совпадающих
    (POD, overlap) у нескольких комбинаций толстая непрозрачная чёрная
    обводка визуально "склеивала" их в одну точку, даже с alpha=0.7 на
    заливке). Теперь заливка и обводка имеют РАЗНУЮ прозрачность (через RGBA-
    цвета + alpha=None у самого scatter, иначе matplotlib принудительно
    ставит один alpha на весь маркер) - заливка полупрозрачная (0.45), обводка
    ещё прозрачнее и тоньше (0.35, linewidth=0.3), чтобы несколько точек в
    одном месте были видны как более тёмное/насыщенное пятно, а не сливались
    в одну сплошную кляксу."""
    extr_types = sorted(summary_df["extr_type"].unique())
    colors = dict(zip(extr_types, plt.cm.tab10.colors))
    eps_values = sorted(summary_df["eps"].unique())
    markers = dict(zip(eps_values, ["o", "s", "^", "D", "v", "P"]))
    size_filters = sorted(summary_df["size_filter"].unique())
    sizes = {sf: 40 + 40 * i for i, sf in enumerate(size_filters)}

    FACE_ALPHA = 0.45
    EDGE_ALPHA = 0.35

    fig, ax = plt.subplots(figsize=(9, 7))
    for extr_type in extr_types:
        for eps in eps_values:
            sub = summary_df[(summary_df["extr_type"] == extr_type) & (summary_df["eps"] == eps)]
            if sub.empty:
                continue
            face_rgba = to_rgba(colors[extr_type], alpha=FACE_ALPHA)
            edge_rgba = to_rgba("black", alpha=EDGE_ALPHA)
            ax.scatter(100 * sub["POD"], sub["median_temporal_overlap_percent"], c=[face_rgba],
                       marker=markers[eps], s=sub["size_filter"].map(sizes), alpha=None,
                       edgecolors=[edge_rgba], linewidths=0.3, label=f"{extr_type}, eps={eps}")

    ax.set_xlabel("POD (%)")
    ax.set_ylabel("Median temporal overlap (%)")
    ax.set_title(f"{title}\nцвет=extr_type, маркер=eps, размер=size_filter")
    ax.grid(linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=7, ncol=2)
    plt.tight_layout()
    output_filename.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Сохранено: {output_filename}")
