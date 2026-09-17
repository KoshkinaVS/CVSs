"""
Stage G — валидация треков TempestExtremes против IBTrACS (huracanpy), по
сетке комбинаций, реально посчитанных 6_grid_runner_TE_ERA5.py.

Адаптация архивных huracanpy_compare_TC_TE_diff_ERA5_config.py /
huracanpy_compare_TC_diff_TE_POD.py под новую пайплайн-архитектуру:

  - Пути к трекам TE больше не хардкодятся здесь. Раньше это была одна
    (level_hPa, size_filter, extr_type)-сетка с ручными путями вида
    ".../Tracks_R2D_txt_files_range_1_5_18h_12h_2010_{sf}points{extr}/".
    Теперь сетка — (eps, size_filter, extr_type, maxgap, mintime,
    prioritize), и единственный источник правды о том, что реально
    посчитано и где лежит — grid_run_log.csv, который построчно (один лист
    x один год) пишет 6_grid_runner_TE_ERA5.py (см. его докстринг: "Stage G
    ... добавится отдельно и будет проходить по combo_log.csv"). Этот
    скрипт читает лог, группирует строки по комбинации (без года) и берёт
    путь tracks_txt прямо оттуда — никакой отдельной формулы путей здесь
    нет, так что расхождение с Stage C/раннером невозможно в принципе.

  - Для сравнения с IBTrACS годы StitchNodes-параметров ERA5 (Stage E' /
    Stage E, csv_Tracks{postfix}_params) не нужны — matching и POD у
    huracanpy строятся только по геометрии треков (lat/lon/time), которая
    уже есть в самом Tracks_R2D_txt_files{postfix}/*.txt. Поэтому в сетку
    включаются строки лога со status в {"ok", "pending_stage_e_prime"} —
    обоим достаточно того, что Stage C/D отработали (tracks_txt существует),
    Stage E (джойн ERA5-параметров) не блокирует Stage G.

  - Карты по каждому IBTrACS-треку (как в оригинале) — для одной комбинации
    это разумно (первые/десятки картинок), но для всей сетки (потенциально
    сотни комбинаций x годы) это были бы тысячи PNG. Поэтому по умолчанию
    карты ВЫКЛЮЧЕНЫ: без --plot-tracks считаются только сводные метрики
    (POD, overlap, расстояния) в CSV. --plot-tracks требует, чтобы фильтры
    (--eps/--size-filter/--extr-type/--maxgap/--mintime/--prioritize)
    сузили сетку ровно до одной комбинации — иначе скрипт откажется рисовать
    карты и подскажет, каких фильтров не хватает.

  - Сводная визуализация по всей сетке (аналог plot_configuration_summary
    из оригинала) переработана: оригинальная 2x2-таблица баров была жёстко
    заточена под ровно 2 pressure level x 2 size_filter x 2 extr_type (8
    комбинаций). Сейчас комбинаций может быть на порядки больше, поэтому
    вместо неё — (a) горизонтальный bar POD по топ-N комбинациям и (b)
    scatter POD vs медианное временное перекрытие (цвет = extr_type,
    маркер = eps, размер = size_filter), который читаем при любом
    количестве точек и показывает компромисс POD/качество сразу по всей
    сетке. Если нужен привычный 2x2-бар для узкого среза (например,
    фиксированные maxgap/mintime, сравниваем eps x size_filter x
    extr_type) — сузьте сетку фильтрами, тогда он тоже строится (см.
    plot_narrow_grid_bars).

Пример запуска
--------------
# все ok/pending_stage_e_prime комбинации из grid_run_log.csv за 2010 год,
# без карт по трекам (быстро, только сводные CSV/PNG)
python 7_compare_TC_TE_with_IBTrACS.py --years 2010

# то же, но только для одной комбинации, с картами по каждому IBTrACS-треку
python 7_compare_TC_TE_with_IBTrACS.py --years 2010 --eps 1 --size-filter 25 \
    --extr-type global --maxgap 3 --mintime 18 --no-prioritize --plot-tracks

# только посмотреть, какие комбинации найдены в логе, ничего не считать
python 7_compare_TC_TE_with_IBTrACS.py --dry-run
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import huracanpy
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# Берём пути/CONFIG (PATH_INIT, DATA_TYPE, COMBO_LOG_PATH, ...) из
# грид-раннера — единая точка правды, как и combo_dir_path там же.
grid_runner = importlib.import_module("6_grid_runner_TE_ERA5")

# ============================================================================
# CONFIG
# ============================================================================

PATH_INIT = grid_runner.PATH_INIT
DATA_TYPE = grid_runner.DATA_TYPE
COMBO_LOG_PATH = grid_runner.COMBO_LOG_PATH

# Статусы строк grid_run_log.csv, для которых tracks_txt считается готовым
# (Stage E/параметры ERA5 для Stage G не нужны)
ELIGIBLE_STATUSES = {"ok", "pending_stage_e_prime"}

# huracanpy.assess.match — те же значения, что были в архивных скриптах
MAX_DIST = 200   # km
MEAN_DIST = 120  # km
MIN_OVERLAP = 3  # minimum number of matching points

OUTPUT_ROOT = Path(PATH_INIT) / "TempestExtremes" / DATA_TYPE / "TC_comparison_huracanpy"

COMBO_DIMS = ["eps", "size_filter", "extr_type", "maxgap", "mintime", "prioritize", "search_range"]


# ============================================================================
# 1. ЧТЕНИЕ grid_run_log.csv И ГРУППИРОВКА ПО КОМБИНАЦИЯМ
# ============================================================================

def read_combo_log(log_path: Path = COMBO_LOG_PATH) -> pd.DataFrame:
    if not log_path.exists():
        raise FileNotFoundError(
            f"Не найден {log_path} — сначала запустите 6_grid_runner_TE_ERA5.py "
            f"(хотя бы --stage tracks), Stage G читает список готовых комбинаций оттуда."
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


def find_combos(
    log_df: pd.DataFrame,
    years: list[int] | None,
    eps: int | None,
    size_filter: int | None,
    extr_type: str | None,
    maxgap: int | None,
    mintime: int | None,
    prioritize: bool | None,
    statuses: set[str] = ELIGIBLE_STATUSES,
) -> list[dict]:
    """Группирует строки grid_run_log.csv по комбинации (без года), оставляя
    только строки с eligible-статусом и существующим tracks_txt. Возвращает
    список словарей: {**combo_dims, "years": [...], "tracks_txt_by_year": {...}}."""

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

    # tracks_txt должен реально существовать на диске (лог может быть
    # старым, а результат — уже удалён/пересчитывается)
    df = df[df["tracks_txt"].apply(lambda p: isinstance(p, str) and Path(p).exists())]

    if df.empty:
        return []

    combos = []
    for combo_key, group in df.groupby(COMBO_DIMS, dropna=False):
        combo = dict(zip(COMBO_DIMS, combo_key))
        group_sorted = group.sort_values("year")
        combo["years"] = [int(y) for y in group_sorted["year"]]
        combo["tracks_txt_by_year"] = dict(zip(group_sorted["year"], group_sorted["tracks_txt"]))
        combos.append(combo)

    return combos


# ============================================================================
# 2. ЗАГРУЗКА ТРЕКОВ
# ============================================================================

def load_te_tracks_for_combo(combo: dict, variable_names=("rad", "r2d", "wspd")):
    all_tracks = []
    for year, tracks_txt in tqdm(
        combo["tracks_txt_by_year"].items(),
        desc=f"Loading TE tracks {combo_label(combo)}",
        leave=False,
    ):
        try:
            tracks_year = huracanpy.load(
                str(tracks_txt), source="tempestextremes", variable_names=list(variable_names),
            )
            all_tracks.append(tracks_year)
        except FileNotFoundError:
            print(f"  {tracks_txt} не найден, пропуск {year}")
            continue

    if not all_tracks:
        raise FileNotFoundError(f"Не удалось загрузить ни одного года треков для {combo_label(combo)}")

    return huracanpy.concat_tracks(all_tracks)


def load_ibtracs_all() -> "xr.Dataset":
    all_tracks_TC = huracanpy.load(source="ibtracs")
    na_tracks = all_tracks_TC.where(all_tracks_TC.basin == "NA", drop=True)
    return na_tracks


# ============================================================================
# 3. MATCHING / POD / ТРЕКОВАЯ СТАТИСТИКА (логика не менялась —
#    перенесено из huracanpy_compare_TC_TE_diff_ERA5_config.py как есть)
# ============================================================================

def calculate_matches(ibtracs_tracks, te_tracks):
    return huracanpy.assess.match(
        [ibtracs_tracks, te_tracks],
        names=["IBTrACS", "TE_NA"],
        max_dist=MAX_DIST,
        mean_dist=MEAN_DIST,
        min_overlap=MIN_OVERLAP,
        tracks1_is_ref=True,
    )


def calculate_pod(matches, ibtracs_tracks):
    return float(huracanpy.assess.pod(matches, ref=ibtracs_tracks, ref_name="IBTrACS"))


def select_best_matches(matches):
    if len(matches) == 0:
        return matches.copy()
    return (
        matches
        .sort_values(["id_IBTrACS", "temp", "dist"], ascending=[True, False, True])
        .drop_duplicates(subset="id_IBTrACS", keep="first")
        .reset_index(drop=True)
    )


def calculate_temporal_overlap(ib_track, te_track):
    ib_times = pd.to_datetime(ib_track.time.values)
    te_times = pd.to_datetime(te_track.time.values)

    if len(ib_times) == 0 or len(te_times) == 0:
        return {
            "overlap_start": pd.NaT, "overlap_end": pd.NaT,
            "overlap_duration_hours": 0, "overlap_duration_fraction": 0,
        }

    ib_start, ib_end = ib_times.min(), ib_times.max()
    te_start, te_end = te_times.min(), te_times.max()
    ib_duration = (ib_end - ib_start).total_seconds() / 3600

    overlap_start = max(ib_start, te_start)
    overlap_end = min(ib_end, te_end)
    overlap_duration = 0 if overlap_start > overlap_end else (overlap_end - overlap_start).total_seconds() / 3600
    fraction = overlap_duration / ib_duration if ib_duration > 0 else 0

    return {
        "overlap_start": overlap_start, "overlap_end": overlap_end,
        "overlap_duration_hours": overlap_duration, "overlap_duration_fraction": fraction,
    }


def calculate_track_statistics(ibtracs_tracks, te_tracks, matches):
    ib_ids = np.unique(ibtracs_tracks.track_id.values)
    rows = []

    best_matches = select_best_matches(matches)
    best_match_dict = {row["id_IBTrACS"]: row for _, row in best_matches.iterrows()}

    for ib_id in ib_ids:
        ib_track = ibtracs_tracks.where(ibtracs_tracks.track_id == ib_id, drop=True).sortby("time")
        ib_times = pd.to_datetime(ib_track.time.values)
        n_ib_points = len(ib_times)
        ib_duration_hours = (
            (ib_times.max() - ib_times.min()).total_seconds() / 3600 if n_ib_points > 1 else 0
        )

        if ib_id not in best_match_dict:
            rows.append({
                "IBTrACS_ID": ib_id, "detected": False, "TE_ID_best": np.nan,
                "IBTrACS_n_points": n_ib_points, "IBTrACS_duration_hours": ib_duration_hours,
                "matched_n_points": 0, "overlap_fraction": 0, "overlap_percent": 0,
                "mean_distance_km": np.nan, "overlap_start": pd.NaT, "overlap_end": pd.NaT,
                "overlap_duration_hours": 0, "overlap_duration_fraction": 0, "overlap_duration_percent": 0,
            })
            continue

        match_row = best_match_dict[ib_id]
        te_id = match_row["id_TE_NA"]
        matched_points = match_row["temp"]
        mean_distance = match_row["dist"]
        overlap_fraction = matched_points / n_ib_points if n_ib_points > 0 else np.nan

        te_track = te_tracks.where(te_tracks.track_id == te_id, drop=True).sortby("time")
        temporal = calculate_temporal_overlap(ib_track, te_track)

        rows.append({
            "IBTrACS_ID": ib_id, "detected": True, "TE_ID_best": te_id,
            "IBTrACS_n_points": n_ib_points, "IBTrACS_duration_hours": ib_duration_hours,
            "matched_n_points": matched_points, "overlap_fraction": overlap_fraction,
            "overlap_percent": 100 * overlap_fraction, "mean_distance_km": mean_distance,
            "overlap_start": temporal["overlap_start"], "overlap_end": temporal["overlap_end"],
            "overlap_duration_hours": temporal["overlap_duration_hours"],
            "overlap_duration_fraction": temporal["overlap_duration_fraction"],
            "overlap_duration_percent": 100 * temporal["overlap_duration_fraction"],
        })

    return pd.DataFrame(rows)


# ============================================================================
# 4. КАРТА ОДНОГО IBTrACS-ТРЕКА (опционально, --plot-tracks)
# ============================================================================

def plot_track_comparison(ibtracs_track, te_tracks, ibtracs_id, year, output_filename):
    te_ids = np.unique(te_tracks.track_id.values) if te_tracks is not None else []

    fig = plt.figure(figsize=(12, 8), dpi=150)
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.set_extent([-110, 15, 0, 73], ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue", alpha=0.3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
    ax.gridlines(draw_labels=True, linestyle="--", alpha=0.5)

    ax.plot(ibtracs_track.lon, ibtracs_track.lat, color="black", linewidth=3,
            transform=ccrs.PlateCarree(), label="IBTrACS")
    ax.plot(ibtracs_track.lon[0], ibtracs_track.lat[0], "go", markersize=8,
            transform=ccrs.PlateCarree(), label="Genesis")
    ax.plot(ibtracs_track.lon[-1], ibtracs_track.lat[-1], "rs", markersize=8,
            transform=ccrs.PlateCarree(), label="Lysis")

    map_colors = ["#FF1493", "#CC00FF", "#00CC00", "#FF0000", "#FF8C00", "#FFD700",
                  "#FF4500", "#7FFF00", "#FF6347", "#FF00FF", "#FFA500", "#ADFF2F"]
    colors = map_colors * (len(te_ids) // len(map_colors) + 1)

    for i, te_id in enumerate(te_ids):
        te_track = te_tracks.where(te_tracks.track_id == te_id, drop=True).sortby("time")
        ax.plot(te_track.lon, te_track.lat, color=colors[i], linewidth=1.5, alpha=0.8,
                transform=ccrs.PlateCarree(), label=f"TE {te_id}")

    storm_name = ibtracs_track.attrs.get("name", "Unknown") if hasattr(ibtracs_track, "attrs") else "Unknown"
    ax.set_title(f"IBTrACS {ibtracs_id} ({storm_name}) - {year}\n"
                 f"Black: IBTrACS | Colored: TempestExtremes ({len(te_ids)} tracks)")
    ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    plt.savefig(output_filename, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================================
# 5. ОДНА КОМБИНАЦИЯ: matching + статистика + (опционально) карты
# ============================================================================

def run_combo(combo: dict, ibtracs_all, plot_tracks: bool, force: bool) -> dict:
    label = combo_label(combo)
    combo_dir = OUTPUT_ROOT / label
    matches_csv = combo_dir / "matches_raw.csv"
    stats_csv = combo_dir / "matching_statistics.csv"

    if not force and stats_csv.exists():
        print(f"[{label}] {stats_csv} уже есть, пропуск (используйте --force для пересчёта)")
        stats_df = pd.read_csv(stats_csv)
        matches = pd.read_csv(matches_csv) if matches_csv.exists() else pd.DataFrame()
        return _summarize(combo, stats_df, matches, n_te_tracks=None)

    combo_dir.mkdir(parents=True, exist_ok=True)

    te_tracks = load_te_tracks_for_combo(combo)
    ibtracs = ibtracs_all.where(
        ibtracs_all.time.dt.year.isin(combo["years"]), drop=True,
    )

    print(f"[{label}] IBTrACS: {ibtracs.track_id.hrcn.nunique()} треков, "
          f"TE: {te_tracks.track_id.hrcn.nunique()} треков ({combo['years']})")

    matches = calculate_matches(ibtracs, te_tracks)
    matches.to_csv(matches_csv, index=False)

    POD = calculate_pod(matches, ibtracs)
    stats_df = calculate_track_statistics(ibtracs, te_tracks, matches)
    stats_df.to_csv(stats_csv, index=False)

    print(f"[{label}] POD={POD:.3f}, matched={int(stats_df.detected.sum())}/{len(stats_df)}")

    if plot_tracks:
        _plot_all_tracks(combo, combo_dir, ibtracs, te_tracks, matches)

    return _summarize(combo, stats_df, matches, n_te_tracks=int(te_tracks.track_id.hrcn.nunique()), pod=POD)


def _plot_all_tracks(combo, combo_dir, ibtracs, te_tracks, matches):
    tracks_output_dir = combo_dir / "tracks"
    ib_ids = np.unique(ibtracs.track_id.values)
    for i, ib_id in enumerate(ib_ids, start=1):
        try:
            ib_track = ibtracs.where(ibtracs.track_id == ib_id, drop=True).sortby("time")
            year = int(pd.Timestamp(ib_track.time.values[0]).year)
            te_ids = set(matches[matches["id_IBTrACS"] == ib_id]["id_TE_NA"].values)
            te_tracks_for_ib = (
                te_tracks.where(te_tracks.track_id.isin(list(te_ids)), drop=True) if te_ids else None
            )
            filename = tracks_output_dir / f"IBTrACS_{ib_id}_{year}_track{i}.png"
            plot_track_comparison(ib_track, te_tracks_for_ib, ib_id, year, str(filename))
        except Exception as e:
            print(f"  ERROR plotting IBTrACS {ib_id}: {e}")
            continue
    print(f"  Карты сохранены в {tracks_output_dir}")


def _summarize(combo, stats_df, matches, n_te_tracks, pod=None) -> dict:
    n_ibtracs = len(stats_df)
    n_matched = int(stats_df.detected.sum()) if n_ibtracs else 0
    detected = stats_df[stats_df.detected] if n_ibtracs else stats_df

    def _agg(col, fn):
        return getattr(detected[col], fn)() if len(detected) else np.nan

    return {
        **{k: combo[k] for k in COMBO_DIMS},
        "years": ",".join(str(y) for y in combo["years"]),
        "n_IBTrACS": n_ibtracs,
        "n_matched": n_matched,
        "n_unmatched": n_ibtracs - n_matched,
        "POD": pod if pod is not None else (n_matched / n_ibtracs if n_ibtracs else np.nan),
        "mean_overlap_percent": _agg("overlap_percent", "mean"),
        "median_overlap_percent": _agg("overlap_percent", "median"),
        "mean_temporal_overlap_percent": _agg("overlap_duration_percent", "mean"),
        "median_temporal_overlap_percent": _agg("overlap_duration_percent", "median"),
        "mean_distance_km": _agg("mean_distance_km", "mean"),
        "median_distance_km": _agg("mean_distance_km", "median"),
        "n_TE_tracks": n_te_tracks,
    }


# ============================================================================
# 6. СВОДНЫЕ ГРАФИКИ ПО ВСЕЙ НАЙДЕННОЙ СЕТКЕ
# ============================================================================

def plot_pod_bar(summary_df: pd.DataFrame, output_filename: Path, top_n: int = 30) -> None:
    df = summary_df.sort_values("POD", ascending=False).head(top_n)
    labels = [combo_label(row) for _, row in df.iterrows()]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(df))))
    ax.barh(labels[::-1], (100 * df["POD"])[::-1], color="tab:blue")
    ax.set_xlabel("POD (%)")
    ax.set_title(f"Top {len(df)} комбинаций по POD (из {len(summary_df)} найденных в grid_run_log.csv)")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Сохранено: {output_filename}")


def plot_pod_vs_overlap_scatter(summary_df: pd.DataFrame, output_filename: Path) -> None:
    extr_types = sorted(summary_df["extr_type"].unique())
    colors = dict(zip(extr_types, plt.cm.tab10.colors))
    eps_values = sorted(summary_df["eps"].unique())
    markers = dict(zip(eps_values, ["o", "s", "^", "D", "v", "P"]))
    size_filters = sorted(summary_df["size_filter"].unique())
    sizes = {sf: 40 + 40 * i for i, sf in enumerate(size_filters)}

    fig, ax = plt.subplots(figsize=(9, 7))
    for extr_type in extr_types:
        for eps in eps_values:
            sub = summary_df[(summary_df["extr_type"] == extr_type) & (summary_df["eps"] == eps)]
            if sub.empty:
                continue
            ax.scatter(
                100 * sub["POD"], sub["median_temporal_overlap_percent"],
                c=[colors[extr_type]], marker=markers[eps],
                s=sub["size_filter"].map(sizes),
                alpha=0.7, edgecolors="black", linewidths=0.4,
                label=f"{extr_type}, eps={eps}",
            )

    ax.set_xlabel("POD (%)")
    ax.set_ylabel("Median temporal overlap (%)")
    ax.set_title(
        f"POD vs temporal overlap по {len(summary_df)} комбинациям\n"
        f"цвет=extr_type, маркер=eps, размер=size_filter"
    )
    ax.grid(linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Сохранено: {output_filename}")


def plot_narrow_grid_bars(summary_df: pd.DataFrame, output_filename: Path) -> pd.DataFrame:
    """Аналог оригинального 2x2 bar-графика (POD / overlap / n_TE_tracks /
    distance) — имеет смысл только для узкого среза сетки (варьируются
    максимум 2 измерения), иначе подписи станут нечитаемыми. Вызывайте
    только когда summary_df уже отфильтрован CLI-флагами до небольшой
    подсетки (см. докстринг модуля)."""

    df = summary_df.copy()
    x = np.arange(len(df))
    labels = [combo_label(row) for _, row in df.iterrows()]

    fig, axes = plt.subplots(2, 2, figsize=(max(10, 0.6 * len(df)), 10))

    def draw_bars(ax, values, ylabel, title, log_scale=False):
        ax.bar(x, values, color="tab:blue", edgecolor="black", linewidth=0.8)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
        if log_scale:
            ax.set_yscale("log")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)

    draw_bars(axes[0, 0], 100 * df["POD"], "POD (%)", "(a) Probability of Detection")
    draw_bars(axes[0, 1], df["median_overlap_percent"], "Median overlap (%)", "(b) Median Overlap")
    draw_bars(axes[1, 0], df["n_TE_tracks"], "Number of TE tracks", "(c) Number of TE tracks", log_scale=True)
    draw_bars(axes[1, 1], df["median_distance_km"], "Median distance (km)", "(d) Median Distance")

    fig.suptitle("TempestExtremes vs IBTrACS — выбранная подсетка", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Сохранено: {output_filename}")
    return df


# ============================================================================
# main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stage G: сравнение треков TempestExtremes с IBTrACS (huracanpy), "
                     "по комбинациям, найденным в grid_run_log.csv (6_grid_runner_TE_ERA5.py)"
    )
    parser.add_argument("--years", type=int, nargs="+", default=None, help="По умолчанию — все годы, найденные в логе")
    parser.add_argument("--eps", type=int, default=None)
    parser.add_argument("--size-filter", type=int, default=None)
    parser.add_argument("--extr-type", choices=["global", "local", "geom"], default=None)
    parser.add_argument("--maxgap", type=int, default=None)
    parser.add_argument("--mintime", type=int, default=None)
    prioritize_group = parser.add_mutually_exclusive_group()
    prioritize_group.add_argument("--prioritize", dest="prioritize", action="store_true", default=None)
    prioritize_group.add_argument("--no-prioritize", dest="prioritize", action="store_false")
    parser.add_argument("--plot-tracks", action="store_true",
                         help="Рисовать карту по каждому IBTrACS-треку (только если фильтры сузили сетку до 1 комбинации)")
    parser.add_argument("--force", action="store_true", help="Пересчитать, даже если matching_statistics.csv уже есть")
    parser.add_argument("--narrow-grid-plot", action="store_true",
                         help="Дополнительно построить 2x2 bar-график (плюс к POD-bar/scatter) — только для небольшой найденной подсетки")
    parser.add_argument("--dry-run", action="store_true", help="Показать найденные комбинации и выйти")
    args = parser.parse_args()

    log_df = read_combo_log()
    combos = find_combos(
        log_df, years=args.years, eps=args.eps, size_filter=args.size_filter,
        extr_type=args.extr_type, maxgap=args.maxgap, mintime=args.mintime, prioritize=args.prioritize,
    )

    if not combos:
        print("Не найдено ни одной подходящей комбинации в grid_run_log.csv (проверьте фильтры/статусы).")
        return

    print(f"Найдено {len(combos)} комбинаций (после фильтров) в {COMBO_LOG_PATH}:")
    for combo in combos:
        print(f"  {combo_label(combo)} | years={combo['years']}")

    if args.dry_run:
        print("--dry-run: выполнение пропущено")
        return

    if args.plot_tracks and len(combos) > 1:
        print(
            f"\n--plot-tracks запрошен, но фильтры оставили {len(combos)} комбинаций — "
            f"карты по трекам рисуются только для РОВНО ОДНОЙ комбинации (иначе счёт на тысячи PNG). "
            f"Сузьте --eps/--size-filter/--extr-type/--maxgap/--mintime/--prioritize. "
            f"Сводные метрики (без карт) всё равно будут посчитаны для всех {len(combos)}."
        )
        plot_tracks_effective = False
    else:
        plot_tracks_effective = args.plot_tracks

    ibtracs_all = load_ibtracs_all()
    print(f"IBTrACS (NA basin) загружен: {ibtracs_all.track_id.hrcn.nunique()} треков всего")

    summaries = []
    for combo in combos:
        try:
            summaries.append(run_combo(combo, ibtracs_all, plot_tracks=plot_tracks_effective, force=args.force))
        except Exception as e:
            print(f"ОШИБКА для {combo_label(combo)}: {e}")
            continue

    summary_df = pd.DataFrame(summaries)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_csv = OUTPUT_ROOT / "summary_all_configurations.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nСводная таблица сохранена: {summary_csv}")
    print(summary_df.to_string(index=False))

    plot_pod_bar(summary_df, OUTPUT_ROOT / "summary_pod_bar.png")
    if len(summary_df) > 1:
        plot_pod_vs_overlap_scatter(summary_df, OUTPUT_ROOT / "summary_pod_vs_overlap.png")
    if args.narrow_grid_plot:
        plot_narrow_grid_bars(summary_df, OUTPUT_ROOT / "summary_narrow_grid_bars.png")


if __name__ == "__main__":
    main()
