"""
NAAD-обвязка над tracks_comparison_lib.py (ERA5) — используется
7_compare_TE_with_reference_NAAD.py (IBTrACS/SyCLoPS) и
8_compare_TE_with_EddyClicker_NAAD.py (EddyClicker).

Почему не отдельная копия всего файла: подавляющая часть tracks_comparison_lib.py
(huracanpy.assess.match/pod, трек-статистика, ВСЕ графики, парсинг SyCLoPS/
EddyClicker) не знает и не должна знать про ERA5 vs NAAD — она работает только
с track_id/lon/lat/time. Дублировать это ради двух функций — верный способ
рассинхронизироваться при следующей правке оригинала. Поэтому здесь:

- всё, что НЕ зависит от типа данных, импортируется напрямую из
  tracks_comparison_lib (реэкспорт ниже) — включая find_combos() и
  load_te_tracks_for_combo(), которые (проверено чтением исходника,
  2026-09-05) НИГДЕ не используют grid_runner/DATA_TYPE/PATH_INIT — работают
  только с уже переданным log_df / combo["tracks_txt_by_year"];
- переопределены только 2 места, которые в оригинале жёстко читают
  6_grid_runner_TE_ERA5 (см. его импорт `grid_runner = importlib.
  import_module("6_grid_runner_TE_ERA5")` в tracks_comparison_lib.py):
  read_combo_log() (путь к grid_run_log_{region}.csv) и combo_output_dir()
  (папка результатов сравнения внутри combo-папки).

Почему это функция configure(), а не модульные константы (как PATH_INIT/
DATA_TYPE/... в оригинале)
--------------------------------------------------------------------------
У ERA5 DATA_TYPE="ERA5" — константа файла, один грид-раннер = один тип
данных, поэтому пути можно вычислить один раз при импорте. У NAAD один
параметризуемый набор скриптов обслуживает и LoRes, и HiRes (--data-type,
см. 1_create_Nodes_from_DBSCAN_NAAD.py/6_grid_runner_TE_NAAD.py) — значит
путь к grid_run_log.csv и к combo-папкам известен только ПОСЛЕ разбора
argparse в вызывающем CLI-скрипте, не при импорте модуля. Поэтому здесь
configure(data_type=...) вызывается один раз в начале main() того скрипта,
до первого read_combo_log()/find_combos()/combo_output_dir().

2026-09-05: не прогонялось на реальных данных (нет доступа к серверу из этой
сессии) — сама привязка (какие функции реально не зависят от ERA5-констант)
проверена чтением исходника tracks_comparison_lib.py, а не запуском.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import naad_config as cfg

# Всё, что НЕ зависит от ERA5 vs NAAD, - как есть из существующей библиотеки.
# (Импорт tracks_comparison_lib попутно импортирует 6_grid_runner_TE_ERA5 и
# ERA5-стадии 1-5 - безобидный побочный эффект, они не используются отсюда,
# просто должны быть импортируемы - т.е. xarray/huracanpy/tqdm должны быть
# установлены, что и так нужно для собственно NAAD-пайплайна.)
from tracks_comparison_lib import (  # noqa: F401 - часть реэкспортируется для вызывающих скриптов
    ELIGIBLE_STATUSES,
    COMBO_DIMS,
    SYCLOPS_TYPES,
    SYCLOPS_ROOT,
    MAX_DIST,
    MEAN_DIST,
    MIN_OVERLAP,
    combo_label,
    stitch_sub_label,
    find_combos,
    load_te_tracks_for_combo,
    dataframe_to_huracanpy,
    build_csv_for_huracanpy,
    normalize_lon_360_to_180,
    find_syclops_files,
    load_syclops_tracks,
    load_eddyclicker_tracks,
    calculate_matches,
    calculate_pod,
    select_best_matches,
    calculate_temporal_overlap,
    calculate_track_statistics,
    summarize_track_statistics,
    sort_summary_by_combined_score,
    merge_summary_tables,
    plot_reference_track,
    plot_all_reference_tracks,
    plot_best_single_match_tracks,
    plot_pod_bar,
    plot_configuration_summary,
    plot_pod_vs_overlap_scatter,
)

# Заполняются configure() - до вызова осмысленно None (падать с понятной
# ошибкой при использовании до configure(), а не тихо ловить None где-то в
# середине пайплайна).
PATH_INIT: str | None = None
DATA_TYPE: str | None = None
REGION_NAME: str | None = None
SIGMA: int | None = None
COMBO_LOG_PATH: Path | None = None
SIGMA_DIR: Path | None = None


def configure(
    data_type: str,
    region_name: str = cfg.DEFAULT_REGION_NAME,
    sigma: int = cfg.DEFAULT_SIGMA,
    path_init: str = cfg.PATH_INIT,
) -> None:
    """Вызвать ОДИН РАЗ в начале CLI-скрипта (после argparse), до любых
    read_combo_log()/find_combos()/combo_output_dir()/combo_dir_path()."""
    global PATH_INIT, DATA_TYPE, REGION_NAME, SIGMA, COMBO_LOG_PATH, SIGMA_DIR
    PATH_INIT = path_init
    DATA_TYPE = data_type
    REGION_NAME = region_name
    SIGMA = sigma
    COMBO_LOG_PATH = Path(PATH_INIT) / "TempestExtremes" / DATA_TYPE / f"grid_run_log_{REGION_NAME}.csv"
    SIGMA_DIR = Path(PATH_INIT) / "TempestExtremes" / DATA_TYPE / f"R2D_{DATA_TYPE}_{REGION_NAME}_sigma_{SIGMA}"


def _check_configured() -> None:
    if DATA_TYPE is None:
        raise RuntimeError(
            "tracks_comparison_lib_NAAD используется до вызова configure(data_type=...) - "
            "вызовите его в начале main() (после argparse), см. докстринг модуля."
        )


def combo_dir_path(eps: int, size_filter: int, extr_type: str) -> Path:
    """Та же формула, что 1_create_Nodes_from_DBSCAN_NAAD.py (run_for_combo)
    и 6_grid_runner_TE_NAAD.py (combo_dir_path) - единая точка правды."""
    _check_configured()
    min_samples = cfg.DEFAULT_MIN_SAMPLES
    return Path(
        f"{PATH_INIT}/TempestExtremes/{DATA_TYPE}/R2D_{DATA_TYPE}_{REGION_NAME}_sigma_{SIGMA}/"
        f"{eps:02d}-{min_samples:02d}-{size_filter:02d}_{extr_type}"
    )


def read_combo_log(log_path: Path | None = None) -> pd.DataFrame:
    """Идентично read_combo_log() в tracks_comparison_lib.py, только путь по
    умолчанию - NAAD-лог (зависит от configure(), не константа файла)."""
    _check_configured()
    log_path = log_path or COMBO_LOG_PATH
    if not log_path.exists():
        raise FileNotFoundError(
            f"Не найден {log_path} — сначала запустите 6_grid_runner_TE_NAAD.py "
            f"(хотя бы --stage tracking) для --data-type {DATA_TYPE} --region {REGION_NAME}."
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


def combo_output_dir(combo: dict, reference_subpath) -> Path:
    """Идентично combo_output_dir() в tracks_comparison_lib.py, только через
    NAAD-версию combo_dir_path() выше (не grid_runner.combo_dir_path,
    завязанный на ERA5)."""
    _check_configured()
    combo_dir = combo_dir_path(combo["eps"], combo["size_filter"], combo["extr_type"])
    return combo_dir / "data_comparison_huracanpy" / Path(reference_subpath) / stitch_sub_label(combo)
