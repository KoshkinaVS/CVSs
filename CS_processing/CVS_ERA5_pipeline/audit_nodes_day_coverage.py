"""
Диагностика: не пропущен ли день в помесячных Nodes-txt (Stage B) и, если
досчитан, в помесячных parquet-чекпоинтах Stage E' (params_by_node_{year}_monthly/).

Зачем этот скрипт
------------------
Skip-check в Stage B/C/E' проверяет только то, что файл был ПОЛНОСТЬЮ и
успешно записан (atomic write + rename) — это гарантирует "файл не
битый/недописанный", но НЕ гарантирует "в исходных данных ERA5 не было дыр".
Если для какого-то дня месяца сам исходный .nc-файл кластеризации
(compute_DBSCAN_latlon_with_rad.py, Stage A) отсутствовал на диске в момент
запуска Stage B — Stage B молча его не обработает (не ошибка, просто нет
файла для этого дня в списке `sorted(files)`), месячный Nodes-txt получится
"валидным", но с дырой, и никакой skip-check этого не заметит. Этот скрипт
читает уже готовые Nodes-txt и явно проверяет, что там представлены ВСЕ
календарные дни месяца (и, отдельно, что для каждого месяца, где Stage E'
уже досчитан, число узлов в parquet-чекпоинте совпадает с числом узлов в
исходном txt).

Использование
-------------
# Проверить все комбинации/годы, которые реально есть на диске под sigma_dir
python audit_nodes_day_coverage.py

# Сузить до конкретных eps/size_filter/extr_type/года
python audit_nodes_day_coverage.py --eps 1 --size-filter 10 --extr-type global --years 2010

# Другой регион (по умолчанию - REGION_PREFIX из 6_grid_runner_TE_ERA5.py, см. --region ниже)
python audit_nodes_day_coverage.py --region Arctic

# Только вывести отчёт в консоль, не сохранять CSV
python audit_nodes_day_coverage.py --no-save-csv

Отчёт также сохраняется в CSV (по умолчанию рядом с sigma_dir):
    stage_b_day_coverage_report.csv
одна строка = одна комбинация x год x месяц, со статусом OK / MISSING_DAYS /
NO_TXT_FILE / STAGE_E_PRIME_MISMATCH.

2026-08-31: --region
---------------------
Раньше REGION_NAME (а от него - SIGMA_DIR, использованная как default для
--sigma-dir) читался ТОЛЬКО из CONFIG 6_grid_runner_TE_ERA5.py в момент
импорта этого модуля - --region в САМОМ 6_grid_runner_TE_ERA5.py на это
никак не влиял (это два разных процесса). Теперь --region здесь есть свой,
пересчитывает и --sigma-dir по умолчанию, и путь к grid_run_log (для секции
"дубли в логе" внизу main() - см. lib.read_combo_log(log_path=...) там же).
Если --sigma-dir указан явно - --region на путь не влияет (сигма-папка и так
однозначно задаёт регион), но всё равно используется для правильного
grid_run_log_{REGION_NAME}.csv в секции дублей.
"""

from __future__ import annotations

import argparse
import calendar
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import importlib
grid_runner = importlib.import_module("6_grid_runner_TE_ERA5")
import tracks_comparison_lib as lib

PATH_INIT = grid_runner.PATH_INIT
DATA_TYPE = grid_runner.DATA_TYPE
SIGMA = grid_runner.SIGMA
MIN_SAMPLES = grid_runner.MIN_SAMPLES

# REGION_NAME/SIGMA_DIR НЕ кэшируем здесь как константы (в отличие от прежней
# версии) - оба региозависимы, пересчитываются в main() из --region (см. её
# докстринг "--region") в локальные region_name/sigma_dir.

COMBO_DIR_RE = re.compile(r"^(\d{2})-(\d{2})-(\d{2})_(global|local|geom)$")


# ============================================================================
# 1. Разбор Nodes-txt (тот же формат, что читает 4_compute_era5_params_for_nodes.py)
# ============================================================================

def parse_nodes_txt_day_hours(path: Path) -> tuple[Counter, Counter]:
    """Возвращает (hour_blocks, node_counts):

    hour_blocks[(month, day)]  -> число часовых блоков (строк-заголовков
                                   "year month day n_extrema hour") этого дня;
    node_counts[(month, day)]  -> РЕАЛЬНОЕ число узлов (строк с ведущим табом)
                                   этого дня - то же самое, что фактически
                                   считает read_nodes_txt_file() в
                                   4_compute_era5_params_for_nodes.py (там
                                   каждая '\t'-строка - один узел, а
                                   'n_extrema' из заголовка нигде не
                                   используется как счётчик).

    ВАЖНО: до этого исправления функция считала только hour_blocks и называла
    их n_extrema_total - это давало заниженное число (1 на часовой блок вместо
    реального числа узлов в нём) и стабильно ломало сверку со Stage E'
    (STAGE_E_PRIME_MISMATCH на каждой строке, т.к. в parquet узлов всегда
    больше, чем часовых блоков).
    """
    hour_blocks = Counter()
    node_counts = Counter()
    current_day_key = None
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            if line.startswith("\t"):
                if current_day_key is not None:
                    node_counts[current_day_key] += 1
                continue
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _year, month, day, _n_extrema, _hour = parts
            current_day_key = (int(month), int(day))
            hour_blocks[current_day_key] += 1
    return hour_blocks, node_counts


def check_month_coverage(txt_path: Path, year: int, month: int) -> dict:
    if not txt_path.exists():
        return {"status": "NO_TXT_FILE", "missing_days": None, "n_days_found": 0,
                "n_days_expected": calendar.monthrange(year, month)[1], "hour_blocks_min": None,
                "hour_blocks_max": None, "n_extrema_total": 0}

    hour_blocks, node_counts = parse_nodes_txt_day_hours(txt_path)
    n_days_expected = calendar.monthrange(year, month)[1]
    days_found = sorted(d for (m, d) in hour_blocks.keys() if m == month)
    missing_days = sorted(set(range(1, n_days_expected + 1)) - set(days_found))

    hour_block_counts = [c for (m, d), c in hour_blocks.items() if m == month]
    node_count_values = [c for (m, d), c in node_counts.items() if m == month]
    n_extrema_total = sum(node_count_values)

    return {
        "status": "OK" if not missing_days else "MISSING_DAYS",
        "missing_days": ",".join(str(d) for d in missing_days) if missing_days else "",
        "n_days_found": len(days_found),
        "n_days_expected": n_days_expected,
        "hour_blocks_min": min(hour_block_counts) if hour_block_counts else 0,
        "hour_blocks_max": max(hour_block_counts) if hour_block_counts else 0,
        "n_extrema_total": n_extrema_total,
    }


# ============================================================================
# 2. Кросс-проверка со Stage E' помесячными чекпоинтами
# ============================================================================

def check_stage_e_prime_month(combo_dir: Path, year: int, month: int, expected_n_extrema: int) -> dict | None:
    """None, если чекпоинта для этого месяца ещё нет (не ошибка - Stage E',
    возможно, просто ещё не досчитан до этого месяца/комбинации)."""
    monthly_parquet = combo_dir / f"params_by_node_{year}_monthly" / f"{year}-{month:02d}.parquet"
    if not monthly_parquet.exists():
        return None
    try:
        df = pd.read_parquet(monthly_parquet)
    except Exception as e:
        return {"status": "UNREADABLE_PARQUET", "n_rows": None, "error": str(e)}

    n_rows = len(df)
    if n_rows != expected_n_extrema:
        return {"status": "STAGE_E_PRIME_MISMATCH", "n_rows": n_rows, "error": ""}
    return {"status": "OK", "n_rows": n_rows, "error": ""}


# ============================================================================
# 3. Обход комбинаций, реально существующих на диске под sigma_dir
# ============================================================================

def discover_combo_dirs(sigma_dir: Path, eps=None, size_filter=None, extr_type=None):
    if not sigma_dir.is_dir():
        return []
    found = []
    for child in sorted(sigma_dir.iterdir()):
        if not child.is_dir():
            continue
        m = COMBO_DIR_RE.match(child.name)
        if not m:
            continue
        c_eps, c_ms, c_sf, c_extr = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)
        if eps is not None and c_eps != eps:
            continue
        if size_filter is not None and c_sf != size_filter:
            continue
        if extr_type is not None and c_extr != extr_type:
            continue
        found.append((c_eps, c_ms, c_sf, c_extr, child))
    return found


def discover_years(combo_dir: Path, years_filter=None):
    years = []
    for child in sorted(combo_dir.glob("R2D_txt_files_*")):
        if not child.is_dir():
            continue
        try:
            year = int(child.name.replace("R2D_txt_files_", ""))
        except ValueError:
            continue
        if years_filter and year not in years_filter:
            continue
        years.append(year)
    return years


# ============================================================================
# main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Проверка полноты дней в Nodes-txt (Stage B) и, где есть, в Stage E'-чекпоинтах")
    parser.add_argument("--eps", type=int, default=None)
    parser.add_argument("--size-filter", type=int, default=None)
    parser.add_argument("--extr-type", choices=["global", "local", "geom"], default=None)
    parser.add_argument("--years", type=int, nargs="+", default=None)
    parser.add_argument(
        "--region", default=grid_runner.REGION_PREFIX,
        help=f"Префикс региона (по умолчанию '{grid_runner.REGION_PREFIX}', как в 6_grid_runner_TE_ERA5.py) - "
             f"определяет REGION_NAME='{{region}}_{{LEVEL_HPA}}hPa' и, если --sigma-dir не задан явно, "
             f"путь по умолчанию к sigma_dir, а также к grid_run_log_{{REGION_NAME}}.csv (секция дублей "
             f"в конце отчёта). LEVEL_HPA - CONFIG-константа в 6_grid_runner_TE_ERA5.py, не CLI (одна "
             f"на все регионы) - если для Arctic нужен другой уровень, --region этого не решает.",
    )
    parser.add_argument("--sigma-dir", type=Path, default=None,
                         help="По умолчанию вычисляется из --region (см. выше)")
    parser.add_argument("--no-save-csv", dest="save_csv", action="store_false", default=True)
    parser.add_argument("--csv-out", type=Path, default=None)
    args = parser.parse_args()

    region_name = f"{args.region}_{grid_runner.LEVEL_HPA}hPa"
    sigma_dir = args.sigma_dir if args.sigma_dir is not None else (
        Path(PATH_INIT) / "TempestExtremes" / DATA_TYPE / f"R2D_{DATA_TYPE}_{region_name}_sigma_{SIGMA}"
    )
    combo_log_path = Path(PATH_INIT) / "TempestExtremes" / DATA_TYPE / f"grid_run_log_{region_name}.csv"
    args.sigma_dir = sigma_dir  # дальше по функции используется только args.sigma_dir - оставляем как было

    combos = discover_combo_dirs(args.sigma_dir, args.eps, args.size_filter, args.extr_type)
    if not combos:
        print(f"Не найдено ни одной combo-папки в {args.sigma_dir} (проверьте фильтры/путь/--region).")
        return

    print(f"Регион: {region_name}. Найдено {len(combos)} combo-папок в {args.sigma_dir}")

    rows = []
    n_problems = 0

    for eps, ms, sf, extr_type, combo_dir in combos:
        years = discover_years(combo_dir, args.years)
        if not years:
            continue

        data_type_prefix = f"{DATA_TYPE}_R2D_extr"

        for year in years:
            nodes_dir = combo_dir / f"R2D_txt_files_{year}"
            for month in range(1, 13):
                txt_path = nodes_dir / f"{data_type_prefix}_{year}-{month:02d}.txt"
                result = check_month_coverage(txt_path, year, month)

                stage_e_prime_status = ""
                if result["status"] != "NO_TXT_FILE" and sf == 10:
                    # Stage E' считается только на пуле size_filter=10 (см. докстринг 6_grid_runner_TE_ERA5.py)
                    ep_check = check_stage_e_prime_month(combo_dir, year, month, result["n_extrema_total"])
                    if ep_check is not None:
                        stage_e_prime_status = ep_check["status"]
                        if ep_check["status"] != "OK":
                            n_problems += 1

                if result["status"] != "OK":
                    n_problems += 1
                    marker = "✗"
                else:
                    marker = "✓"

                if result["status"] != "OK" or (stage_e_prime_status and stage_e_prime_status != "OK"):
                    extra = f", Stage E': {stage_e_prime_status}" if stage_e_prime_status else ""
                    print(f"{marker} eps{eps:02d}-{ms:02d}-{sf:02d}_{extr_type} {year}-{month:02d}: "
                          f"{result['status']} (дней {result['n_days_found']}/{result['n_days_expected']}"
                          f"{', пропущены: ' + result['missing_days'] if result['missing_days'] else ''}){extra}")

                rows.append({
                    "eps": eps, "min_samples": ms, "size_filter": sf, "extr_type": extr_type,
                    "year": year, "month": month, "status": result["status"],
                    "missing_days": result["missing_days"], "n_days_found": result["n_days_found"],
                    "n_days_expected": result["n_days_expected"],
                    "hour_blocks_min": result["hour_blocks_min"], "hour_blocks_max": result["hour_blocks_max"],
                    "n_extrema_total": result["n_extrema_total"],
                    "stage_e_prime_status": stage_e_prime_status,
                })

    report_df = pd.DataFrame(rows)
    print(f"\n{'='*70}\nИТОГО: {len(report_df)} (комбинация x год x месяц), проблем: {n_problems}\n{'='*70}")
    if n_problems == 0:
        print("Дыр не найдено - во всех проверенных месяцах представлены все календарные дни, "
              "Stage E'-чекпоинты (где есть) совпадают по числу узлов с Nodes-txt.")

    if args.save_csv:
        csv_out = args.csv_out or (args.sigma_dir / "stage_b_day_coverage_report.csv")
        report_df.to_csv(csv_out, index=False)
        print(f"\nПолный отчёт сохранён: {csv_out}")

    # Дополнительно: проверка дублей (combo, year) в grid_run_log.csv - тот
    # же класс проблемы, что была найдена и исправлена в find_combos()
    # (tracks_comparison_lib.py) - см. её докстринг про SyCLoPS.
    #
    # 2026-08-31: после разделения 6_grid_runner_TE_ERA5.py на независимые
    # --stage tracking/e (см. его докстринг) один и тот же (combo, год)
    # ТЕПЕРЬ ОЖИДАЕМО получает ДВЕ строки за один сквозной прогон - одну от
    # tracking (status tracking_ok/tracking_error), одну от e (status
    # ok/pending_stage_d/pending_stage_e_prime/error). Это больше не
    # "случайный повторный запуск", а нормальная работа раздельных фаз -
    # поэтому дубли считаем ОТДЕЛЬНО внутри каждой фазы (tracking / джойн),
    # а не по (combo, год) в целом, иначе эта проверка стала бы кричать на
    # каждую строку в норме.
    try:
        # log_path=combo_log_path (не lib.read_combo_log() по умолчанию!) -
        # у lib.COMBO_LOG_PATH своя, отдельно закэшированная при импорте lib
        # копия NA-региона (см. докстринг --region выше) - без явного log_path
        # эта секция молча проверяла бы дубли не в том файле для --region Arctic.
        log_df = lib.read_combo_log(log_path=combo_log_path)
    except FileNotFoundError:
        return

    log_df = log_df.copy()
    log_df["_phase"] = log_df["status"].apply(
        lambda s: "tracking (--stage tracking)" if isinstance(s, str) and s.startswith("tracking_")
        else "джойн (--stage e / all)"
    )
    dup_counts = (
        log_df.groupby(lib.COMBO_DIMS + ["year", "_phase"], dropna=False)
        .size()
        .reset_index(name="n_rows")
    )
    dups = dup_counts[dup_counts["n_rows"] > 1]
    if len(dups):
        print(f"\nВнимание: в {combo_log_path} есть {len(dups)} комбинаций (лист x год x фаза), "
              f"посчитанных больше одного раза В ОДНОЙ И ТОЙ ЖЕ ФАЗЕ (grid_run_log.csv - append-only "
              f"лог, это нормально после повторных запусков ОДНОЙ фазы). Само по себе не проблема - "
              f"find_combos()/7_*/8_* уже берут последнюю запись и не задваивают годы, но для справки:")
        print(dups.to_string(index=False))


if __name__ == "__main__":
    main()
