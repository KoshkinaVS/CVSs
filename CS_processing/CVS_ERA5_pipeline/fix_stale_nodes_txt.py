"""
Автоматическая починка протухших ("устаревших") Nodes-txt (Stage B).

Проблема, которую решает этот скрипт
--------------------------------------
Skip-check в Stage B проверяет только то, что месячный Nodes-txt был ПОЛНОСТЬЮ
и успешно записан (atomic write) - это гарантирует "файл не битый", но НЕ
гарантирует "исходные .nc за этот месяц были все на месте, когда Stage B его
считал". Если на момент запуска Stage B у Stage A (DBSCAN, отдельный процесс/
пайплайн) был готов не весь месяц - Stage B молча запишет "валидный", но
неполный txt, и никакой skip-check дальше этого не заметит - даже если Stage A
позже досчитает оставшиеся дни (см. докстринг audit_nodes_day_coverage.py,
который эти дыры находит и который этот скрипт использует как источник
диагностики).

Раньше починка была ручной: убедиться, что Stage A теперь полон, вручную
удалить protuhший .txt и всё, что от него зависит, перезапустить пайплайн.
Этот скрипт автоматизирует все шаги, КРОМЕ решения "Stage A точно полон?" -
это единственная проверка, которую скрипт делает сам (по числу .nc-файлов за
месяц), и удаляет только те дыры, для которых Stage A уже завершился.

Что именно удаляется для одной "дыры" (eps, size_filter, extr_type, year, month)
------------------------------------------------------------------------------
1. Сам протухший Nodes-txt (Stage B):
       combo_dir/R2D_txt_files_{year}/{DATA_TYPE}_R2D_extr_{year}-{MM}.txt
   Следующий запуск Stage B пересчитает этот месяц с нуля, уже по полным .nc.

2. ВСЕ Stage C (StitchNodes) и Stage D (CSV на трек) выходы ЭТОГО combo_dir
   ЗА ЭТОТ ГОД, по каждому postfix'у (maxgap x mintime x prioritize), который
   реально есть на диске - потому что StitchNodes сшивает узлы ЦЕЛОГО года
   разом, так что дыра в одном месяце портит годовой трек целиком, а не
   только этот месяц:
       combo_dir/Tracks_R2D_txt_files{postfix}/{DATA_TYPE}_TC_tracks_{year}.txt
       combo_dir/csv_Tracks{postfix}/*_track_{year}-*.csv
       combo_dir/csv_Tracks{postfix}_params/*_track_{year}-*.csv   (Stage E, если есть)
   (CSV на трек удаляются точечно, только относящиеся к затронутому году -
   по году в имени файла, NNNNNN_track_YYYY-MM-DDTHH.csv - на случай, если в
   одной папке когда-нибудь окажутся треки нескольких лет.)

3. Если size_filter == 10 (пул узлов - источник Stage E', см. докстринг
   6_grid_runner_TE_ERA5.py "Экономия на Stage E'"): ДОПОЛНИТЕЛЬНО инвалидируется
   Stage E' и всё, что из него собрано:
       combo_dir_10/params_by_node_{year}.parquet                (годовой - целиком)
       combo_dir_10/params_by_node_{year}_monthly/{year}-{MM}.parquet (чекпоинт месяца)
   и Stage E (*_params) выходы ЭТОГО ЖЕ (eps, extr_type, год) для ВСЕХ
   size_filter (10/25/49; п.2 выше их не трогает, т.к. relates только к
   size_filter=10 своему собственному combo_dir) - потому что все они джойнили
   один и тот же (теперь протухший) parquet:
       combo_dir_path(eps, {10,25,49}, extr_type)/csv_Tracks{postfix}_params/*_track_{year}-*.csv

Что НЕ трогается
-----------------
- Годы/месяцы других лет в тех же папках (см. точечное удаление по имени файла).
- Nodes-txt/треки других extr_type/eps/size_filter, не относящихся к дыре.
- Сами исходные .nc (Stage A) - этот скрипт их не пишет и не удаляет.

Проверка "Stage A уже полон?"
------------------------------
Источник Stage A - ОДИН на eps (не зависит от size_filter/extr_type, см.
докстринг 1_create_Nodes_from_DBSCAN.py: DBSCAN считается один раз с
SOURCE_SIZE_FILTER=10, size_filter=25/49 - пост-фильтр поверх того же nc):
    {PATH_INIT}/ERA5/DBSCAN_{eps:02d}-{MIN_SAMPLES:02d}-10_{REGION_NAME}_sigma_{SIGMA}_rad/
        sigma_{SIGMA}_DBSCAN_{DATA_TYPE}_{year}-{MM}-{DD}.nc
Скрипт считает число файлов за месяц и сравнивает с calendar.monthrange - если
меньше ожидаемого, дыра пока НЕ чинится (Stage A ещё не готов), просто
попадает в отчёт как "waiting_for_stage_a" - можно перезапустить скрипт позже.

Использование
-------------
python fix_stale_nodes_txt.py                       # найти и почистить все дыры, для которых Stage A уже полон
python fix_stale_nodes_txt.py --dry-run              # только показать, что было бы удалено, ничего не трогать
python fix_stale_nodes_txt.py --eps 1 --extr-type geom --years 2010
python fix_stale_nodes_txt.py --region Arctic        # другой регион (см. --region ниже)

После читки скрипт сам НЕ перезапускает Stage B/C/D/E' - просто освобождает
skip-check, чтобы следующий обычный запуск 6_grid_runner_TE_ERA5.py
(--stage tracking / --stage eprime / --stage e) пересчитал недостающее.
Список удалённого сохраняется в CSV рядом с sigma_dir (см. --deletions-csv) -
для истории, т.к. удаление автоматическое и без запроса подтверждения.

2026-08-31: --region
---------------------
REGION_NAME (а от него - stage_a_nc_dir(), SIGMA_DIR как default для
--sigma-dir, и combo_dir_path() через grid_runner) раньше читался ТОЛЬКО из
CONFIG 6_grid_runner_TE_ERA5.py в момент импорта - --region в самом
6_grid_runner_TE_ERA5.py на этот (отдельный) процесс не влиял. Теперь при
--region здесь ДОПОЛНИТЕЛЬНО переприсваивается grid_runner.REGION_NAME
(атрибут уже импортированного модуля) - combo_dir_path() читает его как
global НА МОМЕНТ ВЫЗОВА (не на момент импорта), поэтому один раз
переприсвоить в начале main() достаточно, чтобы ВСЕ вызовы combo_dir_path()
ниже по функции корректно использовали новый регион (тот же приём, что и в
--region самого 6_grid_runner_TE_ERA5.py). stage_a_nc_dir() здесь тоже
переведена на чтение grid_runner.REGION_NAME вместо своей локальной
константы-копии - по той же причине.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import importlib
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

grid_runner = importlib.import_module("6_grid_runner_TE_ERA5")
audit = importlib.import_module("audit_nodes_day_coverage")

PATH_INIT = grid_runner.PATH_INIT
DATA_TYPE = grid_runner.DATA_TYPE
SIGMA = grid_runner.SIGMA
MIN_SAMPLES = grid_runner.MIN_SAMPLES
SIZE_FILTER_VALUES = grid_runner.SIZE_FILTER_VALUES
SOURCE_SIZE_FILTER = grid_runner.stage_b_global_local.SOURCE_SIZE_FILTER  # == 10, см. докстринг модуля

# REGION_NAME/SIGMA_DIR НЕ кэшируем здесь как константы (в отличие от прежней
# версии) - оба региозависимы и должны читаться заново после --region в main()
# (см. докстринг модуля "--region"): stage_a_nc_dir() ниже читает
# grid_runner.REGION_NAME напрямую, а sigma_dir - локальная переменная в main().
combo_dir_path = grid_runner.combo_dir_path

DELETIONS_CSV_FIELDS = ["timestamp", "eps", "size_filter", "extr_type", "year", "month", "path", "dry_run"]


# ============================================================================
# 1. Проверка полноты Stage A (источник - один на eps, не на combo)
# ============================================================================

def stage_a_nc_dir(eps: int) -> Path:
    # grid_runner.REGION_NAME (не локальная константа REGION_NAME выше, которая
    # осталась как NA-умолчание, зафиксированное при импорте) - main() ниже
    # переприсваивает grid_runner.REGION_NAME из --region ДО первого вызова
    # этой функции, см. докстринг модуля "--region".
    return Path(
        f"{PATH_INIT}/ERA5/DBSCAN_{eps:02d}-{MIN_SAMPLES:02d}-{SOURCE_SIZE_FILTER:02d}_"
        f"{grid_runner.REGION_NAME}_sigma_{SIGMA}_rad"
    )


def stage_a_day_count(eps: int, year: int, month: int) -> int:
    nc_dir = stage_a_nc_dir(eps)
    if not nc_dir.is_dir():
        return 0
    pattern = f"sigma_{SIGMA}_DBSCAN_{DATA_TYPE}_{year}-{month:02d}-*.nc"
    return len(list(nc_dir.glob(pattern)))


def stage_a_is_complete(eps: int, year: int, month: int) -> bool:
    expected = calendar.monthrange(year, month)[1]
    found = stage_a_day_count(eps, year, month)
    return found >= expected


# ============================================================================
# 2. Удаление (с учётом --dry-run) + лог удалённого
# ============================================================================

def _delete_file(path: Path, deleted: list, dry_run: bool) -> None:
    if path.exists():
        deleted.append(path)
        if not dry_run:
            path.unlink()


def _delete_year_matches(dir_path: Path, year: int, deleted: list, dry_run: bool) -> None:
    if not dir_path.is_dir():
        return
    for f in sorted(dir_path.glob(f"*_track_{year}-*.csv")):
        deleted.append(f)
        if not dry_run:
            f.unlink()


def discover_postfix_dirs(combo_dir: Path) -> list[str]:
    """Все РЕАЛЬНО просчитанные postfix'ы (Tracks_R2D_txt_files{postfix}) в
    combo_dir - не полный перебор сетки maxgap x mintime x prioritize, чтобы
    не трогать то, чего для этой комбинации никогда не считали."""
    prefix = "Tracks_R2D_txt_files"
    postfixes = []
    if not combo_dir.is_dir():
        return postfixes
    for child in sorted(combo_dir.glob(f"{prefix}*")):
        if child.is_dir():
            postfixes.append(child.name[len(prefix):])
    return postfixes


def invalidate_tracking_outputs(combo_dir: Path, year: int, deleted: list, dry_run: bool) -> None:
    """Stage C/D (и Stage E, если уже джойнился) ЭТОГО combo_dir за этот год -
    по каждому найденному на диске postfix'у (см. докстринг модуля, п.2)."""
    for postfix in discover_postfix_dirs(combo_dir):
        tracks_txt = combo_dir / f"Tracks_R2D_txt_files{postfix}" / f"{DATA_TYPE}_TC_tracks_{year}.txt"
        _delete_file(tracks_txt, deleted, dry_run)
        # огрызок прерванного StitchNodes, если остался - см. 2_run_StitchNodes_by_year.py
        _delete_file(Path(str(tracks_txt) + ".partial"), deleted, dry_run)
        _delete_year_matches(combo_dir / f"csv_Tracks{postfix}", year, deleted, dry_run)
        _delete_year_matches(combo_dir / f"csv_Tracks{postfix}_params", year, deleted, dry_run)


def invalidate_stage_e_prime(eps: int, extr_type: str, year: int, month: int, deleted: list, dry_run: bool) -> None:
    """Только когда протухший месяц принадлежит size_filter=10 (источник Stage
    E'). Инвалидирует сам parquet (годовой + месячный чекпоинт) и Stage E
    (*_params) НА ВСЕХ size_filter этого (eps, extr_type, год) - см. докстринг
    модуля, п.3, и докстринг 6_grid_runner_TE_ERA5.py про переиспользование
    Stage E' между size_filter."""
    combo_dir_10 = combo_dir_path(eps, size_filter=10, extr_type=extr_type)
    yearly_parquet = combo_dir_10 / f"params_by_node_{year}.parquet"
    monthly_parquet = combo_dir_10 / f"params_by_node_{year}_monthly" / f"{year}-{month:02d}.parquet"
    _delete_file(yearly_parquet, deleted, dry_run)
    _delete_file(monthly_parquet, deleted, dry_run)

    for sf in SIZE_FILTER_VALUES:
        if sf == SOURCE_SIZE_FILTER:
            continue  # == 10: этот combo_dir уже полностью обработан invalidate_tracking_outputs() выше
        combo_dir_sf = combo_dir_path(eps, sf, extr_type)
        for postfix in discover_postfix_dirs(combo_dir_sf):
            _delete_year_matches(combo_dir_sf / f"csv_Tracks{postfix}_params", year, deleted, dry_run)


def fix_one_hole(eps: int, size_filter: int, extr_type: str, year: int, month: int, dry_run: bool) -> list:
    combo_dir = combo_dir_path(eps, size_filter, extr_type)
    nodes_txt = combo_dir / f"R2D_txt_files_{year}" / f"{DATA_TYPE}_R2D_extr_{year}-{month:02d}.txt"

    deleted: list = []
    _delete_file(nodes_txt, deleted, dry_run)
    invalidate_tracking_outputs(combo_dir, year, deleted, dry_run)
    if size_filter == 10:
        invalidate_stage_e_prime(eps, extr_type, year, month, deleted, dry_run)
    return deleted


# ============================================================================
# main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Автоматически удаляет протухшие (неполные) Nodes-txt Stage B "
                     "и всё, что от них зависит - только там, где Stage A уже полон."
    )
    parser.add_argument("--eps", type=int, default=None)
    parser.add_argument("--size-filter", type=int, default=None)
    parser.add_argument("--extr-type", choices=["global", "local", "geom"], default=None)
    parser.add_argument("--years", type=int, nargs="+", default=None)
    parser.add_argument(
        "--region", default=grid_runner.REGION_PREFIX,
        help=f"Префикс региона (по умолчанию '{grid_runner.REGION_PREFIX}', как в 6_grid_runner_TE_ERA5.py) - "
             f"см. докстринг модуля '--region'. LEVEL_HPA - CONFIG-константа в 6_grid_runner_TE_ERA5.py, "
             f"не CLI (одна на все регионы).",
    )
    parser.add_argument("--sigma-dir", type=Path, default=None,
                         help="По умолчанию вычисляется из --region (см. выше)")
    parser.add_argument("--dry-run", action="store_true", help="Только показать, что будет удалено, ничего не трогать")
    parser.add_argument("--deletions-csv", type=Path, default=None)
    args = parser.parse_args()

    # Переприсваиваем ДО первого использования combo_dir_path()/stage_a_nc_dir()
    # ниже - см. докстринг модуля "--region" про то, почему это достаточно и
    # почему это единственное надёжное место (main() вызывается один раз за
    # процесс, как и в самом 6_grid_runner_TE_ERA5.py).
    grid_runner.REGION_NAME = f"{args.region}_{grid_runner.LEVEL_HPA}hPa"
    sigma_dir = args.sigma_dir if args.sigma_dir is not None else (
        Path(PATH_INIT) / "TempestExtremes" / DATA_TYPE / f"R2D_{DATA_TYPE}_{grid_runner.REGION_NAME}_sigma_{SIGMA}"
    )
    args.sigma_dir = sigma_dir

    combos = audit.discover_combo_dirs(args.sigma_dir, args.eps, args.size_filter, args.extr_type)
    if not combos:
        print(f"Не найдено ни одной combo-папки в {args.sigma_dir} (проверьте фильтры/путь/--region).")
        return

    print(f"Регион: {grid_runner.REGION_NAME}")

    holes = []  # (eps, ms, sf, extr_type, year, month)
    for eps, ms, sf, extr_type, combo_dir in combos:
        years = audit.discover_years(combo_dir, args.years)
        for year in years:
            nodes_dir = combo_dir / f"R2D_txt_files_{year}"
            for month in range(1, 13):
                txt_path = nodes_dir / f"{DATA_TYPE}_R2D_extr_{year}-{month:02d}.txt"
                result = audit.check_month_coverage(txt_path, year, month)
                if result["status"] == "MISSING_DAYS":
                    holes.append((eps, sf, extr_type, year, month))

    if not holes:
        print("Дыр (MISSING_DAYS) не найдено - чинить нечего. "
              "Если ожидали дыры - убедитесь, что stage_b_day_coverage_report.csv свежий "
              "(запустите сначала audit_nodes_day_coverage.py).")
        return

    print(f"Найдено {len(holes)} протухших месяцев (MISSING_DAYS). Проверяю готовность Stage A для каждого...")

    deletion_rows = []
    n_fixed = 0
    n_waiting = 0

    for eps, sf, extr_type, year, month in holes:
        if not stage_a_is_complete(eps, year, month):
            found = stage_a_day_count(eps, year, month)
            expected = calendar.monthrange(year, month)[1]
            print(
                f"⏳ eps{eps:02d}-{MIN_SAMPLES:02d}-{sf:02d}_{extr_type} {year}-{month:02d}: "
                f"Stage A ещё не полон ({found}/{expected} .nc в {stage_a_nc_dir(eps)}) - пропуск, "
                f"не трогаю (перезапустите скрипт позже)."
            )
            n_waiting += 1
            continue

        deleted = fix_one_hole(eps, sf, extr_type, year, month, args.dry_run)
        n_fixed += 1
        verb = "было бы удалено" if args.dry_run else "удалено"
        print(
            f"{'🔎' if args.dry_run else '✓'} eps{eps:02d}-{MIN_SAMPLES:02d}-{sf:02d}_{extr_type} "
            f"{year}-{month:02d}: Stage A полон -> {verb} {len(deleted)} файл(а/ов):"
        )
        for p in deleted:
            print(f"    {p}")
            deletion_rows.append({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "eps": eps, "size_filter": sf, "extr_type": extr_type,
                "year": year, "month": month, "path": str(p), "dry_run": args.dry_run,
            })

    print(f"\n{'='*70}\nИТОГО: {len(holes)} дыр(а/ы) - {n_fixed} починено (Stage A готов), "
          f"{n_waiting} ждут Stage A.\n{'='*70}")
    if n_fixed and not args.dry_run:
        print(
            "Удалённые файлы будут пересчитаны обычным запуском "
            "6_grid_runner_TE_ERA5.py (--stage tracking / --stage eprime / --stage e - "
            "skip-check сам увидит, что файлов больше нет)."
        )

    if deletion_rows and not args.dry_run:
        # Пишем журнал только при реальном удалении - --dry-run не должен оставлять
        # никаких следов на диске (иначе "просто посмотреть" перестаёт быть безопасным).
        csv_out = args.deletions_csv or (args.sigma_dir / "stage_b_fix_deletions_log.csv")
        is_new = not csv_out.exists()
        with open(csv_out, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=DELETIONS_CSV_FIELDS)
            if is_new:
                writer.writeheader()
            writer.writerows(deletion_rows)
        print(f"\nЖурнал удалений дописан в: {csv_out}")
    elif deletion_rows and args.dry_run:
        print(f"\n(--dry-run: журнал удалений НЕ записан на диск - см. список выше)")


if __name__ == "__main__":
    main()
