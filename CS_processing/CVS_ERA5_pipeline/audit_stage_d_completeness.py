"""
audit_stage_d_completeness.py — проверяет, что Stage D (CSV на трек) полностью
и без усечения отработал для КАЖДОГО уже существующего csv_Tracks{postfix}, а
не только для того одного листа, что был виден в live-выводе терминала.

Зачем это нужно
----------------
До 2026-09-01 запись Stage D в csv_Tracks{postfix} была НЕ атомарной (файлы
писались по одному прямо в целевую папку), а skip-check в старом
run_stage_tracking_leaf()/run_one_leaf() смотрел только "непустая ли папка"
(_dir_has_files) — НЕ "полностью ли она записана". Если процесс прерывался
(Ctrl+C, kill, OOM) посреди конвертации одного года, папка оставалась ЧАСТИЧНО
заполненной — а на следующем запуске ТОГО ЖЕ листа этот skip-check видел
"непустая папка" и МОЛЧА пропускал Stage D, оставляя усечённый набор треков
БЕЗ единой ошибки или предупреждения в логе. Лог мог показать tracking_ok/ok,
как будто всё отработало полностью.

Прямое подтверждение такого случая уже есть в вашем логе
(grid_run_log_NA_for_TC_850hPa.csv): лист maxgap=4h/mintime=9h/prioritize=False
(eps=1, size_filter=10, extr_type=global) был запущен в 2026-08-31T12:01:45
(строка со status=NaN — процесс прерван ДО того, как код успел записать
статус в конце run_stage_tracking_leaf) и ПОВТОРНО в 2026-08-31T12:31:18
(status=tracking_ok). Между этими двумя запусками csv_Tracks_range_1_5_9h_4h
могла остаться частично заполненной первым прерванным прогоном, а второй
прогон её не пересчитал — просто пропустил как "уже не пустая". То есть
это ВТОРОЙ кандидат на усечение, отдельный от того, что виден в live-выводе
StitchNodes прямо сейчас.

Почему нельзя просто довериться логу
--------------------------------------
Строка в grid_run_log.csv пишется в блоке `finally` — если процесс убит через
SIGKILL/OOM (а не Ctrl+C/SIGTERM с нормальным разворачиванием стека), `finally`
вообще не выполнится, и лог НИЧЕГО не покажет для этого прогона — ни строки, ни
NaN-статуса. Такие случаи в принципе невозможно найти по логу, поэтому этот
скрипт проверяет ФАКТИЧЕСКОЕ содержимое папок на диске, а не полагается на лог.

Как проверяется
----------------
Для каждого листа сетки (eps, size_filter, extr_type, maxgap, mintime,
prioritize) x year, для которого существует Tracks_R2D_txt_files{postfix}/..txt
(Stage C уже отработал) И csv_Tracks{postfix} (Stage D вроде бы тоже
отработал):

    ожидаемое_число_треков = число строк "start" в TE_TC_tracks_{year}.txt
                              (формат StitchNodes — ровно одна "start"-строка
                              на трек, не зависит от huracanpy)
    фактическое_число_треков = число *.csv в csv_Tracks{postfix}

Несовпадение = подозрение на усечённую Stage D-папку. НЕ проверяет
huracanpy-специфику (например, отфильтровал ли huracanpy какие-то треки по
другой причине) — если ожидаемое/фактическое расходится на единицы, а не на
порядки, стоит проверить вручную; расхождение "было 27131, стало 4037" —
однозначно усечение.

Пример запуска
--------------
python audit_stage_d_completeness.py --region NA_for_TC --years 2010
python audit_stage_d_completeness.py --region NA_for_TC --years 2010 --fix
    (--fix: подозрительные csv_Tracks{postfix} переименовываются в
    {postfix}.SUSPECT_TRUNCATED_pid{getpid()} - НЕ удаляются автоматически,
    чтобы можно было посмотреть на них перед удалением; --stage csv на новом
    коде пересоздаст их заново, т.к. увидит отсутствие csv_Tracks{postfix}.)
"""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path

grid_runner = importlib.import_module("6_grid_runner_TE_ERA5")
stage_c = importlib.import_module("2_run_StitchNodes_by_year")


def count_start_lines(txt_file: Path) -> int:
    """Число треков в StitchNodes-txt = число строк, начинающихся с 'start'
    (заголовок каждого трека в этом формате) - не требует huracanpy."""
    n = 0
    with open(txt_file, "r", errors="replace") as f:
        for line in f:
            if line.startswith("start"):
                n += 1
    return n


def audit(region: str, years: list[int], fix: bool) -> list[dict]:
    grid_runner.REGION_NAME = f"{region}_{grid_runner.LEVEL_HPA}hPa"
    suspects = []
    checked = 0

    for eps, size_filter, extr_type, maxgap, mintime, prioritize in grid_runner.iter_grid():
        combo_dir = grid_runner.combo_dir_path(eps, size_filter, extr_type)
        postfix = stage_c.build_stitch_postfix(grid_runner.SEARCH_RANGE, mintime, maxgap, prioritize)
        for year in years:
            tracks_txt = combo_dir / f"Tracks_R2D_txt_files{postfix}" / f"{grid_runner.DATA_TYPE}_TC_tracks_{year}.txt"
            csv_tracks_dir = combo_dir / f"csv_Tracks{postfix}"

            if not tracks_txt.exists() or not csv_tracks_dir.is_dir():
                continue  # Stage C и/или Stage D для этого листа ещё не отработали - нечего сверять

            checked += 1
            expected = count_start_lines(tracks_txt)
            actual = len(list(csv_tracks_dir.glob("*.csv")))

            if actual != expected:
                suspects.append({
                    "eps": eps, "size_filter": size_filter, "extr_type": extr_type,
                    "maxgap": maxgap, "mintime": mintime, "prioritize": prioritize, "year": year,
                    "csv_tracks_dir": str(csv_tracks_dir),
                    "expected": expected, "actual": actual,
                })
                print(
                    f"[ПОДОЗРЕНИЕ НА УСЕЧЕНИЕ] eps={eps} size_filter={size_filter} extr_type={extr_type} "
                    f"maxgap={maxgap}h mintime={mintime}h prioritize={prioritize} year={year}: "
                    f"ожидалось {expected} треков (start-строк в {tracks_txt.name}), "
                    f"на диске {actual} csv в {csv_tracks_dir}"
                )
                if fix:
                    suspect_name = csv_tracks_dir.name + f".SUSPECT_TRUNCATED_pid{os.getpid()}"
                    dest = csv_tracks_dir.with_name(suspect_name)
                    csv_tracks_dir.rename(dest)
                    print(f"    -> переименовано в {dest} (запустите --stage csv заново, чтобы пересчитать)")

    print(f"\nПроверено листьев (Stage C и Stage D оба уже есть на диске): {checked}")
    print(f"Подозрительных (усечённых) папок: {len(suspects)}")
    return suspects


def main():
    parser = argparse.ArgumentParser(
        description="Сверяет число CSV-треков в каждой csv_Tracks{postfix} с числом "
                     "'start'-строк в соответствующем Tracks_R2D_txt_files{postfix}/*.txt - "
                     "ищет папки, усечённые старым не-атомарным Stage D (см. докстринг модуля)."
    )
    parser.add_argument("--region", default=grid_runner.REGION_PREFIX, help="Как --region в 6_grid_runner_TE_ERA5.py")
    parser.add_argument("--years", type=int, nargs="+", default=[2010])
    parser.add_argument(
        "--fix", action="store_true",
        help="Переименовать подозрительные csv_Tracks{postfix} в "
             "{postfix}.SUSPECT_TRUNCATED_pid<N> вместо удаления - посмотрите на них, "
             "затем удалите руками и пересчитайте через --stage csv."
    )
    args = parser.parse_args()

    audit(args.region, args.years, args.fix)


if __name__ == "__main__":
    main()
