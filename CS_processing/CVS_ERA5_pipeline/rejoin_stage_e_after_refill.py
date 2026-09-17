"""
Достраивает Stage D (CSV на трек) там, где его ещё нет, и пересобирает Stage E
(джойн ERA5-параметров в CSV-треки) для указанной combo-папки - после того,
как refill_pbl_trop_frac.py / refill_pbl_trop_frac_grid.py обновили общий
params_by_node_{year}.parquet (Stage E', пул size_filter=10) исправленным
pbl_trop_frac (баг с несовпадающей сеткой tropopause, fix 2026-09-02 в
add_params_v2_fast_2026-08-24.py).

2026-09-15: расширено. Раньше скрипт предполагал, что Stage D (csv_Tracks*)
уже посчитан для всех нужных листьев combo-папки, и только пересобирал Stage E
поверх него. На практике (см. ls 01-04-49_local у пользователя) в combo-папке
может быть много уже посчитанных Stage C (Tracks_R2D_txt_files{postfix} -
собственно StitchNodes-треки, по одному на каждую комбинацию
maxgap/mintime/prioritize), но НИ ОДНОГО csv_Tracks{postfix} (Stage D) -
конвертация в CSV на трек для size_filter=49 просто ещё не запускалась.
Поэтому источник истины теперь - Tracks_R2D_txt_files* (Stage C, что реально
уже оттрекано), а не csv_Tracks* (Stage D может отсутствовать вовсе).

Для каждого найденного листа Tracks_R2D_txt_files{postfix}:
  1) Stage D, если csv_Tracks{postfix} ещё пустая/не существует - конвертирует
     tracks_txt в CSV на трек (та же функция, что в 6_grid_runner_TE_ERA5.py,
     3_create_csv_tracks_from_StitchNodes.convert_year). Уже непустая
     csv_Tracks{postfix} НЕ пересчитывается (Stage D не зависит от
     ERA5-параметров, не устаревает от refill) - если очень нужно, --force-stage-d.
  2) Stage E - джойн параметров. Если csv_Tracks{postfix}_params уже
     существует и не пуста (собрана из СТАРОГО, ещё не отрефиленного parquet)
     - переименовывается В СТОРОНУ (.stale_pbl_trop_frac_pid<N>), не
     удаляется, и джойнится заново из --params-parquet.

Все константы/функции стадий (DATA_TYPE, STAGE_D_VARIABLE_NAMES,
STAGE_D_FAST, combo_dir_path, stage_d.convert_year, stage_e.join_combo) -
берутся напрямую из уже загруженного 6_grid_runner_TE_ERA5.py (импортом, как
в refill_pbl_trop_frac_grid.py), а не дублируются здесь - чтобы не разъехаться
с реальными путями/константами пайплайна (та же причина, по которой возник
исходный баг с сеткой tropopause - см. докстринг refill_pbl_trop_frac_grid.py).

ВАЖНО про --params-parquet: Stage E' считается ОДИН РАЗ на (eps, extr_type),
на пуле size_filter=10 (см. докстринг run_stage_e_prime() в
6_grid_runner_TE_ERA5.py) - а не отдельно на каждый size_filter. Поэтому для
--combo-dir .../01-04-49_local нужный parquet лежит НЕ в 01-04-49_local, а в
общей папке пула - .../01-04-10_local/params_by_node_{year}.parquet (тот же
eps и extr_type, size_filter=10). Аналогично для 02-04-49_global - берите
parquet из .../02-04-10_global/, а не .../02-04-49_global/.

Пример запуска
--------------
# весь перебор (все уже оттрекованные maxgap/mintime/prioritize под combo-папкой):
python rejoin_stage_e_after_refill.py \
    --combo-dir /storage/thalassa/users/vkoshkina/data/TempestExtremes/ERA5/R2D_ERA5_NA_for_TC_850hPa_sigma_2/01-04-49_local \
    --params-parquet /storage/thalassa/users/vkoshkina/data/TempestExtremes/ERA5/R2D_ERA5_NA_for_TC_850hPa_sigma_2/01-04-10_local/params_by_node_2010.parquet

python rejoin_stage_e_after_refill.py \
    --combo-dir /storage/thalassa/users/vkoshkina/data/TempestExtremes/ERA5/R2D_ERA5_NA_for_TC_850hPa_sigma_2/02-04-49_global \
    --params-parquet /storage/thalassa/users/vkoshkina/data/TempestExtremes/ERA5/R2D_ERA5_NA_for_TC_850hPa_sigma_2/02-04-10_global/params_by_node_2010.parquet

# 2026-09-15: только один конкретный лист (по maxgap/mintime) - например,
# "01-04-49_local_3h_24h" = maxgap=3h, mintime=24h (порядок как в
# MAXGAP_VALUES/MINTIME_VALUES - НЕ порядок внутри имени папки postfix, там
# сначала mintime, потом maxgap: Tracks_R2D_txt_files_range_1_5_24h_3h). Оба
# варианта prioritize (с ним и без) - если нужен только один, добавьте
# --prioritize true/false:
python rejoin_stage_e_after_refill.py \
    --combo-dir .../01-04-49_local --params-parquet .../01-04-10_local/params_by_node_2010.parquet \
    --maxgap 3 --mintime 24

# то же самое, но точным postfix (как называется папка Tracks_R2D_txt_files*,
# без самого префикса) - для однозначности, если сомневаетесь в порядке выше:
python rejoin_stage_e_after_refill.py \
    --combo-dir .../01-04-49_local --params-parquet .../01-04-10_local/params_by_node_2010.parquet \
    --postfix _range_1_5_24h_3h

# год не указан явно - берётся из имени файла (params_by_node_2010.parquet -> 2010);
# если у вас несколько лет в одной parquet или нестандартное имя - передайте --year явно.
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
import sys
from pathlib import Path
from typing import List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
grid_runner = importlib.import_module("6_grid_runner_TE_ERA5")
stage_d = grid_runner.stage_d
stage_e = grid_runner.stage_e

STAGE_C_PREFIX = "Tracks_R2D_txt_files"


def _dir_has_files(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def find_stage_c_leaves(combo_dir: Path) -> List[Path]:
    """Tracks_R2D_txt_files{postfix} - Stage C (StitchNodes), источник истины
    о том, какие StitchNodes-листья (maxgap/mintime/prioritize) уже оттрекованы
    под этой combo-папкой - в отличие от csv_Tracks* (Stage D), которого может
    вообще не быть ни для одного листа."""
    return sorted(
        p for p in combo_dir.glob(f"{STAGE_C_PREFIX}*")
        if p.is_dir() and ".partial." not in p.name
    )


def postfix_from_stage_c_dir(stage_c_dir: Path) -> str:
    return stage_c_dir.name[len(STAGE_C_PREFIX):]


# _range_{search_range}_{mintime}h_{maxgap}h[_prioritize_r2d] - см.
# build_stitch_postfix() в 2_run_StitchNodes_by_year.py (ЕДИНСТВЕННОЕ место,
# где это формируется - здесь только разбираем обратно для фильтра, саму
# сборку строки не дублируем).
_POSTFIX_RE = re.compile(r"^_range_(?P<range>[\d_]+)_(?P<mintime>\d+)h_(?P<maxgap>\d+)h(?P<prioritize>_prioritize_r2d)?$")


def parse_postfix(postfix: str) -> Optional[dict]:
    m = _POSTFIX_RE.match(postfix)
    if not m:
        return None
    return {
        "mintime": int(m.group("mintime")),
        "maxgap": int(m.group("maxgap")),
        "prioritize": m.group("prioritize") is not None,
    }


def leaf_matches_filter(
    postfix: str, exact_postfix: Optional[str], maxgap: Optional[int],
    mintime: Optional[int], prioritize: Optional[bool],
) -> bool:
    """Без фильтров (--postfix/--maxgap/--mintime/--prioritize все не заданы) -
    пропускает всё (весь перебор, поведение по умолчанию - см. докстринг
    модуля и CLI). --postfix - точное совпадение имени суффикса (без разбора,
    для однозначности). --maxgap/--mintime/--prioritize - разбирают postfix
    (parse_postfix) и сравнивают только заданные измерения; НЕ заданные -
    пропускают любое значение. Postfix нестандартного вида (parse_postfix
    вернул None) не проходит числовой фильтр - лучше явно пропустить лист,
    чем молча взять не тот."""
    if exact_postfix is not None:
        return postfix == exact_postfix
    if maxgap is None and mintime is None and prioritize is None:
        return True
    parsed = parse_postfix(postfix)
    if parsed is None:
        return False
    if maxgap is not None and parsed["maxgap"] != maxgap:
        return False
    if mintime is not None and parsed["mintime"] != mintime:
        return False
    if prioritize is not None and parsed["prioritize"] != prioritize:
        return False
    return True


def process_leaf(
    combo_dir: Path, postfix: str, year: int, params_parquet: Path,
    force_stage_d: bool, fast_join: bool,
) -> None:
    tracks_txt = combo_dir / f"{STAGE_C_PREFIX}{postfix}" / f"{grid_runner.DATA_TYPE}_TC_tracks_{year}.txt"
    csv_tracks_dir = combo_dir / f"csv_Tracks{postfix}"
    final_dir = combo_dir / f"csv_Tracks{postfix}_params"

    if not tracks_txt.exists():
        print(f"  [{postfix}] нет {tracks_txt.name} за {year} год в этом листе - пропуск")
        return

    if force_stage_d or not _dir_has_files(csv_tracks_dir):
        print(f"  [{postfix}] Stage D: {tracks_txt.name} -> {csv_tracks_dir.name} ...")
        stage_d.convert_year(tracks_txt, csv_tracks_dir, grid_runner.STAGE_D_VARIABLE_NAMES, fast=grid_runner.STAGE_D_FAST)
    else:
        print(f"  [{postfix}] {csv_tracks_dir.name} уже не пуста - Stage D пропущен "
              f"(не зависит от ERA5-параметров, не устаревает от refill; --force-stage-d для пересчёта)")

    if not _dir_has_files(csv_tracks_dir):
        print(f"  [{postfix}] {csv_tracks_dir.name} всё ещё пуста после Stage D - джойн параметров пропущен")
        return

    if final_dir.is_dir() and any(final_dir.iterdir()):
        stale_dir = final_dir.with_name(final_dir.name + f".stale_pbl_trop_frac_pid{os.getpid()}")
        print(f"  [{postfix}] {final_dir.name} уже существует (старый джойн) - переименовываю в "
              f"{stale_dir.name} (не удаляю)")
        final_dir.rename(stale_dir)

    print(f"  [{postfix}] Stage E: джойн параметров -> {final_dir.name} (params: {params_parquet.name})")
    stage_e.join_combo(
        tracks_dir=csv_tracks_dir, params_parquet=params_parquet, output_dir=final_dir, fast=fast_join,
    )


def rejoin_combo(
    combo_dir: Path, params_parquet: Path, year: int, force_stage_d: bool = False, fast_join: bool = True,
    exact_postfix: Optional[str] = None, maxgap: Optional[int] = None,
    mintime: Optional[int] = None, prioritize: Optional[bool] = None,
) -> None:
    if not combo_dir.is_dir():
        raise FileNotFoundError(f"{combo_dir} не найдена")

    all_leaves = find_stage_c_leaves(combo_dir)
    if not all_leaves:
        print(f"[{combo_dir}] не найдено ни одной {STAGE_C_PREFIX}* (Stage C/StitchNodes ещё не "
              f"считался для этой combo-папки?) - пропуск")
        return

    filtering = exact_postfix is not None or maxgap is not None or mintime is not None or prioritize is not None
    leaves = [
        p for p in all_leaves
        if leaf_matches_filter(postfix_from_stage_c_dir(p), exact_postfix, maxgap, mintime, prioritize)
    ]

    if filtering:
        print(f"[{combo_dir}] найдено {len(all_leaves)} StitchNodes-листьев всего, "
              f"под фильтр подошло {len(leaves)}: {', '.join(p.name for p in leaves) or '(ничего)'}")
    else:
        print(f"[{combo_dir}] найдено {len(leaves)} StitchNodes-листьев (Stage C, фильтр не задан - "
              f"обрабатываю все): {', '.join(p.name for p in leaves)}")

    for stage_c_dir in leaves:
        postfix = postfix_from_stage_c_dir(stage_c_dir)
        process_leaf(combo_dir, postfix, year, params_parquet, force_stage_d, fast_join)


def _guess_year(params_parquet: Path) -> Optional[int]:
    m = re.search(r"params_by_node_(\d{4})\.parquet$", params_parquet.name)
    return int(m.group(1)) if m else None


def main():
    parser = argparse.ArgumentParser(
        description="Достраивает Stage D (там, где его ещё нет) и пересобирает Stage E (джойн "
                     "ERA5-параметров) для ВСЕХ уже оттрекованных (Stage C) StitchNodes-листьев "
                     "указанной combo-папки - после обновления params_by_node_{year}.parquet "
                     "(например, refill_pbl_trop_frac.py/refill_pbl_trop_frac_grid.py). Старые "
                     "csv_Tracks*_params переименовываются в сторону (не удаляются)."
    )
    parser.add_argument("--combo-dir", required=True, type=Path,
                         help="Папка комбинации eps-min_samples-size_filter_extr_type, например "
                              ".../01-04-49_local (НЕ папка size_filter=10, где лежит сам parquet - "
                              "см. докстринг про --params-parquet).")
    parser.add_argument("--params-parquet", required=True, type=Path,
                         help="Свежий params_by_node_{year}.parquet - обычно из .../<eps>-<min_samples>-10_"
                              "<extr_type>/, общий пул size_filter=10 (см. докстринг run_stage_e_prime "
                              "в 6_grid_runner_TE_ERA5.py).")
    parser.add_argument("--year", type=int, default=None,
                         help="Год (по умолчанию берётся из имени --params-parquet: "
                              "params_by_node_{year}.parquet -> year). Укажите явно, если имя нестандартное.")
    parser.add_argument("--maxgap", type=int, default=None,
                         help="Фильтр: обработать только лист(ья) с этим maxgap (часы), например 3. "
                              "Без --maxgap/--mintime/--prioritize/--postfix обрабатываются ВСЕ уже "
                              "оттрекованные листья (весь перебор) - это поведение по умолчанию.")
    parser.add_argument("--mintime", type=int, default=None,
                         help="Фильтр: обработать только лист(ья) с этим mintime (часы), например 24.")
    parser.add_argument("--prioritize", choices=["true", "false"], default=None,
                         help="Фильтр: true - только листья с _prioritize_r2d, false - только без него. "
                              "Не задано - оба варианта (в сочетании с --maxgap/--mintime).")
    parser.add_argument("--postfix", default=None,
                         help="Фильтр: точное имя суффикса листа, как у папки Tracks_R2D_txt_files{постфикс} "
                              "БЕЗ самого префикса - например --postfix _range_1_5_24h_3h. Однозначнее, чем "
                              "--maxgap/--mintime (там порядок в имени папки mintime_maxgap, а не наоборот) - "
                              "если сомневаетесь, скопируйте имя папки из ls. Взаимоисключающе с --maxgap/"
                              "--mintime/--prioritize (если задано и то, и другое - используется --postfix).")
    parser.add_argument("--force-stage-d", action="store_true",
                         help="Пересчитать Stage D (CSV на трек) заново, даже если csv_Tracks{postfix} "
                              "уже не пуста (обычно не нужно - Stage D не зависит от ERA5-параметров).")
    parser.add_argument("--legacy-loop", action="store_true",
                         help="Медленный файл-за-файлом джойн вместо fast=True (см. join_combo в "
                              "5_join_params_into_tracks.py) - для сверки/отката.")
    args = parser.parse_args()

    if not args.params_parquet.exists():
        raise FileNotFoundError(
            f"{args.params_parquet} не существует - сначала запустите refill_pbl_trop_frac.py / "
            f"refill_pbl_trop_frac_grid.py (или Stage E' целиком), потом уже этот скрипт."
        )

    year = args.year or _guess_year(args.params_parquet)
    if year is None:
        parser.error(
            "Не удалось определить год из имени --params-parquet (ожидается params_by_node_{year}.parquet) "
            "- укажите явно --year."
        )

    prioritize = {"true": True, "false": False, None: None}[args.prioritize]

    rejoin_combo(
        args.combo_dir, args.params_parquet, year,
        force_stage_d=args.force_stage_d, fast_join=not args.legacy_loop,
        exact_postfix=args.postfix, maxgap=args.maxgap, mintime=args.mintime, prioritize=prioritize,
    )


if __name__ == "__main__":
    main()