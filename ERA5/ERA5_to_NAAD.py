#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
from datetime import datetime
from multiprocessing import Pool, cpu_count
import argparse

from pathlib import Path

# Конфигурационные параметры по умолчанию

# Пути к данным
DEFAULT_INPUT_DIR = Path("/storage/thalassa/DATA/ERA5/PL/NC/uvwth_500hPa")
path_init = f'/storage/thalassa/users/vkoshkina/data'
DEFAULT_OUTPUT_DIR = Path(f'{path_init}/ERA5/ERA5_interp/')
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # Создаем папку, если её нет

DEFAULT_INPUT_DIR = Path("/storage/thalassa/users/gavr/Coherents/ERA5/DATA/ERA5NC")

path_init = f'/storage/thalassa/users/vkoshkina/data'
DEFAULT_OUTPUT_DIR = Path(f'{path_init}/ERA5/ERA5_interp_NAAD/ERA5_heights')
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # Создаем папку, если её нет

# DEFAULT_GRID_FILE = f'{path_init}/ERA5/naad_grid.txt'
DEFAULT_GRID_FILE = f'{path_init}/ERA5/grid_file.nc'
ERA5_GRID_FILE = f'/storage/thalassa/users/vkoshkina/scripts/ERA5/grid.txt'  # ← ДОБАВЛЕНО: grid.txt для ERA5



DEFAULT_VARIABLES = ["u", "v"]
# DEFAULT_LEVEL = "500hPa"
DEFAULT_START_YEAR = 2010
DEFAULT_END_YEAR = 2010
# DEFAULT_END_YEAR = 2024

DEFAULT_METHOD = "remapbil"
DEFAULT_NPROC = cpu_count()

def parse_args():
    """Разбор аргументов командной строки"""
    parser = argparse.ArgumentParser(description="Параллельная переинтерполяция ERA5 -> NAAD")
    parser.add_argument('-i', '--input-dir', default=DEFAULT_INPUT_DIR, 
                       help=f"Директория с входными файлами (по умолчанию: {DEFAULT_INPUT_DIR})")
    parser.add_argument('-o', '--output-dir', default=DEFAULT_OUTPUT_DIR,
                       help=f"Директория для результатов (по умолчанию: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument('-g', '--grid-file', default=DEFAULT_GRID_FILE,
                       help=f"Файл с описанием сетки NAAD (по умолчанию: {DEFAULT_GRID_FILE})")
    parser.add_argument('-e5', '--era5-grid', default=ERA5_GRID_FILE, 
                        help="ERA5 grid.txt файл")
    # parser.add_argument('-l', '--level', default=DEFAULT_LEVEL,
    #                    help=f"Уровень давления (по умолчанию: {DEFAULT_LEVEL})")
    parser.add_argument('-s', '--start-year', type=int, default=DEFAULT_START_YEAR,
                       help=f"Начальный год (по умолчанию: {DEFAULT_START_YEAR})")
    parser.add_argument('-e', '--end-year', type=int, default=DEFAULT_END_YEAR,
                       help=f"Конечный год (по умолчанию: {DEFAULT_END_YEAR})")
    parser.add_argument('-m', '--method', default=DEFAULT_METHOD,
                       choices=['remapbil', 'remapnn', 'remapcon', 'remapdis'],
                       help=f"Метод интерполяции (по умолчанию: {DEFAULT_METHOD})")
    parser.add_argument('-n', '--nproc', type=int, default=DEFAULT_NPROC,
                       help=f"Количество процессов (по умолчанию: {DEFAULT_NPROC})")
    parser.add_argument('--dry-run', action='store_true',
                       help="Показать план обработки без реального выполнения")
    return parser.parse_args()

def check_tools():
    """Проверка наличия необходимых утилит"""
    try:
        subprocess.run(["cdo", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        raise SystemExit("Ошибка: CDO не установлен или не доступен в PATH")

def process_file(args, year, month, day):
    input_fname = f"era5_pl_{year}-{month:02d}-{day:02d}.nc"
    output_fname = f"era5_naad_{year}-{month:02d}-{day:02d}.nc"    
    
    input_path = os.path.join(args.input_dir, input_fname)
    output_path = os.path.join(args.output_dir, output_fname)
    
    if not os.path.exists(input_path):
        return (year, month, day, "skip", "input file not found")
    
    if not os.path.exists(args.era5_grid) or not os.path.exists(args.grid_file):
        return (year, month, day, "error", "grid files missing")
    
    if args.dry_run:
        cmd_str = f"cdo -O -L remapbil,{args.grid_file} -selname,{'/'.join(DEFAULT_VARIABLES)} -setgrid,{args.era5_grid} {input_path} {output_path}"
        return (year, month, day, "dry-run", cmd_str)
    
    # **ОДНА КОМАНДА** - правильный порядок операторов (справа налево):
    cmd = [
        "cdo", "-O", "-L",
        args.method,           # remapbil
        args.grid_file,        # NAAD grid
        "-selname," + ",".join(DEFAULT_VARIABLES),  # u,v (или z,u,v)
        "-setgrid," + args.era5_grid,  # ERA5 grid.txt
        input_path,            # input
        output_path            # output
    ]
    
    start_time = datetime.now()
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        elapsed = (datetime.now() - start_time).total_seconds()
        return (year, month, day, "success", f"{elapsed:.1f} sec")
    except subprocess.CalledProcessError as e:
        error_msg = f"CMD: {' '.join(cmd[:5])}... ERROR: {e.stderr[:200]}"
        return (year, month, day, "error", error_msg)

        
def main():
    args = parse_args()
    
    # Проверка зависимостей
    check_tools()
    
    # Создание выходной директории
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Проверка существования файла с сеткой
    if not os.path.exists(args.grid_file):
        raise SystemExit(f"Файл с описанием сетки {args.grid_file} не найден!")
    
    # Подготовка списка задач
    tasks = [(args, year, month, day) 
             for year in range(args.start_year, args.end_year + 1)
             for month in range(8, 10)
             for day in range(1,32)]
    
    print(f"Параллельная обработка ERA5 -> NAAD ({args.method})")
    print(f"Директория входа: {args.input_dir}")
    print(f"Директория выхода: {args.output_dir}")
    print(f"Сетка: {args.grid_file}")
    # print(f"Уровень: {args.level}")
    print(f"Период: {args.start_year}-{args.end_year}")
    print(f"Процессы: {args.nproc}")
    print(f"Всего задач: {len(tasks)}")
    print("-" * 50)
    
    # Параллельное выполнение
    with Pool(processes=args.nproc) as pool:
        results = pool.starmap(process_file, tasks)
    
    print("\nРезультаты обработки:")
    for year, month, day, status, message in results:
        stats[status] += 1
        if status != "success":
            print(f"{year}-{month:02d}-{day:02d}: {status} - {message}")

    
    print("\nСтатистика:")
    for status, count in stats.items():
        print(f"{status}: {count}")
    
    if stats["success"] > 0:
        print("\nОбработка завершена успешно!")

if __name__ == "__main__":
    main()