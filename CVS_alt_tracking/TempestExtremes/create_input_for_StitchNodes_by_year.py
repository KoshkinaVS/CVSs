import os

path_init = '/storage/thalassa/users/vkoshkina/data/TempestExtremes'


years = range(1979, 2025)   # до 2024 включительно
months = range(1, 13)       # от 1 до 12

sigmas = [
    0, 
    2, 4
]


for sigma in sigmas:
    sigma_dir = f"{path_init}/ERA5/R2D_ERA5_level_500_sigma_{sigma}"
    
    # Поддиректории
    input_list_dir = f"{sigma_dir}/Input_list"
    tracks_dir = f"{sigma_dir}/Tracks"
    nodes_dir = f"{sigma_dir}/Nodes"
    
    # Создаём нужные директории
    os.makedirs(input_list_dir, exist_ok=True)
    os.makedirs(tracks_dir, exist_ok=True)
    os.makedirs(nodes_dir, exist_ok=True)

    for year in years:
        base_path_in = f"{sigma_dir}/Input/sigma_{sigma}_R2D_ERA5_level_500"
        base_path_out = f"{nodes_dir}/ERA5_R2D_extr"  # месячные файлы .txt

        # Пути к файлам списков и треков
        input_list_file = f"{input_list_dir}/ERA5_input_r2d_{year}.txt"
        output_list_file = f"{input_list_dir}/ERA5_R2D_extr_{year}.txt"
        output_tracks_file = f"{tracks_dir}/ERA5_TC_tracks_{year}.txt"

        # Генерация списка файлов (полные пути к monthly .txt)
        with open(input_list_file, 'w') as fin, open(output_list_file, 'w') as fout:
            for month in months:
                month_str = f"{month:02d}"
                output_nc = f"{base_path_out}_{year}-{month_str}.txt"
                fout.write(f"{output_nc}\n")

                input_file = f"{base_path_in}_{year}-{month_str}.nc"
                fin.write(input_file + "\n")