import os

path_init = '/storage/thalassa/users/vkoshkina/data/TempestExtremes'


years = range(1979, 2025)   # до 2024 включительно
months = range(1, 13)       # от 1 до 12

sigmas = [
#     0, 
    2, 
#     4
]

folder = 'R2D_ERA5_NA_for_TC_500hPa_sigma'

for sigma in sigmas:
    base_path_in = f"{path_init}/ERA5/{folder}_{sigma}/Input/sigma_{sigma}_R2D_ERA5_level_500"
    base_path_out = f"{path_init}/ERA5/{folder}_{sigma}/Nodes/ERA5_R2D_extr"

    input_list_file = f"{path_init}/ERA5/{folder}_{sigma}/ERA5_input.txt"
    output_list_file = f"{path_init}/ERA5/{folder}_{sigma}/ERA5_R2D_extr.txt"

    
    # Создаём выходную директорию, если её нет
    output_dir = os.path.dirname(base_path_out)
    os.makedirs(output_dir, exist_ok=True)

    # Записываем списки и создаём пустые .txt файлы
    with open(input_list_file, 'w') as fin, open(output_list_file, 'w') as fout:
        for year in years:
            for month in months:
                month_str = f"{month:02d}"
                input_file = f"{base_path_in}_{year}-{month_str}.nc"
                output_nc = f"{base_path_out}_{year}-{month_str}.txt"
                nodefile_entry = f"{output_nc}\n"
    
                # Пишем в списки
                fin.write(input_file + "\n")
                fout.write(nodefile_entry)
    
                # Создаём пустой файл
                open(output_nc, 'a').close()  # создаёт файл, если его нет

    print(f"Созданы файлы:")
    print(f" - {input_list_file}")
    print(f" - {output_list_file}")
    print(f"Выходная директория: {output_dir}")
    print(f"Создано файлов: {len(years) * len(months)} пустых .txt файлов")