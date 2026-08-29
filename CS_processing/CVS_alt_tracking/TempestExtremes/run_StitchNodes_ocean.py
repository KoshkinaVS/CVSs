import os
from tqdm import tqdm
import calendar

path_TE = '/storage/thalassa/users/vkoshkina/data/TempestExtremes'


sigmas = [
#     4,
    # 2,
    0,
]


in_file_name_list = ['LV_alt_0125deg_2008-2009', 'LV_alt_Novoselova_0125', 'LV_alt_0125deg_2014',
                         # 'LV_alt_025deg_2008-2009', 'LV_alt_Novoselova_025', 'LV_alt_025deg_2014',
                      ]

print('data type: ')
data_type = input() 

if data_type == 'LoRes' or data_type == 'HiRes':
    level = 12
    years = range(1979, 2019)   
    u_name = 'ue'
    v_name = 've'
    search_range = 2.0
elif data_type == 'SMP':
    level = 10
    years = range(2019, 2020)
    u_name = 'ua'
    v_name = 'va'
    search_range = 0.5

elif data_type == 'GPN':
    level = 22
    level = 12
    
    years = range(2022, 2023)   
    months = range(2, 3)       # !!!!!!!!!!!!!
    u_name = 'ua'
    v_name = 'va'
    search_range = 0.4

elif data_type == 'GLORYS':
    level = 8
    # level = 15
    
    years = range(2023, 2024)   
    # years = range(1979, 1980)   
    
    u_name = 'uo'
    v_name = 'vo'
    search_range = 0.25
    search_range = 0.15 # 2026-08-04
    
    sigmas = [0]
elif data_type == 'ALT':
    level = 0
    years = range(2023, 2024)   
    
    u_name = 'ugos'
    v_name = 'vgos'
    search_range = 0.25
    # search_range = 0.15 # 2026-08-04
    # search_range = 0.4 # 2026-08-09 for 0.25 data
    
    
    sigmas = [0]
elif data_type == 'ERA5':
    level = 500
    years = range(1979, 2025)   
    # years = range(1979, 1980)   
    
    u_name = 'u'
    v_name = 'v'
    search_range = 1.5 # # 2026-07-14 - translation speed of the fastest EXs <= 140 km per hour (Bernhardt&DeGaetano,2012;Lodiseetal.,2022).
    
    # #### 2026-07-09
    # search_range = 1.
    # #### 2026-07-13
    # search_range = 2.
    


if data_type == 'HiRes':
    search_range = 0.4

stitchnodes_cmd = "StitchNodes"  # Убедись, что доступен в PATH

for sigma in tqdm(sigmas):
    

    region = 'Arctic'
    # region = 'NA'
    region = 'BarKara'
    region = 'LV'

    circ = 'AC'
    # circ = 'C'
    
    
    name_crit = 'R2D'
    # name_crit = 'Q'
    # name_crit = 'lambda2'

    threshholded = True

    if threshholded:
        name_crit = f'{name_crit}_th'

    for in_file_name in tqdm(in_file_name_list):
    
        sigma_dir = f"{path_TE}/{data_type}/LV/{in_file_name}"
        
        # input_list_dir = f"{sigma_dir}/Input_list_for_tracking"
        # tracks_dir = f"{sigma_dir}/Tracks"
        # nodes_dir = f"{sigma_dir}/Nodes"
        
        input_list_dir = f"{sigma_dir}/{name_crit}_txt_files_list_for_tracking"
        nodes_dir = f"{sigma_dir}/{name_crit}_txt_files"
        tracks_dir = f"{sigma_dir}/Tracks_{name_crit}_txt_files_range_025_144h_72h"
    
        
        
        # Создаём нужные директории
        os.makedirs(input_list_dir, exist_ok=True)
        os.makedirs(tracks_dir, exist_ok=True)
        os.makedirs(nodes_dir, exist_ok=True)


        annual_file = f"{nodes_dir}/{circ}_{in_file_name}_{name_crit}_extr.txt"
        
        # Пути к файлам списков и треков
        output_list_file = f"{input_list_dir}/{circ}_{data_type}_{name_crit}_extr.txt"
        output_tracks_file = f"{tracks_dir}/{circ}_{data_type}_TC_tracks.txt"
        
        # Проверяем, существует ли годовой файл с данными
        if not os.path.exists(annual_file):
            print(f"Предупреждение: файл {annual_file} не найден, пропускаем {in_file_name}")
            continue

        # ========== ИЗМЕНЕНИЕ: Создаем список из одного файла ==========
        with open(output_list_file, 'w') as fout:
            fout.write(f"{annual_file}\n")
        
        # Проверяем, что список не пустой
        if os.path.getsize(output_list_file) == 0:
            print(f"Внимание: список файлов для {in_file_name} пуст, пропускаем")
            continue

        # Переход в директорию sigma_dir для выполнения команды
        original_dir = os.getcwd()
        try:
            os.chdir(sigma_dir)
            print(f"Перешли в директорию: {sigma_dir}")

            cmd = (
                f"{stitchnodes_cmd} "
                f"--in_list {output_list_file} "
                f"--out {output_tracks_file} "
                f"--in_fmt \"lon,lat,wind,r2d\" "
                f"--range {search_range} "
                # f"--mintime \"12h\" " # попробовать 18h - чтобы точно не ловить суточные
                # f"--maxgap \"3h\" "
                # f"--mintime \"18h\" " # 2026-07-14 - как в статье Han,Y.,& Ullrich,P.A.(2025).
                # f"--maxgap \"12h\" " # 2026-07-14 - как в статье Han,Y.,& Ullrich,P.A.(2025).

                f"--mintime \"144h\" " # 2026-08-03 - ocean
                f"--maxgap \"72h\" " # 2026-08-03 - ocean
            
                # f"--prioritize -r2d"  # Приоритет соединения с узлом, имеющим МАКСИМАЛЬНОЕ r2d
                # f"--min_endpoint_dist 0.5 " # это если фильтр по стационирующим КВС
            )

            print(f"Выполняется: {cmd}")
            exit_code = os.system(cmd)

            if exit_code == 0:
                print(f"Успешно: треки для sigma={sigma}, {in_file_name} сохранены в {output_tracks_file}")
            else:
                print(f"Ошибка при обработке sigma={sigma}, {in_file_name}")

        except Exception as e:
            print(f"Ошибка при выполнении команды для sigma={sigma}, {in_file_name}: {e}")
        finally:
            os.chdir(original_dir)  # Возвращаемся в исходную директорию