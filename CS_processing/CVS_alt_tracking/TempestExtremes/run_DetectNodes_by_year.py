import os
from tqdm import tqdm
import calendar

path_init = '/storage/thalassa/users/vkoshkina/data/TempestExtremes'


months = range(1, 13)       # от 1 до 12

sigmas = [
#     4,
    2,
#     0,
]


# timefilter = 3

print('data type: ')
data_type = input() 


region = 'Arctic'
region = 'NA'

if data_type == 'LoRes' or data_type == 'HiRes':
    level = 12
    years = range(1979, 2019)   
    u_name = 'ue'
    v_name = 've'
    minlat = 2.0
    maxlat = 85.0
    minlon = -100.0
    maxlon = 17.0
    mergedist = 0.5
elif data_type == 'SMP':
    level = 10
    years = range(2019, 2020)   
    # months = range(1, 7)       # !!!!!!!!!!!!!
    
    u_name = 'ua'
    v_name = 'va'
    minlat = 56.0
    maxlat = 90.0
    minlon = -180.0
    maxlon = 180.0

    minlat = 56.0
    maxlat = 89.0
    minlon = 0.0
    maxlon = 120.0

    mergedist = 0.05
    mergedist = 0.3  # Увеличено для мезоциклонов (~18 км)

    # timefilter = 1

elif data_type == 'GPN':
    level = 22
    level = 12
    
    years = range(2022, 2023)   
    months = range(2, 3)       # !!!!!!!!!!!!!
    
    u_name = 'ua'
    v_name = 'va'
    
    minlat = 62.3
    maxlat = 78.7
    minlon = 22.32
    maxlon = 76.0

    mergedist = 0.05

elif data_type == 'ERA5':

    if region == 'Arctic':
        # Арктика
        level = 850
        region_name = f'Arctic_{level}hPa'

        minlat = 65.0
        maxlat = 85.0
        minlon = -30.0
        maxlon = 80.0
        mergedist = 0.75 # 0.5
        
    else:
        # Атлантика
        level = 500
        level = 850
        
        region_name = f'NA_for_TC_{level}hPa'
    

        minlat = 0.0
        maxlat = 71.0
        minlon = -110.0
        maxlon = 15.0
        mergedist = 0.75 # в исходном моем был ровно шаг по сетке, 0.25?
        mergedist = 0.25 # 
        

    years = range(1979, 2026)   
    u_name = 'u'
    v_name = 'v'


if data_type == 'HiRes':
    mergedist = 0.1

detectnodes_cmd = "DetectNodes"  # Убедитесь, что команда доступна в PATH

for sigma in tqdm(sigmas):
    sigma_dir = f"{path_init}/{data_type}/R2D_{data_type}_level_{level}_sigma_{sigma}"
    
    #### 2026-07-08
    sigma_dir = f"{path_init}/{data_type}/R2D_{data_type}_{region_name}_sigma_{sigma}"
    
    
    
    # Поддиректории
    input_dir = f"{sigma_dir}/Input"
    input_list_dir = f"{sigma_dir}/Input_mergedist_025"
    nodes_dir = f"{sigma_dir}/Nodes_mergedist_025"
    
#     # Поддиректории
#     input_dir = f"{sigma_dir}/Input_yearly"
#     input_list_dir = f"{sigma_dir}/Input_yearly"
#     nodes_dir = f"{sigma_dir}/Nodes_yearly"

    # # Поддиректории
    # input_dir = f"{sigma_dir}/Input"
    # input_list_dir = f"{sigma_dir}/Input_list_mergedist_05"
    # nodes_dir = f"{sigma_dir}/Nodes_mergedist_05"
    
    # Создаём нужные директории
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(input_list_dir, exist_ok=True)
    os.makedirs(nodes_dir, exist_ok=True)

    for year in tqdm(years):
        # Базовые пути к файлам
        base_path_in = f"{input_dir}/sigma_{sigma}_R2D_{data_type}_level_{level}"
        base_path_out = f"{nodes_dir}/{data_type}_R2D_extr"

        # Файлы списков
        input_list_file = f"{input_list_dir}/{data_type}_input_{year}.txt"
        output_list_file = f"{input_list_dir}/{data_type}_R2D_extr_{year}.txt"

        # Создаем список входных файлов
        with open(input_list_file, 'w') as fin, open(output_list_file, 'w') as fout:
            
#             # Входные и выходные файлы
#             input_file = f"{base_path_in}_{year}.nc"
#             output_file = f"{base_path_out}_{year}.txt"
            
#             # Записываем в списки
#             fin.write(f"{input_file}\n")
#             fout.write(f"{output_file}\n")
            
#             # Создаём пустой выходной файл
#             open(output_file, 'a').close()
                    
            for month in months:

                if data_type == 'SMP' or data_type == 'GPN':
                    num_days = calendar.monthrange(year, month)[1]
        
                    for day in range(1, num_days + 1):
                        # Входные и выходные файлы
                        input_file = f"{base_path_in}_{year}-{month:02d}-{day:02d}.nc"
                        output_file = f"{base_path_out}_{year}-{month:02d}-{day:02d}.txt"
                        
                        # Записываем в списки
                        fin.write(f"{input_file}\n")
                        fout.write(f"{output_file}\n")
                        
                        # Создаём пустой выходной файл
                        open(output_file, 'a').close()
                else:
                    # Входные и выходные файлы
                    input_file = f"{base_path_in}_{year}-{month:02d}.nc"
                    output_file = f"{base_path_out}_{year}-{month:02d}.txt"
                    
                    # Записываем в списки
                    fin.write(f"{input_file}\n")
                    fout.write(f"{output_file}\n")
                    
                    # Создаём пустой выходной файл
                    open(output_file, 'a').close()
                    
                
        # Переход в директорию sigma_dir для выполнения команды
        original_dir = os.getcwd()
        try:
            os.chdir(sigma_dir)
            print(f"Перешли в директорию: {sigma_dir}")

            # Команда DetectNodes с параметрами из первого скрипта
            cmd = (
                f"{detectnodes_cmd} "
                f"--in_data_list {input_list_file} "
                # f"--timefilter \"{timefilter}hr\" "
                f"--timefilter \"{year}-..-.. .*\" "
                f"--out_file_list {output_list_file} "
                f"--searchbymax R2D "
                f"--mergedist {mergedist} "
                # f"--mergeequal "
                f"--regional "
                f"--minlat {minlat} "
                f"--maxlat {maxlat} "
                f"--minlon {minlon} "
                f"--maxlon {maxlon} "
                f"--thresholdcmd \"R2D,>,0,0\" "
                f"--outputcmd \"_VECMAG({u_name},{v_name}),max,2;R2D,max,0\""
            )

            print(f"Выполняется: {cmd}")
            exit_code = os.system(cmd)

            if exit_code == 0:
                print(f"Успешно: узлы для sigma={sigma}, {year} сохранены в {nodes_dir}")
            else:
                print(f"Ошибка при обработке sigma={sigma}, год {year}")

        except Exception as e:
            print(f"Ошибка при выполнении команды для sigma={sigma}, год {year}: {e}")
        finally:
            os.chdir(original_dir)  # Возвращаемся в исходную директорию