import os
from tqdm import tqdm

path_init = '/storage/thalassa/users/vkoshkina/data/TempestExtremes'

months = range(1, 13)       # от 1 до 12

sigmas = [
#     4,
    2,
#     0,
]

print('data type: ')
data_type = input() 

if data_type == 'LoRes' or data_type == 'HiRes':
    level = 12
    years = range(1979, 2019)   
    u_name = 'ue'
    v_name = 've'
elif data_type == 'ERA5':
    level = 500
    years = range(1979, 2025)   
    u_name = 'u'
    v_name = 'v'

nodefileeditor_cmd = "NodeFileEditor"  # Убедись, что доступен в PATH


for sigma in tqdm(sigmas):
    sigma_dir = f"{path_init}/{data_type}/R2D_{data_type}_level_{level}_sigma_{sigma}"
    os.chdir(sigma_dir)

    # input_list_dir = f"{sigma_dir}/Input_list_mergedist_05"
    # nodes_dir = f"{sigma_dir}/Nodes_mergedist_05"
    # tracks_dir = f"{sigma_dir}/Tracks_05"
    # output_dir = f"{sigma_dir}/Tracks_with_rads_05"  # Директория для выходных файлов

    input_list_dir = f"{sigma_dir}/Input_list_timefilter_1h"
    nodes_dir = f"{sigma_dir}/Nodes_mergedist_timefilter_1h"
    tracks_dir = f"{sigma_dir}/Tracks_timefilter_1h"
    output_dir = f"{sigma_dir}/Tracks_with_rads_timefilter_1h"  # Директория для выходных файлов
    
    os.makedirs(output_dir, exist_ok=True)

    for year in tqdm(years):
        # Пути к файлам
        input_list_file = f"{input_list_dir}/{data_type}_input_{year}.txt"
        input_tracks_file = f"{tracks_dir}/{data_type}_TC_tracks_{year}.txt"
        output_nodefile = f"{output_dir}/{data_type}_TC_rad_crit_{year}.txt"

        try:
            # # Команда NodeFileEditor с относительными путями
            # cmd = (
            #     f"{nodefileeditor_cmd} "
            #     f"--in_data_list {input_list_file} "
            #     f"--in_nodefile {input_tracks_file} "
            #     f"--in_nodefile_type \"SN\" "
            #     f"--in_fmt \"lon,lat,wind,r2d\" "
            
            #     f"--out_nodefile {output_nodefile} "
            #     f"--out_fmt \"lon,lat,wind,r2d,rsize\" "
            
            #     f"--timefilter \"{year}-..-.. .*\" "
            #     f"--regional "
            #     f"--calculate \"rprof=radial_profile(R2D,79,0.125);rsize=lastwhere(rprof,>,0)\" "
            # )


            # Вычисление интегральных свойств вдоль треков
            cmd = (
                f"{nodefileeditor_cmd} "
                f"--in_nodefile {input_tracks_file} "
                f"--in_nodefile_type \"SN\" "
                f"--in_data_list {input_list_file} "
                f"--in_data \"sst,mslp,sh,lh,t2\" "
                f"--regional "
                f"--calculate \"sh=;lh= \" "
                f"--out_nodefile {output_nodefile} "
                f"--out_fmt \"lon,lat,wind,r2d,rad,sh,lh\" "
            )





            exit_code = os.system(cmd)

        except Exception as e:
            print(f"Ошибка при выполнении команды для sigma={sigma}, год {year}: {e}")