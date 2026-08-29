# from func_for_add_params_nv import *

from func_for_add_params_upd import *
import multiprocessing as mp

data_type = 'LoRes'
sigma = 2
circ = 'C'

pref_tracking = 'update_2026-05-13'
CVS_speed = 'adv_speed'
tracking_type = 'tracking_local_2_phase'
results_dir = f"{tracking_type}_{CVS_speed}_{pref_tracking}"


path_dir_data = '/storage/thalassa/users/vkoshkina/data/LoRes/'
TRACKS_PATH = f"{path_dir_data}/{data_type}/{data_type}_tracks_sigma_{sigma}/{pref_tracking}/{results_dir}/tracks_{circ}/"

# TRACKS_PATH = "/storage/thalassa/users/vkoshkina/data/LoRes/LoRes/LoRes_tracks/all_points_bound/tracking_local_2_phase_adv_speed_all_points_bound/DBSCAN_02-04-10_level_12_smoothing_sigma_2/tracks_C/"

OUTPUT_PATH = f"{path_dir_data}/{data_type}/{data_type}_tracks/LoRes_tracks_params_mattiew_2026-05-13"



def main():
    print("!!!!!!!!!!!!!!!!!ЗАПУСК ОБРАБОТКИ!!!!!!!!!!!!!!!!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!ПАРАЛЛЕЛЬНАЯ ОБРАБОТКА ТРЕКОВ!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("   • Каждое ядро обрабатывает отдельный трек")
    print("   • Часы внутри трека обрабатываются последовательно\n")

    years = np.arange(1979,2019)
    months = np.arange(1,13)

    for year in tqdm(years):
        for month in months:
            TRACKS_PATH_monthly = f"{TRACKS_PATH}/{year}-{month:02d}/*.csv"
            WRF_PATH = f"/storage/NAAD/NAAD/LoRes/{year}/wrfout_d01_{year}-{month:02d}*" 
            OUTPUT_PATH_monthly = f"{OUTPUT_PATH}/{year}-{month:02d}"
            
            # Укажите количество процессов
            results = process_all_tracks(TRACKS_PATH_monthly, WRF_PATH, OUTPUT_PATH_monthly, data_type='LoRes', n_processes=12)
    
    print("\n" + "="*60)
    print("!!!!!!!!!!!!!!!ОБРАБОТКА ЗАВЕРШЕНА!!!!!!!!!!!!!!!!!!!!!")
    if results:
        print(f"Успешно обработано: {len(results)} треков")
    else:
        print("Нет обработанных данных")


if __name__ == "__main__":
    mp.set_start_method("fork")  # важно для Python 3.14 под Linux
    main()