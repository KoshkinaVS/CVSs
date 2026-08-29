from func_for_add_params_2026 import *


# Инициализация путей и параметров
path_init = f'/storage/thalassa/users/vkoshkina'
folder = 'scripts/CS_processing/CVS_tracking/after_70RAE/'
sys.path.insert(2, f'{path_init}/{folder}')


tracking_type = 'tracking_local_2_phase'



json_file = f'{path_init}/{folder}/tracking_init.json'

dist_m, our_level, level, x_unit, y_unit, u_unit, v_unit, time_unit, level_unit, crit, th, min_samples, eps, CS_points_th = get_init_params(data_type, json_file, print_info=True)


path_data = f'{path_init}/data'  

circ = 'C'



pref_tracking = 'all_points_bound'
CVS_speed = 'adv_speed'


DBSCAN_name = f'DBSCAN_{eps:02d}-{min_samples:02d}-{CS_points_th:02d}_level_{level}_smoothing'

results_dir = f"{tracking_type}_{CVS_speed}_{pref_tracking}"

if data_type == 'HiRes':
    path_data_tracks = f'{path_data}/TC_tracks/my_tracking_results/{pref_tracking}/{results_dir}/{data_type}_tracks_sigma_{sigma}/'
elif data_type == 'ERA5':
    path_data_tracks = f"{path_data}/TC_tracks/{data_type}/my_tracking_results/{pref_tracking}/{results_dir}/{data_type}_tracks_sigma_{sigma}/"
else:
    path_data_tracks = f'{path_data}/{data_type}/{data_type}/{data_type}_tracks/{pref_tracking}/{results_dir}/{DBSCAN_name}' ### old folder




if data_type == 'LoRes':
    years = np.arange(1979, 2019)
    path_tracks_dir = f'{path_data_tracks}/tracks_{circ}'
    # new_circ_folder = f'tracks_{circ}_params_new'
    new_circ_folder = f'tracks_{circ}_params'
    path_data_tracks = f'{path_data}/TC_tracks/{data_type}/{data_type}_sigma_{sigma}/'
elif data_type == 'ERA5':
    years = np.arange(1979, 2025)
    path_tracks_dir = f'{path_data_tracks}/tracks_{circ}'
    new_circ_folder = f'tracks_{circ}_params'
    
    
    

months = np.arange(1,13,1)
days = np.arange(1,32,1)


if data_type != 'ERA5':
    params_config = [
        {'param': 'SST', 'level': 0, 'agg': 'median'},
        {'param': 'LH', 'level': 0, 'agg': 'max'},
        {'param': 'HFX', 'level': 0, 'agg': 'max'},
        {'param': 'slp', 'level': 0, 'agg': 'min'},
        {'param': 'wspd', 'level': 0, 'agg': 'max'},
        {'param': 'T2', 'level': 0, 'agg': 'median'},
        {'param': 'theta', 'level': 12, 'agg': 'median'},  # 900 hPa
        {'param': 'wa', 'level': 12, 'agg': 'median'},
        {'param': 'cape_2d', 'level': 0, 'agg': 'max'},
        {'param': 'cape_3d', 'level': 23, 'agg': 'max'}, # 500 hPa
        {'param': 'pvo', 'level': 23, 'agg': 'max'},
        {'param': 'helicity', 'level': 0, 'agg': 'max'},
        {'param': 'updraft_helicity', 'level': 0, 'agg': 'max'},
        {'param': 'pw', 'level': 0, 'agg': 'max'},
        {'param': 'slp', 'level': 0, 'agg': 'delta'},
        {'param': 'T2', 'level': 0, 'agg': 'delta'},
        
        # НОВЫЕ ПАРАМЕТРЫ ДЛЯ 850 hPa:
        {'param': 'theta', 'level': 14, 'agg': 'median'},  # 864 hPa
        {'param': 'temp', 'level': 14, 'agg': 'median'},   # температура на 850 hPa
    ]
    
    # Если индекс 8 не подходит, измените на правильный после проверки
    # Например: level=9 или level=850 (если используются прямые значения давления)

    km = 77 #### change for HiRes
else:
    params_config = [
        {'param': 'msl', 'level': 0, 'agg': 'min'},
        {'param': 'mlhf', 'level': 0, 'agg': 'max'},
        {'param': 'mshf', 'level': 0, 'agg': 'max'},
        {'param': 'sst', 'level': 0, 'agg': 'median'},
        {'param': 'wspd', 'level': 0, 'agg': 'max'},
        {'param': 't2m', 'level': 0, 'agg': 'median'},
        {'param': 't', 'level': 500, 'agg': 'median'},
        {'param': 'w', 'level': 500, 'agg': 'median'},
        {'param': 'tp', 'level': 0, 'agg': 'median'},
        
    ]

    km = 25


# Добавьте в начало скрипта
import warnings
warnings.filterwarnings('ignore')

# Настройки для ускорения
pd.options.mode.chained_assignment = None  # отключаем предупреждения

# Использование
if data_type == 'LoRes':
    # path_tracks_dir = '/storage/thalassa/users/vkoshkina/data/TC_tracks/LoRes/LoRes_sigma_2/tracks_C_params_new'
    path_tracks_dir = f'{path_data}/LoRes/LoRes/LoRes_tracks_sigma_2/all_points_bound/tracking_local_2_phase_no_speed_all_points_bound/tracks_AC'
    path_tracks_dir = f'{path_data}/LoRes/LoRes/LoRes_tracks_sigma_2/all_points_bound/tracking_global_only_no_speed_all_points_bound/tracks_AC'

    
    level_850 = 0
    
    params_config_new = [
        # {'param': 'theta', 'level': level_850, 'agg': 'median'},
        # {'param': 'temp', 'level': level_850, 'agg': 'median'},
        {'param': 'wspd', 'level': level_850, 'agg': 'max'},
        
    ]
    
    # Используем параллельную версию
    add_new_parameters_parallel(
        data_type='LoRes',
        path_tracks_dir=path_tracks_dir,
        params_config_new=params_config_new,
        km=77,
        output_suffix='_with_wspd',
        n_workers=30  # Увеличьте в зависимости от CPU
    )

#     # ИСПОЛЬЗОВАНИЕ:
# if data_type == 'LoRes':
#     # Указываем путь к существующим трекам
#     path_tracks_dir = '/storage/thalassa/users/vkoshkina/data/TC_tracks/LoRes/LoRes_sigma_2/tracks_C_params_new'
    
#     level_850 = 14  # или 9, или другой индекс после проверки
    
#     # Создаем конфигурацию только для новых параметров
#     params_config_new = [
#         {'param': 'theta', 'level': level_850, 'agg': 'median'},  # замените level_850 на правильный индекс
#         {'param': 'temp', 'level': level_850, 'agg': 'median'},
#     ]
    
#     # Добавляем новые параметры
#     add_new_parameters_to_tracks(
#         data_type='LoRes',
#         path_tracks_dir=path_tracks_dir,
#         params_config_new=params_config_new,
#         km=77,
#         output_suffix='_with_850hPa'  # новая папка: tracks_C_params_new_with_850hPa
#     )
    