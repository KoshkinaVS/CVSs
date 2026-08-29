from func_for_add_params_upd import *
import multiprocessing as mp

TRACKS_PATH = "/storage/thalassa/users/vkoshkina/data/LoRes/LoRes/EddyClicker_Egor_2010/*.csv"
WRF_PATH = "/storage/NAAD/NAAD/LoRes/2010/wrfout_d01_2010*" 
OUTPUT_PATH = "/storage/thalassa/users/vkoshkina/data/LoRes/LoRes/EddyClicker_tracks_Egor_2010_params"


def main():
    print("!!!!!!!!!!!!!!!!!!!!!!!!!ПАРАЛЛЕЛЬНАЯ ОБРАБОТКА ТРЕКОВ!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("   • Каждое ядро обрабатывает отдельный трек")
    print("   • Часы внутри трека обрабатываются последовательно\n")

    # сначала попробуйте меньше процессов, чтобы убедиться, что всё живое
    results = process_all_tracks(TRACKS_PATH, WRF_PATH, OUTPUT_PATH, n_processes=12)

    print("\n" + "="*60)
    print("!!!!!!!!!!!!!!!ОБРАБОТКА ЗАВЕРШЕНА!!!!!!!!!!!!!!!!!!!!!")
    if results:
        print(f"Успешно обработано: {len(results)} треков")
    else:
        print("Нет обработанных данных")


if __name__ == "__main__":
    mp.set_start_method("fork")  # важно для Python 3.14 под Linux
    main()