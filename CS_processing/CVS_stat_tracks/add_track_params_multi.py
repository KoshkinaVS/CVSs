# import os
# import pandas as pd
# import xarray as xr
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from matplotlib.patches import Ellipse
# import math
# import glob
# from datetime import datetime
# import matplotlib.dates as mdates

from func_for_stat import *

# Constants for parameter processing
PARAMETERS = {
    'T2': {
        'name': 'Temperature',
        'unit': '°C',
        'convert_kelvin': True,
        'plot_range': None,
        'plot_func': None
    },
    'wind_speed': {
        'name': 'Wind Speed',
        'unit': 'm/s',
        'convert_kelvin': False,
        'plot_range': None,
        'plot_func': None
    },
    'hfx': {
        'name': 'Sensible Heat Flux',
        'unit': 'W/m²',
        'convert_kelvin': False,
        'plot_range': None,
        'plot_func': None
    },
    'lh': {
        'name': 'Latent Heat Flux',
        'unit': 'W/m²',
        'convert_kelvin': False,
        'plot_range': None,
        'plot_func': None
    },
    'PSFC': {
        'name': 'Surface Pressure',
        'unit': 'hPa',
        'convert_kelvin': False,
        'plot_range': None,
        'plot_func': None
    },

    'msl': {
        'name': 'Mean Sea Level',
        'unit': 'hPa',
        'convert_kelvin': False,
        'plot_range': None,
        'plot_func': None
    },
    
    'total_heat_flux': {
        'name': 'Total Heat Flux',
        'unit': 'W/m²',
        'convert_kelvin': False,
        'plot_range': None,
        'plot_func': lambda row: row['hfx_median'] + row['lh_median']
    }
}

STAT_TYPES = ['mean', 'median', 'max', 'min', 'std', 'p90', 'anomaly']

def load_lores_coords(lores_file):
    """Загружает координаты (XLONG, XLAT) из файла LoRes"""
    ds = xr.open_dataset(lores_file)
    return ds['XLONG'], ds['XLAT']

def load_hires_data(hires_dir, date_str):
    """Load HiRes data for a specific date."""
    pattern = os.path.join(hires_dir, f"wrfout_d01_{date_str[:10]}*")
    files = glob.glob(pattern)
    if files:
        return xr.open_dataset(files[0])
    return None

def find_hires_indices(lon, lat, hires_ds):
    """Находит индексы (y, x) в HiRes данных, ближайшие к заданным координатам."""
    hires_lon = hires_ds.XLONG.values[0]
    hires_lat = hires_ds.XLAT.values[0]
    
    distances = np.sqrt((hires_lon - lon)**2 + (hires_lat - lat)**2)
    y_idx, x_idx = np.unravel_index(np.argmin(distances), distances.shape)
    return y_idx, x_idx

def compute_statistics(data_array, mask, convert_from_kelvin=False):
    """Compute extended statistics for a data array within a mask."""
    if data_array is None:
        return {stat: np.nan for stat in STAT_TYPES} | {'n_points': 0}

    values = data_array.values if hasattr(data_array, 'values') else data_array
    masked_data = values[mask]

    if len(masked_data) == 0 or np.all(np.isnan(masked_data)):
        return {stat: np.nan for stat in STAT_TYPES} | {'n_points': 0}
    
    if convert_from_kelvin:
        masked_data = masked_data - 273.15

    stats = {
        'mean': np.nanmean(masked_data),
        'median': np.nanmedian(masked_data),
        'max': np.nanmax(masked_data),
        'min': np.nanmin(masked_data),
        'std': np.nanstd(masked_data),
        'p90': np.nanpercentile(masked_data, 90),
        'anomaly': np.nanmax(masked_data) - np.nanmean(masked_data),
        'n_points': np.sum(~np.isnan(masked_data))
    }
    
    return stats

def calculate_parameter_stats(hires_data, time_idx, mask, param_name):
    """Calculate statistics for a specific parameter."""
    param_config = PARAMETERS[param_name]
    
    if param_name == 'total_heat_flux':
        # Special handling for derived parameter
        hfx_stats = calculate_parameter_stats(hires_data, time_idx, mask, 'hfx')
        lh_stats = calculate_parameter_stats(hires_data, time_idx, mask, 'lh')
        
        return {
            'mean': hfx_stats['mean'] + lh_stats['mean'],
            'median': hfx_stats['median'] + lh_stats['median'],
            'max': hfx_stats['max'] + lh_stats['max'],
            'min': hfx_stats['min'] + lh_stats['min'],
            'std': np.sqrt(hfx_stats['std']**2 + lh_stats['std']**2),
            'p90': hfx_stats['p90'] + lh_stats['p90'],
            'anomaly': np.nan,
            'n_points': min(hfx_stats['n_points'], lh_stats['n_points'])
        }

    if param_name == 'msl':
        data_var = {
        'msl': hires_data['msl'],
                    }.get(param_name)
        data_slice = data_var.isel(time=time_idx)
        return compute_statistics(data_slice, mask, param_config['convert_kelvin'])
        
    
    data_var = {
        'T2': hires_data['T2'],
        'wind_speed': np.sqrt(hires_data['U10']**2 + hires_data['V10']**2),
        'hfx': hires_data['HFX'],
        'lh': hires_data['LH'],
        'PSFC': hires_data['PSFC']/100,
        
    }.get(param_name)
    
    if data_var is None:
        return {stat: np.nan for stat in STAT_TYPES} | {'n_points': 0}
    
    data_slice = data_var.isel(Time=time_idx)
    return compute_statistics(data_slice, mask, param_config['convert_kelvin'])

def process_track_with_hires_analysis(temp_file, track_file, hires_dir, output_dir, circ='C', visualize=True):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing track with HiRes analysis")
    track_df = pd.read_csv(track_file)
    print(f"Track loaded with {len(track_df)} points")

    temp_data = xr.open_dataset(temp_file)
    lores_xlong, lores_xlat = load_lores_coords(temp_file)
    
    # Initialize results dictionary
    results = {}
    for param in PARAMETERS:
        for stat in STAT_TYPES:
            results[f'hires_{param}_{stat}'] = []
        results[f'hires_{param}_n_points'] = []
    
    print("Processing elliptical regions for each track point...")
    for i in tqdm(range(len(track_df))):
        row = track_df.iloc[i]
        x_idx = int(row['pxc_ind'])
        y_idx = int(row['pyc_ind'])
        
        lon = lores_xlong.isel(west_east=x_idx, south_north=y_idx).values
        lat = lores_xlat.isel(west_east=x_idx, south_north=y_idx).values
        track_df.at[i, 'longitude'] = lon
        track_df.at[i, 'latitude'] = lat

        date_str = str(row['time']) if 'time' in row else "2010-08-15"
        hires_data = load_hires_data(hires_dir, date_str)
        
        if hires_data is None:
            print(f"No HiRes data found for {date_str}")
            for param in PARAMETERS:
                for stat in STAT_TYPES:
                    results[f'hires_{param}_{stat}'].append(np.nan)
                results[f'hires_{param}_n_points'].append(0)
            continue

        track_time = np.datetime64(pd.to_datetime(date_str))
        time_idx = np.argmin(np.abs(hires_data['XTIME'].values - track_time))
        center_y, center_x = find_hires_indices(float(lon), float(lat), hires_data)
        
        a = float(row.semi_major_axis)
        b = float(row.semi_minor_axis)
        theta = float(row.orientation_rad)
        
        hires_scale_factor = 77/14  # LoRes 77km / HiRes 14km
        a_hires = a * hires_scale_factor
        b_hires = b * hires_scale_factor
        
        south_north_dim = 'south_north' if 'south_north' in hires_data.sizes else 'y'
        west_east_dim = 'west_east' if 'west_east' in hires_data.sizes else 'x'
        mask_shape = (hires_data.sizes[south_north_dim], hires_data.sizes[west_east_dim])
        hires_mask = get_ellipse_mask(center_x, center_y, a_hires, b_hires, theta, mask_shape)
        
        try:
            for param in PARAMETERS:  
                if param == 'msl':
                    temp_file = "/storage/OPENDATA/NAAD/HiRes/Surface/msl/NAAD14km_msl_2010.nc"
                    temp_data = xr.open_dataset(temp_file)
                    time_idx = np.argmin(np.abs(temp_data['time'].values - track_time))
                    # print(f'{time_idx} for {track_time}')
                    stats = calculate_parameter_stats(temp_data, time_idx, hires_mask, 'msl')
                    print(f'msl stats computed')
                    
                    for stat in STAT_TYPES:
                        results[f'hires_msl_{stat}'].append(stats[stat])
                    results[f'hires_msl_n_points'].append(stats['n_points'])
                else:
                    stats = calculate_parameter_stats(hires_data, time_idx, hires_mask, param)
                    for stat in STAT_TYPES:
                        results[f'hires_{param}_{stat}'].append(stats[stat])
                    results[f'hires_{param}_n_points'].append(stats['n_points'])


                                    
        except Exception as e:
            print(f"Error processing HiRes point {i}: {e}")
            for param in PARAMETERS:
                for stat in STAT_TYPES:
                    results[f'hires_{param}_{stat}'].append(np.nan)
                results[f'hires_{param}_n_points'].append(0)
    
    # Add results to dataframe
    for key, values in results.items():
        track_df[key] = values
        
    # Save and visualize results
    output_file = f"{output_dir}/track_hires_analysis.csv"
    track_df.to_csv(output_file, index=False) 
     
    # Print summary statistics
    for param, config in PARAMETERS.items():
        print(f"\n{config['name']} statistics summary ({config['unit']}):")
        for stat in ['mean', 'median', 'max', 'min', 'p90']:
            col_name = f'hires_{param}_{stat}'
            if col_name in track_df.columns:
                print(f"  {stat.capitalize()}: {np.nanmean(track_df[col_name]):.2f}")
    
    # Create plots
    create_composite_plots(track_df, output_dir, hires_prefix='hires_')
    plot_dual_resolution_comparison(track_df, output_dir)
    
    return track_df

def create_composite_plots(track_df, output_dir, hires_prefix=''):
    """Create all composite plots using unified parameter definitions."""
    # Temperature plot
    plot_parameter_over_time(track_df, output_dir, 'T2', hires_prefix)
    
    # Wind speed plot
    plot_parameter_over_time(track_df, output_dir, 'wind_speed', hires_prefix)

    plot_parameter_over_time(track_df, output_dir, 'lh', hires_prefix)
    
    plot_parameter_over_time(track_df, output_dir, 'hfx', hires_prefix)

    plot_parameter_over_time(track_df, output_dir, 'PSFC', hires_prefix)
    
    
    
    
    # # Heat fluxes plot
    # plot_heat_fluxes_over_time(track_df, output_dir, hires_prefix)
    
    # Combined variables plot
    plot_all_variables_min_max_range(track_df, output_dir, hires_prefix)

def plot_parameter_over_time(track_df, output_dir, param_name, hires_prefix=''):
    """Generic function to plot any parameter over time."""
    param_config = PARAMETERS[param_name]
    
    plt.figure(figsize=(14, 8))
    
    if 'time' in track_df.columns:
        if not isinstance(track_df['time'].iloc[0], pd.Timestamp):
            track_df['time'] = pd.to_datetime(track_df['time'])
        dates = track_df['time']
    else:
        base_date = datetime(2010, 8, 15)
        dates = [base_date + timedelta(hours=3*idx) for idx in track_df['time_ind']]
    
    mean_col = f"{hires_prefix}{param_name}_mean"
    median_col = f"{hires_prefix}{param_name}_median"
    std_col = f"{hires_prefix}{param_name}_std"
    p90_col = f"{hires_prefix}{param_name}_p90"
    
    if median_col in track_df.columns:
        plt.plot(dates, track_df[median_col], 'b-', linewidth=2, 
                label=f'Median {param_config["name"]}')
    
    if mean_col in track_df.columns and std_col in track_df.columns:
        plt.fill_between(dates, 
                        track_df[mean_col] - track_df[std_col], 
                        track_df[mean_col] + track_df[std_col], 
                        color='blue', alpha=0.2, label='Mean ± Std Dev')
    
    if p90_col in track_df.columns:
        plt.plot(dates, track_df[p90_col], 'r-', linewidth=2, 
                label=f'90th Percentile {param_config["name"]}')
    
    plt.xlabel('Date/Time')
    plt.ylabel(f'{param_config["name"]} ({param_config["unit"]})')
    plt.title(f'{param_config["name"]} Evolution')
    plt.legend()
    plt.grid(True)
    
    date_format = mdates.DateFormatter('%Y-%m-%d %H:%M')
    plt.gca().xaxis.set_major_formatter(date_format)
    plt.gcf().autofmt_xdate()
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{param_name}_over_time_composite.png", dpi=150, bbox_inches='tight')
    plt.close()

# [Keep all the other existing functions like plot_heat_fluxes_over_time, 
#  plot_all_variables_min_max_range, plot_dual_resolution_comparison, etc.]
# They can be updated similarly to use the PARAMETERS dictionary

def main():
    """Main function to run the analysis."""
    data_dir = "./TC_tracks"
    temp_file = "/storage/OPENDATA/NAAD/LoRes/Surface/t2/NAAD77km_t2_2010.nc"
    
    track_file = os.path.join(data_dir, "EARL_500hpa.csv")
    TC_name = 'DANIELLE'
    TC_name = 'IGOR'
    
    track_file = os.path.join(data_dir, f"{TC_name}_extended_analysis_500hPa.csv")
    
    hires_dir = "/storage/NAAD/NAAD/HiRes/2010"
    output_dir = f"output/hires_analysis_{TC_name}"

    try:
        processed_df = process_track_with_hires_analysis(
            temp_file=temp_file,
            track_file=track_file,
            hires_dir=hires_dir,
            output_dir=output_dir,
            circ='C',
            visualize=True
        )
        print("HiRes analysis completed successfully!")
    except Exception as e:
        print(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()