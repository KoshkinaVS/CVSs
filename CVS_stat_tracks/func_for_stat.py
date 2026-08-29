import os
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.patches import Ellipse
import math
import glob
from datetime import datetime
import matplotlib.dates as mdates

def is_point_in_ellipse(x, y, center_x, center_y, a, b, theta):
    """Check if a point is inside an ellipse."""
    x_t = x - center_x
    y_t = y - center_y

    x_r = x_t * np.cos(theta) + y_t * np.sin(theta)
    y_r = -x_t * np.sin(theta) + y_t * np.cos(theta)

    return (x_r/a)**2 + (y_r/b)**2 <= 1
    
def get_ellipse_mask(center_x, center_y, a, b, theta, shape):
    """Create a boolean mask for an ellipse within a given shape."""
    y_dim, x_dim = shape
    mask = np.zeros((y_dim, x_dim), dtype=bool)

    for i in range(y_dim):
        for j in range(x_dim):
            if is_point_in_ellipse(j, i, center_x, center_y, a, b, theta):
                mask[i, j] = True

    return mask

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



def plot_all_variables_min_max_range(track_df, output_dir, hires=''):
    """
    Create a composite plot showing the min-max range for temperature, wind speed, 
    and heat fluxes over time.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from matplotlib.dates import DateFormatter
    import pandas as pd
    from datetime import datetime, timedelta
    
    plt.figure(figsize=(16, 12))
    
    # Create 3 subplots: temperature, wind speed, and heat fluxes
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 15), sharex=True)
    
    # Check if 'time' column contains datetime objects, if not, convert it
    if 'time' in track_df.columns:
        if not isinstance(track_df['time'].iloc[0], pd.Timestamp):
            track_df['time'] = pd.to_datetime(track_df['time'])
        dates = track_df['time']
    else:
        # If no time column, use time_ind with a base date (adjust as needed)
        base_date = datetime(2010, 8, 15)  # Adjust this based on your data
        dates = [base_date + timedelta(hours=3*idx) for idx in track_df['time_ind']]
    
    # 1. Temperature plot
    ax1.plot(dates, track_df[f'{hires}T2_median'], 'b-', linewidth=2, label='Mean Temperature')
    ax1.fill_between(dates, track_df[f'{hires}T2_min'], track_df[f'{hires}T2_max'], 
                     color='blue', alpha=0.2, label='Min-Max Range')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('Temperature Evolution')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # 2. Wind Speed plot
    ax2.plot(dates, track_df[f'{hires}wind_speed_median'], 'g-', linewidth=2, label='Mean Wind Speed')
    ax2.fill_between(dates, track_df[f'{hires}wind_speed_median'], track_df['wind_speed_max'], 
                     color='green', alpha=0.2, label='Min-Max Range')
    ax2.set_ylabel('Wind Speed (m/s)')
    ax2.set_title('Wind Speed Evolution')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    # 3. Heat Fluxes plot
    ax3.plot(dates, track_df[f'{hires}hfx_median'], 'r-', linewidth=2, label='Sensible Heat Flux (HFX)')
    ax3.plot(dates, track_df[f'{hires}lh_median'], 'b-', linewidth=2, label='Latent Heat Flux (LH)')
    ax3.plot(dates, track_df[f'{hires}total_heat_flux_median'], 'purple', linewidth=2, label='Total Heat Flux')
    
    # Add min-max range for total heat flux
    total_flux_max = track_df[f'{hires}hfx_p90'] + track_df[f'{hires}lh_p90']
    total_flux_min = track_df[f'{hires}hfx_min'] + track_df[f'{hires}lh_min']
    ax3.fill_between(dates, total_flux_min, total_flux_max, 
                     color='purple', alpha=0.2, label='Total Heat Flux Range')
    
    ax3.set_ylabel('Heat Flux (W/m²)')
    ax3.set_title('Heat Flux Evolution')
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend()
    
    # Format x-axis dates
    date_format = DateFormatter('%Y-%m-%d %H:%M')
    ax3.xaxis.set_major_formatter(date_format)
    plt.gcf().autofmt_xdate()  # Rotate date labels for better readability
    
    plt.xlabel('Date/Time')
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/all_variables_composite.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_dual_resolution_comparison(track_df, output_dir):
    """Plot comparison between LoRes and HiRes results with time axis and grid."""
    plt.figure(figsize=(18, 12))
    
    # Преобразуем время в datetime, если оно еще не в этом формате
    if not pd.api.types.is_datetime64_any_dtype(track_df['time']):
        track_df['time'] = pd.to_datetime(track_df['time'])
    
    # Temperature comparison
    ax1 = plt.subplot(2, 2, 1)
    plt.plot(track_df['time'], track_df['T2_median'], 'b-', label='LoRes Median')
    plt.plot(track_df['time'], track_df['hires_T2_median'], 'r-', label='HiRes Median')
    plt.plot(track_df['time'], track_df['T2_p90'], 'b--', label='LoRes p90')
    plt.plot(track_df['time'], track_df['hires_T2_p90'], 'r--', label='HiRes p90')
    
    # Горизонтальная линия на 26°C
    plt.axhline(y=26, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    
    plt.title("Temperature (°C) Comparison")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    
    # Wind speed comparison
    ax2 = plt.subplot(2, 2, 2, sharex=ax1)
    plt.plot(track_df['time'], track_df['wind_speed_median'], 'b-', label='LoRes Median')
    plt.plot(track_df['time'], track_df['hires_wind_speed_median'], 'r-', label='HiRes Median')
    plt.plot(track_df['time'], track_df['wind_speed_p90'], 'b--', label='LoRes p90')
    plt.plot(track_df['time'], track_df['hires_wind_speed_p90'], 'r--', label='HiRes p90')
    
    plt.title("Wind Speed (m/s) Comparison")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    
    # SH Heat Flux comparison
    ax3 = plt.subplot(2, 2, 3, sharex=ax1)
    plt.plot(track_df['time'], track_df['hfx_median'], 'b-', label='LoRes Median')
    plt.plot(track_df['time'], track_df['hires_hfx_median'], 'r-', label='HiRes Median')
    plt.plot(track_df['time'], track_df['hfx_p90'], 'b--', label='LoRes p90')
    plt.plot(track_df['time'], track_df['hires_hfx_p90'], 'r--', label='HiRes p90')
    
    plt.title("SH Heat Flux (W/m²) Comparison")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    
    # LH Heat Flux comparison
    ax4 = plt.subplot(2, 2, 4, sharex=ax1)
    plt.plot(track_df['time'], track_df['lh_median'], 'b-', label='LoRes Median')
    plt.plot(track_df['time'], track_df['hires_lh_median'], 'r-', label='HiRes Median')
    plt.plot(track_df['time'], track_df['lh_p90'], 'b--', label='LoRes p90')
    plt.plot(track_df['time'], track_df['hires_lh_p90'], 'r--', label='HiRes p90')
    
    plt.title("LH Heat Flux (W/m²) Comparison")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    
    # Форматирование временной оси
    date_format = mdates.DateFormatter('%Y-%m-%d %H:%M')
    for ax in [ax1, ax2, ax3, ax4]:
        ax.xaxis.set_major_formatter(date_format)
    
    # Улучшение компоновки
    plt.tight_layout()
    
    # Сохранение графика
    plt.savefig(f"{output_dir}/dual_resolution_comparison_new.png", dpi=300, bbox_inches='tight')
    plt.close()