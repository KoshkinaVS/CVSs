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

def load_lores_coords(lores_file):
    """Загружает координаты (XLONG, XLAT) из файла LoRes"""
    ds = xr.open_dataset(lores_file)
    return ds['XLONG'], ds['XLAT']

def load_hires_data(hires_dir, date_str):
    """Load HiRes data for a specific date."""
    pattern = os.path.join(hires_dir, f"wrfout_d01_{date_str[:10]}*")  # Для домена d02 (HiRes)
    files = glob.glob(pattern)
    if files:
        return xr.open_dataset(files[0])
    return None

def find_hires_indices(lon, lat, hires_ds):
    """Находит индексы (y, x) в HiRes данных, ближайшие к заданным координатам."""
    hires_lon = hires_ds.XLONG.values
    hires_lat = hires_ds.XLAT.values
    
    # Вычисляем расстояния до всех точек HiRes сетки
    distances = np.sqrt((hires_lon - lon)**2 + (hires_lat - lat)**2)
    y_idx, x_idx = np.unravel_index(np.argmin(distances), distances.shape)
    return y_idx, x_idx

def plot_comparison(track_df, output_dir):
    plt.figure(figsize=(12, 6))
    
    # Температура
    plt.subplot(1, 2, 1)
    plt.plot(track_df['T2_mean'], label='LoRes (77km)')
    plt.plot(track_df['hires_T2_mean'], label='HiRes (14km)')
    plt.title("Сравнение средней температуры")
    plt.legend()
    
    # Скорость ветра
    plt.subplot(1, 2, 2)
    plt.plot(track_df['wind_speed_mean'], label='LoRes (77km)')
    plt.plot(track_df['hires_wind_speed_mean'], label='HiRes (14km)')
    plt.title("Сравнение скорости ветра")
    plt.legend()
    
    plt.savefig(f"{output_dir}/lores_vs_hires_comparison.png")


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
    
def calculate_ellipse_parameters_improved(track_df):
    """Calculate ellipse parameters for each point in the track."""
    result_df = track_df.copy()

    for i, row in track_df.iterrows():
        x0 = row['pxc_ind']  
        y0 = row['pyc_ind'] 
        x1 = row['px1_ind']  
        y1 = row['py1_ind']  
        x2 = row['px2_ind']  
        y2 = row['py2_ind']  
        x3 = row['px3_ind']  
        y3 = row['py3_ind'] 

        p1 = np.array([x1 - x0, y1 - y0])
        p2 = np.array([x2 - x0, y2 - y0])
        p3 = np.array([x3 - x0, y3 - y0])

        len_p1 = np.linalg.norm(p1)
        len_p2 = np.linalg.norm(p2)
        a = max(len_p1, len_p2)
        major_axis = np.array([x2 - x1, y2 - y1])
        major_axis_len = np.linalg.norm(major_axis)

        if major_axis_len > 0:
            major_axis = major_axis / major_axis_len

            orientation = np.arctan2(major_axis[1], major_axis[0])

            perp_axis = np.array([-major_axis[1], major_axis[0]])

            p3_proj_perp = np.abs(np.dot(p3, perp_axis))
            b = p3_proj_perp
        else:
            b = np.linalg.norm(p3)
            orientation = np.arctan2(p3[1], p3[0]) + np.pi/2

        if b > a:
            a, b = b, a
            orientation += np.pi/2

        a = max(a, 3)
        b = max(b, 2)

        eccentricity = np.sqrt(1 - (b/a)**2) if a > 0 else 0

        result_df.at[i, 'semi_major_axis'] = a
        result_df.at[i, 'semi_minor_axis'] = b
        result_df.at[i, 'eccentricity'] = eccentricity
        result_df.at[i, 'orientation_rad'] = orientation
        result_df.at[i, 'orientation_deg'] = np.degrees(orientation)

    return result_df
    
def get_ellipse_perimeter_improved(center_x, center_y, a, b, theta, npoints=100):
    """Get points along the perimeter of an ellipse."""
    t = np.linspace(0, 2*np.pi, npoints)
    x = center_x + a * np.cos(t) * np.cos(theta) - b * np.sin(t) * np.sin(theta)
    y = center_y + a * np.cos(t) * np.sin(theta) + b * np.sin(t) * np.cos(theta)

    return x, y

def visualize_ellipse_improved(temp_data, track_df, time_idx, output_dir):
    """Visualize the temperature field with ellipse."""
    if 'south_north' in temp_data.dims:
        temp = temp_data.t2.isel(time=time_idx).values
    else:
        temp = temp_data.t2.isel(time=time_idx).values

    # Convert K to °C for visualization
    temp = temp - 273.15

    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(temp, cmap='coolwarm', origin='lower')
    plt.colorbar(im, ax=ax, label='Temperature (°C)')

    track_point = track_df[track_df['time_ind'] == time_idx]

    if len(track_point) > 0:
        point = track_point.iloc[0]

        ax.plot(point['pxc_ind'], point['pyc_ind'], 'ko', markersize=8)

        x_ellipse, y_ellipse = get_ellipse_perimeter_improved(
            point['pxc_ind'], 
            point['pyc_ind'],
            point['semi_major_axis'],
            point['semi_minor_axis'],
            point['orientation_rad']
        )
        ax.plot(x_ellipse, y_ellipse, 'k-', linewidth=2)

        ax.plot(point['px1_ind'], point['py1_ind'], 'ro', markersize=6)
        ax.plot(point['px2_ind'], point['py2_ind'], 'go', markersize=6)
        ax.plot(point['px3_ind'], point['py3_ind'], 'bo', markersize=6)
        major_endpoint_x = point['pxc_ind'] + point['semi_major_axis'] * np.cos(point['orientation_rad'])
        major_endpoint_y = point['pyc_ind'] + point['semi_major_axis'] * np.sin(point['orientation_rad'])
        ax.plot([point['pxc_ind'], major_endpoint_x], [point['pyc_ind'], major_endpoint_y], 'r--', linewidth=1)

        minor_orientation = point['orientation_rad'] + np.pi/2
        minor_endpoint_x = point['pxc_ind'] + point['semi_minor_axis'] * np.cos(minor_orientation)
        minor_endpoint_y = point['pyc_ind'] + point['semi_minor_axis'] * np.sin(minor_orientation)
        ax.plot([point['pxc_ind'], minor_endpoint_x], [point['pyc_ind'], minor_endpoint_y], 'b--', linewidth=1)

        time_str = str(point['time']) if 'time' in point else f"Time index {time_idx}"
        ax.set_title(f"Temperature field with improved ellipse at {time_str}")
    else:
        ax.set_title(f"Temperature field at time index {time_idx}")

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/improved_ellipsetime{time_idx}.png", dpi=150, bbox_inches='tight')
    plt.close()

def plot_temperature_over_time(track_df, output_dir):
    """
    Plot mean temperature over the lifetime of the track with standard deviation bands.
    Uses real dates on x-axis and shows std deviation bands around the mean temperature.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from matplotlib.dates import DateFormatter
    import pandas as pd
    
    plt.figure(figsize=(14, 8))
    
    # Check if 'time' column contains datetime objects, if not, convert it
    if 'time' in track_df.columns:
        if not isinstance(track_df['time'].iloc[0], pd.Timestamp):
            track_df['time'] = pd.to_datetime(track_df['time'])
        dates = track_df['time']
    else:
        # If no time column, use time_ind with a base date (adjust as needed)
        import pandas as pd
        from datetime import datetime, timedelta
        base_date = datetime(2010, 8, 15)  # Adjust this based on your data
        dates = [base_date + timedelta(hours=3*idx) for idx in track_df['time_ind']]
    
    # Create a line for mean temperature with standard deviation bands
    mean_temp = track_df['T2_mean']
    std_temp = track_df['T2_std']
    
    # Plot mean temperature line
    plt.plot(dates, mean_temp, 'b-', linewidth=2, label='Mean Temperature (°C)')
    
    # Plot standard deviation bands (композит)
    plt.fill_between(dates, 
                     mean_temp - std_temp, 
                     mean_temp + std_temp, 
                     color='blue', 
                     alpha=0.2, 
                     label='Mean ± Std Dev')
    
    # Create a line for maximum temperature
    plt.plot(dates, track_df['T2_max'], 'r-', linewidth=2, label='Maximum Temperature (°C)')
    
    # Create a line for minimum temperature
    plt.plot(dates, track_df['T2_min'], 'g-', linewidth=2, label='Minimum Temperature (°C)')
    
    plt.xlabel('Date/Time')
    plt.ylabel('Temperature (°C)')
    plt.title('Temperature Evolution with Standard Deviation')
    plt.legend()
    plt.grid(True)
    
    # Format x-axis dates
    date_format = DateFormatter('%Y-%m-%d %H:%M')
    plt.gca().xaxis.set_major_formatter(date_format)
    plt.gcf().autofmt_xdate()  # Rotate date labels for better readability
    
    # Ensure y-axis has some margin
    y_min = min(min(mean_temp - std_temp), min(track_df['T2_min']))
    y_max = max(max(mean_temp + std_temp), max(track_df['T2_max']))
    plt.ylim(y_min - 1, y_max + 1)
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/temperature_over_time_composite.png", dpi=150, bbox_inches='tight')
    plt.close()



def plot_wind_speed_over_time(track_df, output_dir):
    """
    Plot mean wind speed over the lifetime of the track with standard deviation bands.
    Uses real dates on x-axis and shows std deviation bands around the mean wind speed.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from matplotlib.dates import DateFormatter
    import pandas as pd
    
    plt.figure(figsize=(14, 8))
    
    # Check if 'time' column contains datetime objects, if not, convert it
    if 'time' in track_df.columns:
        if not isinstance(track_df['time'].iloc[0], pd.Timestamp):
            track_df['time'] = pd.to_datetime(track_df['time'])
        dates = track_df['time']
    else:
        # If no time column, use time_ind with a base date (adjust as needed)
        import pandas as pd
        from datetime import datetime, timedelta
        base_date = datetime(2010, 8, 15)  # Adjust this based on your data
        dates = [base_date + timedelta(hours=3*idx) for idx in track_df['time_ind']]
    
    # Create a line for mean wind speed with standard deviation bands
    mean_wind = track_df['wind_speed_mean']
    std_wind = track_df['wind_speed_std']
    
    # Plot mean wind speed line
    plt.plot(dates, mean_wind, 'b-', linewidth=2, label='Mean Wind Speed (m/s)')
    
    # Plot standard deviation bands
    plt.fill_between(dates, 
                     mean_wind - std_wind, 
                     mean_wind + std_wind, 
                     color='blue', 
                     alpha=0.2, 
                     label='Mean ± Std Dev')
    
    # Create a line for maximum wind speed
    plt.plot(dates, track_df['wind_speed_max'], 'r-', linewidth=2, label='Maximum Wind Speed (m/s)')
    
    # Create a line for minimum wind speed
    plt.plot(dates, track_df['wind_speed_min'], 'g-', linewidth=2, label='Minimum Wind Speed (m/s)')
    
    plt.xlabel('Date/Time')
    plt.ylabel('Wind Speed (m/s)')
    plt.title('Wind Speed Evolution with Standard Deviation')
    plt.legend()
    plt.grid(True)
    
    # Format x-axis dates
    date_format = DateFormatter('%Y-%m-%d %H:%M')
    plt.gca().xaxis.set_major_formatter(date_format)
    plt.gcf().autofmt_xdate()  # Rotate date labels for better readability
    
    # Ensure y-axis has some margin
    y_min = min(min(mean_wind - std_wind), min(track_df['wind_speed_min']))
    y_max = max(max(mean_wind + std_wind), max(track_df['wind_speed_max']))
    plt.ylim(y_min - 1, y_max + 1)
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/wind_speed_over_time_composite.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_heat_fluxes_over_time(track_df, output_dir):
    """
    Plot heat fluxes (sensible, latent, and total) over the lifetime of the track.
    Uses real dates on x-axis.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from matplotlib.dates import DateFormatter
    import pandas as pd
    
    plt.figure(figsize=(14, 8))
    
    # Check if 'time' column contains datetime objects, if not, convert it
    if 'time' in track_df.columns:
        if not isinstance(track_df['time'].iloc[0], pd.Timestamp):
            track_df['time'] = pd.to_datetime(track_df['time'])
        dates = track_df['time']
    else:
        # If no time column, use time_ind with a base date (adjust as needed)
        import pandas as pd
        from datetime import datetime, timedelta
        base_date = datetime(2010, 8, 15)  # Adjust this based on your data
        dates = [base_date + timedelta(hours=3*idx) for idx in track_df['time_ind']]
    
    # Plot sensible heat flux (HFX)
    plt.plot(dates, track_df['hfx_mean'], 'r-', linewidth=2, label='Sensible Heat Flux (HFX)')
    
    # Plot latent heat flux (LH)
    plt.plot(dates, track_df['lh_mean'], 'b-', linewidth=2, label='Latent Heat Flux (LH)')
    
    # Plot total heat flux (HFX + LH)
    plt.plot(dates, track_df['total_heat_flux_mean'], 'g-', linewidth=2, label='Total Heat Flux (HFX+LH)')
    
    # Add standard deviation bands for total heat flux
    # Calculate std of total as sqrt of sum of squared stds (assuming independence)
    total_std = np.sqrt(track_df['hfx_std']**2 + track_df['lh_std']**2)
    plt.fill_between(dates, 
                     track_df['total_heat_flux_mean'] - total_std, 
                     track_df['total_heat_flux_mean'] + total_std, 
                     color='green', 
                     alpha=0.2, 
                     label='Total Heat Flux ± Std Dev')
    
    plt.xlabel('Date/Time')
    plt.ylabel('Heat Flux (W/m²)')
    plt.title('Heat Flux Evolution Over Time')
    plt.legend()
    plt.grid(True)
    
    # Format x-axis dates
    date_format = DateFormatter('%Y-%m-%d %H:%M')
    plt.gca().xaxis.set_major_formatter(date_format)
    plt.gcf().autofmt_xdate()  # Rotate date labels for better readability
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/heat_fluxes_over_time_composite.png", dpi=150, bbox_inches='tight')
    plt.close()



def plot_all_variables_min_max_range(track_df, output_dir):
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
    ax1.plot(dates, track_df['T2_mean'], 'b-', linewidth=2, label='Mean Temperature')
    ax1.fill_between(dates, track_df['T2_min'], track_df['T2_max'], 
                     color='blue', alpha=0.2, label='Min-Max Range')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('Temperature Evolution')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # 2. Wind Speed plot
    ax2.plot(dates, track_df['wind_speed_mean'], 'g-', linewidth=2, label='Mean Wind Speed')
    ax2.fill_between(dates, track_df['wind_speed_min'], track_df['wind_speed_max'], 
                     color='green', alpha=0.2, label='Min-Max Range')
    ax2.set_ylabel('Wind Speed (m/s)')
    ax2.set_title('Wind Speed Evolution')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    # 3. Heat Fluxes plot
    ax3.plot(dates, track_df['hfx_mean'], 'r-', linewidth=2, label='Sensible Heat Flux (HFX)')
    ax3.plot(dates, track_df['lh_mean'], 'b-', linewidth=2, label='Latent Heat Flux (LH)')
    ax3.plot(dates, track_df['total_heat_flux_mean'], 'purple', linewidth=2, label='Total Heat Flux')
    
    # Add min-max range for total heat flux
    total_flux_max = track_df['hfx_max'] + track_df['lh_max']
    total_flux_min = track_df['hfx_min'] + track_df['lh_min']
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



# def find_wind_file_for_date(wind_dir, component, date_str):
#     """Find the wind file for a specific date (ignoring time part)."""
#     # Extract only the date part (YYYY-MM-DD) if there's a time component
#     if len(date_str) > 10:
#         date_str = date_str[:10]  # Take only YYYY-MM-DD part
    
#     pattern = os.path.join(wind_dir, f"{component}_{date_str}.nc")
#     files = glob.glob(pattern)
#     if files:
#         return files[0]
#     return None


def find_wind_file_for_date(wind_dir, component, date_str):
    """Find the wind file for a specific date (ignoring time part)."""
    
    pattern = os.path.join(wind_dir, f"{component}_{date_str}.nc")
    files = glob.glob(pattern)
    if files:
        return files[0]
    return None


def load_wind_data(wind_dir, date_str, pressure_level=500):
    """
    Load wind data (ue and ve) for a specific date and pressure level.
    pressure_level should be specified in hPa directly (e.g., 500 for 500 hPa, 850 for 850 hPa)
    
    Examples:
        - pressure_level=500 corresponds to 500 hPa level
        - pressure_level=850 corresponds to 850 hPa level
    """
        
    ue_file = find_wind_file_for_date(f"{wind_dir}/u10e", "u10e", date_only)
    ve_file = find_wind_file_for_date(f"{wind_dir}/v10e", "v10e", date_only)
    
    if not ue_file or not ve_file:
        print(f"Wind data files not found for date {date_only}")
        return None, None
    
    try:
        ue_data = xr.open_dataset(ue_file)
        ve_data = xr.open_dataset(ve_file)
        
        # Check which time dimension is present: 'time' or 'Time'
        time_dim = 'Time' if 'Time' in ue_data.dims else 'time'
        print(f"Using time dimension: '{time_dim}' found in wind data")
        
        # If time is specified in date_str, try to find the closest time in the data
        if len(date_str) > 10:
            try:
                # Parse the full datetime from input string
                full_datetime = pd.to_datetime(date_str)
                
                # Convert dataset time values to pandas datetime objects
                if time_dim == 'Time':
                    ue_times = pd.to_datetime(ue_data.Time.values)
                else:
                    ue_times = pd.to_datetime(ue_data.time.values)
                
                # Find closest time index
                time_idx = np.argmin(np.abs((ue_times - full_datetime).total_seconds()))
                print(f"Matched time {full_datetime} to closest available time in wind data (index {time_idx})")
            except Exception as e:
                print(f"Warning: Could not match exact time, using first time step. Error: {e}")
                time_idx = 0
        else:
            # If no time specified, use first time step
            time_idx = 0
            
        # # Select appropriate time and pressure level
        # # First select the time, then the pressure level by actual hPa value
        # print(f"Selecting pressure level: {pressure_level} hPa")
        # ue_slice = ue_data.ue.isel(**{time_dim: time_idx}).sel(interp_level=pressure_level, method='nearest')
        # ve_slice = ve_data.ve.isel(**{time_dim: time_idx}).sel(interp_level=pressure_level, method='nearest')
        
        # # Print the actual selected pressure level for verification
        # selected_level = ue_slice.interp_level.values
        # print(f"Selected pressure level: {selected_level} hPa")
        
        # # Create a simple dataset to return with the wind slices
        # wind_ds = xr.Dataset({
        #     'ue': ue_slice,
        #     've': ve_slice
        # })

        # Create a simple dataset to return with the wind slices
        wind_ds = xr.Dataset({
            'ue': ue_data,
            've': ve_data
        })
        
        return wind_ds, wind_ds  # Return same dataset for both ue and ve for compatibility
        
    except Exception as e:
        print(f"Error loading wind data for date {date_only}: {e}")
        return None, None

def load_wind_10m_data(wind_dir, date_str):
    """
    Load 10m wind data (u10e and v10e).
    """
    # Extract only date part if datetime with time is provided
    if len(date_str) > 10:
        date_only = date_str[:4]  # Take only YYYY-MM-DD part
    else:
        date_only = date_str
        
    u10e_file = find_wind_file_for_date(f"{wind_dir}/u10e", "NAAD77km_u10e", date_only)
    v10e_file = find_wind_file_for_date(f"{wind_dir}/v10e", "NAAD77km_v10e", date_only)
    
    if not u10e_file or not v10e_file:
        print(f"10m wind data files not found for date {date_only}")
        return None, None
    
    try:
        u10e_data = xr.open_dataset(u10e_file)
        v10e_data = xr.open_dataset(v10e_file)
        
        # Check which time dimension is present: 'time' or 'Time'
        time_dim = 'Time' if 'Time' in u10e_data.dims else 'time'
        print(f"Using time dimension: '{time_dim}' found in wind data")
        
        # If time is specified in date_str, try to find the closest time in the data
        if len(date_str) > 10:
            try:
                # Parse the full datetime from input string
                full_datetime = pd.to_datetime(date_str)
                
                # Convert dataset time values to pandas datetime objects
                if time_dim == 'Time':
                    u10e_times = pd.to_datetime(u10e_data.Time.values)
                else:
                    u10e_times = pd.to_datetime(u10e_data.time.values)
                
                # Find closest time index
                time_idx = np.argmin(np.abs((u10e_times - full_datetime).total_seconds()))
                print(f"Matched time {full_datetime} to closest available time in wind data (index {time_idx})")
            except Exception as e:
                print(f"Warning: Could not match exact time, using first time step. Error: {e}")
                time_idx = 0
        else:
            # If no time specified, use first time step
            time_idx = 0
            
        # Select appropriate time
        u10e_slice = u10e_data.u10e.isel(**{time_dim: time_idx})
        v10e_slice = v10e_data.v10e.isel(**{time_dim: time_idx})
        
        # Create a simple dataset to return with the wind slices
        wind_ds = xr.Dataset({
            'u10e': u10e_slice,
            'v10e': v10e_slice
        })
        
        return wind_ds, wind_ds  # Return same dataset for both u10e and v10e for compatibility
        
    except Exception as e:
        print(f"Error loading 10m wind data for date {date_only}: {e}")
        return None, None


def find_wrf_file_for_date(data_dir, date_str):
    """Find the WRF output file for a specific date."""
    # Extract only the date part (YYYY-MM-DD) if there's a time component
    if len(date_str) > 10:
        date_only = date_str[:10]  # Take only YYYY-MM-DD part
    else:
        date_only = date_str
    
    # Try direct pattern - note that WRF files include time in their name
    # Example: wrfout_d01_2010-01-01_00:00:00
    pattern = os.path.join(data_dir, f"wrfout_d01_{date_only}_*")
    files = glob.glob(pattern)
    
    if files:
        print(f"Found WRF file matching date {date_only}: {os.path.basename(files[0])}")
        return files[0]
    
    # If no exact match, try a more flexible approach 
    # Look for any WRF files in the directory
    pattern = os.path.join(data_dir, "wrfout_d01_*")
    all_files = glob.glob(pattern)
    
    if all_files:
        # Just return the first file if we can't find an exact match
        # You might want to enhance this with more sophisticated date matching
        print(f"No exact match for {date_only}, using first available WRF file: {os.path.basename(all_files[0])}")
        return all_files[0]
    
    print(f"No WRF files found in directory {data_dir}")
    return None

def load_heat_flux_data(data_dir, date_str):
    """Load HFX and LH heat flux data from WRF output files."""
    # Extract only date part if datetime with time is provided
    if len(date_str) > 10:
        date_only = date_str[:10]  # Take only YYYY-MM-DD part
    else:
        date_only = date_str
    
    wrf_file = find_wrf_file_for_date(data_dir, date_only)
    
    if not wrf_file:
        print(f"WRF output files not found for date {date_only}")
        return None, None
    
    try:
        wrf_data = xr.open_dataset(wrf_file)
        
        # Check which time dimension is present: 'time' or 'Time'
        time_dim = 'Time' if 'Time' in wrf_data.dims else 'time'
        print(f"Using time dimension: '{time_dim}' found in WRF data")
        
        # If time is specified in date_str, try to find the closest time in the data
        if len(date_str) > 10:
            try:
                # Parse the full datetime from input string
                full_datetime = pd.to_datetime(date_str)
                
                # Handle different time formats in WRF files
                if time_dim == 'Time' and hasattr(wrf_data, 'Times'):
                    # Convert byte strings to datetime objects
                    time_strings = []
                    for time_bytes in wrf_data.Times.values:
                        if isinstance(time_bytes, bytes):
                            time_str = time_bytes.decode('utf-8')
                        elif isinstance(time_bytes, np.ndarray):
                            # Join character array into string
                            time_str = ''.join([c.decode('utf-8') if isinstance(c, bytes) else c for c in time_bytes])
                        else:
                            time_str = str(time_bytes)
                        time_strings.append(time_str)
                    
                    # Convert to datetime objects
                    wrf_times = pd.to_datetime(time_strings, format='%Y-%m-%d_%H:%M:%S')
                else:
                    # Standard xarray time handling
                    wrf_times = pd.to_datetime(getattr(wrf_data, time_dim).values)
                
                # Find closest time index
                time_idx = np.argmin(np.abs((wrf_times - full_datetime).total_seconds()))
                print(f"Matched time {full_datetime} to closest available time in WRF data (index {time_idx})")
            except Exception as e:
                print(f"Warning: Could not match exact time in WRF data, using first time step. Error: {e}")
                time_idx = 0
        else:
            # If no time specified, use first time step
            time_idx = 0
        
        # Extract HFX (sensible heat flux) and LH (latent heat flux) at the selected time
        if 'HFX' in wrf_data and 'LH' in wrf_data:
            hfx_data = wrf_data['HFX'].isel(**{time_dim: time_idx})
            lh_data = wrf_data['LH'].isel(**{time_dim: time_idx})
            return hfx_data, lh_data
        else:
            print(f"HFX or LH variables not found in WRF data")
            return None, None
    except Exception as e:
        print(f"Error loading heat flux data for date {date_str}: {e}")
        return None, None

def compute_statistics(data_array, mask, convert_from_kelvin=False):
    """Compute extended statistics for a data array within a mask.
    If convert_from_kelvin is True, converts all temperature values from K to °C before computing statistics."""
    if data_array is None:
        return {
            'mean': np.nan,
            'median': np.nan,
            'max': np.nan,
            'min': np.nan,
            'std': np.nan,
            'p90': np.nan,
            'anomaly': np.nan,
            'n_points': 0
        }

    # Check if data_array has values attribute (xarray) or is already a numpy array
    if hasattr(data_array, 'values'):
        values = data_array.values
    else:
        values = data_array
        
    masked_data = values[mask]

    if len(masked_data) == 0 or np.all(np.isnan(masked_data)):
        return {
            'mean': np.nan,
            'median': np.nan,
            'max': np.nan,
            'min': np.nan,
            'std': np.nan,
            'p90': np.nan,
            'anomaly': np.nan,
            'n_points': 0
        }
    
    # Convert from Kelvin to Celsius if needed (BEFORE computing statistics)
    if convert_from_kelvin:
        masked_data = masked_data - 273.15

    mean_val = np.nanmean(masked_data)
    median_val = np.nanmedian(masked_data)
    max_val = np.nanmax(masked_data)
    min_val = np.nanmin(masked_data)
    std_val = np.nanstd(masked_data)
    p90_val = np.nanpercentile(masked_data, 90)

    # Anomaly calculation (max - mean for cyclonic, min - mean for anticyclonic)
    anomaly = max_val - mean_val  # For cyclonic (can be modified based on circ parameter)

    return {
        'mean': mean_val,
        'median': median_val,
        'max': max_val,
        'min': min_val,
        'std': std_val,
        'p90': p90_val,
        'anomaly': anomaly,
        'n_points': np.sum(~np.isnan(masked_data))
    }
def process_track_with_extended_analysis(temp_file, track_file, wrf_data_dir, wind_dir, output_dir, circ='C', visualize=True, pressure_level=500):
    """
    Process track with extended analysis including:
    - Temperature statistics
    - Wind statistics at specified pressure level (default 500 hPa)
    - Heat flux statistics (HFX and LH in W/m²)
    
    Parameters:
    -----------
    temp_file : str
        Path to temperature data file
    track_file : str
        Path to track CSV file
    wrf_data_dir : str
        Directory containing WRF output files
    wind_dir : str
        Directory containing wind data files
    output_dir : str
        Directory to save output files
    circ : str, default 'C'
        Circulation type ('C' for cyclonic, 'A' for anticyclonic)
    visualize : bool, default True
        Whether to create visualization images
    pressure_level : int, default 500
        Pressure level in hPa (e.g., 500 for 500 hPa, 850 for 850 hPa)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing track with extended elliptical analysis")
    print(f"Temperature file: {temp_file}")
    print(f"Track file: {track_file}")
    print(f"WRF data directory: {wrf_data_dir}")
    print(f"Wind directory: {wind_dir}")
    print(f"Using pressure level: {pressure_level} hPa")
    
    print("Loading temperature data...")
    temp_data = xr.open_dataset(temp_file)
    print(f"Temperature data loaded successfully!")
    print(f"Dimensions: {temp_data.dims}")
    print(f"Variables: {list(temp_data.data_vars)}")
    
    print(f"Loading track file...")
    track_df = pd.read_csv(track_file)
    print(f"Track loaded with {len(track_df)} points")
    
    print("Calculating ellipse parameters with improved method...")
    track_df = calculate_ellipse_parameters_improved(track_df)
    
    # Initialize lists for statistics
    results = {
        'T2_mean': [], 'T2_median': [], 'T2_max': [], 'T2_min': [], 'T2_std': [], 
        'T2_p90': [], 'T2_anomaly': [], 'ellipse_points': [],
        'wind_speed_mean': [], 'wind_speed_median': [], 'wind_speed_max': [], 
        'wind_speed_min': [], 'wind_speed_std': [], 'wind_speed_p90': [],
        'hfx_mean': [], 'hfx_median': [], 'hfx_max': [], 'hfx_min': [], 'hfx_std': [], 'hfx_p90': [],
        'lh_mean': [], 'lh_median': [], 'lh_max': [], 'lh_min': [], 'lh_std': [], 'lh_p90': [],
        'total_heat_flux_mean': []  # Сумма HFX и LH
    }
    
    # Get dimensions for mask creation
    south_north_dim = 'south_north' if 'south_north' in temp_data.sizes else 'y'
    west_east_dim = 'west_east' if 'west_east' in temp_data.sizes else 'x'
    mask_shape = (temp_data.sizes[south_north_dim], temp_data.sizes[west_east_dim])
    
    print("Processing elliptical regions for each track point...")
    for i in tqdm(range(len(track_df))):
        row = track_df.iloc[i]
        time_idx = int(row.time_ind)
        
        # Extract center coordinates and ellipse parameters
        center_x = int(row.pxc_ind)
        center_y = int(row.pyc_ind)
        a = float(row.semi_major_axis)
        b = float(row.semi_minor_axis)
        theta = float(row.orientation_rad)
        
        # Ensure minimum sizes for a and b
        a = max(a, 3)
        b = max(b, 2)
        
        # Create ellipse mask
        mask = get_ellipse_mask(center_x, center_y, a, b, theta, mask_shape)
        
        try:
            # Process temperature data
            temp_slice = temp_data['t2'].isel(time=time_idx)
            
            # Pass convert_from_kelvin=True to handle Kelvin to Celsius conversion
            temp_stats = compute_statistics(temp_slice, mask, convert_from_kelvin=True)
            
            # Extract date from row if available, otherwise construct from other fields
            full_date_str = None
            if 'time' in row:
                full_date_str = str(row['time'])
                date_str = full_date_str.split('T')[0] if 'T' in full_date_str else full_date_str
            else:
                # Construct date from time_ind and available date information
                # This is a placeholder - adjust based on your actual data structure
                date_str = "2010-08-15"
                full_date_str = date_str
            
            # Process heat flux data (HFX and LH)
            hfx_data, lh_data = load_heat_flux_data(wrf_data_dir, full_date_str)
            
            hfx_stats = {'mean': np.nan, 'median': np.nan, 'max': np.nan, 'min': np.nan, 'std': np.nan, 'p90': np.nan}
            lh_stats = {'mean': np.nan, 'median': np.nan, 'max': np.nan, 'min': np.nan, 'std': np.nan, 'p90': np.nan}
            
            if hfx_data is not None and lh_data is not None:
                # Data is already at the right time from load_heat_flux_data
                hfx_stats = compute_statistics(hfx_data, mask)
                lh_stats = compute_statistics(lh_data, mask)
            
            # Process 10m wind data
            wind_ds, _ = load_wind_10m_data(wind_dir, full_date_str)
            
            wind_speed_stats = {'mean': np.nan, 'median': np.nan, 'max': np.nan, 
                               'min': np.nan, 'std': np.nan, 'p90': np.nan}
            
            if wind_ds is not None:
                # Data is already at the right time from load_wind_10m_data
                u10e_slice = wind_ds['u10e'] 
                v10e_slice = wind_ds['v10e']
                
                # Calculate wind speed
                if hasattr(u10e_slice, 'values'):
                    u10e_values = u10e_slice.values
                    v10e_values = v10e_slice.values
                else:
                    u10e_values = u10e_slice
                    v10e_values = v10e_slice
                    
                wind_speed = np.sqrt(u10e_values**2 + v10e_values**2)
                wind_speed_stats = compute_statistics(wind_speed, mask)
    
            # Store results
            results['wind_speed_mean'].append(wind_speed_stats['mean'])
            results['wind_speed_median'].append(wind_speed_stats['median'])
            results['wind_speed_max'].append(wind_speed_stats['max'])
            results['wind_speed_min'].append(wind_speed_stats['min'])
            results['wind_speed_std'].append(wind_speed_stats['std'])
            results['wind_speed_p90'].append(wind_speed_stats['p90'])
        
            # Store results
            results['T2_mean'].append(temp_stats['mean'])
            results['T2_median'].append(temp_stats['median'])
            results['T2_max'].append(temp_stats['max'])
            results['T2_min'].append(temp_stats['min'])
            results['T2_std'].append(temp_stats['std'])
            results['T2_p90'].append(temp_stats['p90'])
            results['ellipse_points'].append(temp_stats['n_points'])
            
            
            results['hfx_mean'].append(hfx_stats['mean'])
            results['hfx_median'].append(hfx_stats['median'])
            results['hfx_max'].append(hfx_stats['max'])
            results['hfx_min'].append(hfx_stats['min'])
            results['hfx_std'].append(hfx_stats['std'])
            results['hfx_p90'].append(hfx_stats['p90'])
            
            results['lh_mean'].append(lh_stats['mean'])
            results['lh_median'].append(lh_stats['median'])
            results['lh_max'].append(lh_stats['max'])
            results['lh_min'].append(lh_stats['min'])
            results['lh_std'].append(lh_stats['std'])
            results['lh_p90'].append(lh_stats['p90'])
            
            # Calculate total heat flux (HFX + LH)
            if not np.isnan(hfx_stats['mean']) and not np.isnan(lh_stats['mean']):
                results['total_heat_flux_mean'].append(hfx_stats['mean'] + lh_stats['mean'])
            else:
                results['total_heat_flux_mean'].append(np.nan)
            
        except Exception as e:
            print(f"Error processing point {i} (time={time_idx}): {e}")
            import traceback
            traceback.print_exc()
            # Fill with NaN for this point
            for key in results.keys():
                results[key].append(np.nan)

    # Add all results to the track dataframe
    for key, values in results.items():
        track_df[key] = values
    
    # Save the results
    output_file = f"{output_dir}/track_extended_analysis_{pressure_level}hPa.csv"
    track_df.to_csv(output_file, index=False)
    print(f"Processed track saved to {output_file}")
    
    # Print summary statistics
    print("\nTemperature statistics summary (°C):")
    print(f"  Mean T2: {np.nanmean(track_df['T2_mean']):.2f}")
    print(f"  Max T2: {np.nanmax(track_df['T2_max']):.2f}")
    print(f"  Min T2: {np.nanmin(track_df['T2_min']):.2f}")
    print(f"  Median T2: {np.nanmean(track_df['T2_median']):.2f}")
    print(f"  90th percentile T2: {np.nanmean(track_df['T2_p90']):.2f}")
    print(f"  Mean points in ellipse: {np.nanmean(track_df['ellipse_points']):.1f}")
    
    print(f"\nWind statistics summary at {pressure_level} hPa (m/s):")
    print(f"  Mean wind speed: {np.nanmean(track_df['wind_speed_mean']):.2f}")
    print(f"  Median wind speed: {np.nanmean(track_df['wind_speed_median']):.2f}")
    print(f"  Max wind speed: {np.nanmax(track_df['wind_speed_max']):.2f}")
    print(f"  Min wind speed: {np.nanmin(track_df['wind_speed_min']):.2f}")
    
    print("\nHeat flux statistics summary (W/m²):")
    print(f"  Mean sensible heat flux (HFX): {np.nanmean(track_df['hfx_mean']):.2f}")
    print(f"  Mean latent heat flux (LH): {np.nanmean(track_df['lh_mean']):.2f}")
    print(f"  Mean total heat flux (HFX+LH): {np.nanmean(track_df['total_heat_flux_mean']):.2f}")
    
    # Create composite plots for all variables
    print("Creating composite plots for temperature, wind speed, and heat fluxes...")
    
    # Temperature composite plot (existing)
    plot_temperature_over_time(track_df, output_dir)
    
    # NEW: Wind speed composite plot
    plot_wind_speed_over_time(track_df, output_dir)
    
    # NEW: Heat fluxes composite plot
    plot_heat_fluxes_over_time(track_df, output_dir)
    
    # NEW: Combined variables composite plot
    plot_all_variables_min_max_range(track_df, output_dir)
    
    if visualize:
        print("Creating visualizations with improved ellipses for select time steps...")
        # Only visualize a few time steps to avoid too many images
        vis_indices = np.linspace(0, len(track_df)-1, min(5, len(track_df)), dtype=int)
        for i in vis_indices:
            time_idx = int(track_df.iloc[i].time_ind)
            visualize_ellipse_improved(temp_data, track_df, time_idx, output_dir)
        print(f"Visualizations saved to {output_dir}")
    
    return track_df

def main():
    """Main function to run the analysis."""
    # Input directories and files
    data_dir = "./"
    temp_file = os.path.join("/storage/OPENDATA/NAAD/LoRes/Surface/t2/NAAD77km_t2_2010.nc")
    track_file = os.path.join(data_dir, "earl_500hpa.csv")
    wrf_data_dir = os.path.join("/storage/NAAD/NAAD/LoRes/2010")
    wind_dir = os.path.join("/storage/OPENDATA/NAAD/LoRes/Surface")
    output_dir = "output/extended_elliptical_analysis"
    
    circ = 'C'  # Cyclonic circulation
    
    # Use actual pressure levels in hPa now
    pressure_level = 500  # 500 hPa
    
    print(f"Using circulation type: {circ} (Cyclonic)")
    print(f"Using pressure level: {pressure_level} hPa")
    print("Using improved ellipse calculation that matches check_track.py")
    
    try:
        processed_df = process_track_with_extended_analysis(
            temp_file=temp_file,
            track_file=track_file,
            wrf_data_dir=wrf_data_dir,
            wind_dir=wind_dir,
            output_dir=output_dir,
            circ=circ,
            visualize=True,
            pressure_level=pressure_level
        )
        
        print("Extended analysis completed successfully!")
        
    except Exception as e:
        print(f"Error in main execution: {e}")


if __name__ == "__main__":
    main()