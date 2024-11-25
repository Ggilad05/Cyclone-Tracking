import multiprocessing
from datetime import datetime, timedelta
import numpy as np
import xarray as xr
import glob
import re
from Arrange_data_corrected import get_nc
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors


def adjust_map(ds, time, level=None, time_dim='time', level_dim='level'):
    """
       Adjusts the spatial range of a dataset by tripeling and shifting longitude values.

       Parameters:
           ds (xarray.Dataset): The input dataset to adjust.
           time (datetime): The specific time to extract data for.
           level (int, optional): Pressure level to select. Default is None.
           time_dim (str): Name of the time dimension in the dataset.
           level_dim (str): Name of the pressure level dimension in the dataset.

       Returns:
           xarray.Dataset: Dataset with adjusted spatial range.
    """

    # Select data based on the specified time and level
    selected_data = ds.sel(**{time_dim: time})
    if level:
        selected_data = selected_data.sel(**{level_dim: level})

    # Step 1: Create shifted longitude arrays by duplicating the data
    ds_shifted_r = selected_data.copy(deep=True)
    ds_shifted_l = selected_data.copy(deep=True)
    ds_shifted_r['longitude'] = selected_data['longitude'] + 360
    ds_shifted_l['longitude'] = selected_data['longitude'] - 360

    # Combine the original and shifted datasets for longitudes
    ds_long_tripled = xr.concat([ds_shifted_l, selected_data, ds_shifted_r], dim='longitude')


    return ds_long_tripled


def cut_polygon(reanalysis_data, time, x, y, space_lon, space_lat, resolution, var, level=None):
    """
      Extracts a rectangular region (polygon) around a cyclone center from the reanalysis data.

      Parameters:
          reanalysis_data (xarray.DataArray): Dataset containing reanalysis data.
          time (datetime): Time of the cyclone.
          x (float): Longitude of the cyclone center.
          y (float): Latitude of the cyclone center.
          space_lon (float): Longitudinal range to extract.
          space_lat (float): Latitudinal range to extract.
          resolution (float): Spatial resolution of the data.
          var (str): Variable to extract (e.g., 'temperature').
          level (int, optional): Pressure level to select. Default is None.

      Returns:
          np.ndarray: Extracted data region as a numpy array.
      """

    longitudes = np.arange(x - space_lon, x + space_lon, resolution)
    latitudes = np.arange(y - space_lat, y + space_lat, resolution)

    # Define time and pressure level dimension names based on availability in data
    time_dim = 'valid_time' if 'valid_time' in reanalysis_data.dims else 'time'
    level_dim = 'pressure_level' if 'pressure_level' in reanalysis_data.dims else 'level'

    if np.any(longitudes >= 360) or np.any(longitudes < 0):
        # Tripled the map and adjust longitudes
        tripled_reanalysis_data = adjust_map(reanalysis_data, time, level, time_dim, level_dim)

        # Select data from the tripeled map
        polygon = tripled_reanalysis_data.sel(
            **{"longitude": longitudes, "latitude": latitudes},
            method='nearest'
        )
        # # Optional: plot the data for verification
        # plot_combined_and_tripeled_data(tripled_reanalysis_data, polygon, longitudes, latitudes, var, level)
    else:
        polygon = reanalysis_data.sel(**{time_dim: time, "longitude": longitudes, "latitude": latitudes}, method='nearest')
        reanalysis_data_red = reanalysis_data.sel(**{time_dim: time})
        if level:
            polygon = polygon.sel(**{level_dim: level})
            reanalysis_data_red = reanalysis_data_red.sel(**{level_dim: level})

        # plot_combined_and_tripeled_data(reanalysis_data_red, polygon, longitudes, latitudes, var, level)

    return np.flipud(polygon[var].to_numpy())  # Flip data for proper orientation


def plot_combined_and_tripeled_data(doubled_reanalysis_data, combined_ds, longitudes, latitudes, var, level=None):
    """
      Visualizes the doubled dataset and the extracted region for verification.

      Parameters:
          doubled_reanalysis_data (xarray.Dataset): Dataset with adjusted spatial range.
          combined_ds (xarray.Dataset): Extracted region of interest.
          longitudes (np.ndarray): Longitude values of the extracted region.
          latitudes (np.ndarray): Latitude values of the extracted region.
          var (str): Variable being plotted.
          level (int, optional): Pressure level being plotted. Default is None.
      """

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Get the color normalization range based on combined_ds[var]
    vmin = combined_ds[var].min().item()
    vmax = combined_ds[var].max().item()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # First subplot: Doubled Reanalysis Data with Marked Area
    ax1 = axes[0]
    im1 = ax1.imshow(
        np.flipud(doubled_reanalysis_data[var]),
        extent=[
            doubled_reanalysis_data.longitude.min(),
            doubled_reanalysis_data.longitude.max(),
            doubled_reanalysis_data.latitude.min(),
            doubled_reanalysis_data.latitude.max(),
        ],
        origin='lower',
        norm=norm,  # Apply the same normalization
        cmap='coolwarm',  # Use the 'bwr' colormap
    )
    # Add rectangle to the plot
    rect = patches.Rectangle(
        (longitudes.min(), latitudes.min()),
        longitudes.ptp(),
        latitudes.ptp(),
        linewidth=2,
        edgecolor='b',
        facecolor='none',
    )
    ax1.add_patch(rect)

    # Set title and labels for the first subplot
    if level:
        ax1.set_title(f'Doubled Reanalysis Data with Marked Area, level= {level}')

    else:
        ax1.set_title('Doubled Reanalysis Data with Marked Area')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')

    # Second subplot: Combined Data
    ax2 = axes[1]
    combined_ds[var].plot(
        ax=ax2,
        norm=norm,  # Use the same normalization
        cmap='coolwarm',  # Same colormap
        add_colorbar=False,  # Suppress the colorbar for this subplot
    )
    ax2.set_title('Combined Dataset')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')

    # Add a horizontal colorbar below the first subplot
    cbar = fig.colorbar(
        im1, ax=ax1, orientation='horizontal', fraction=0.05, pad=0.1
    )
    cbar.set_label(f"{var} (units)")  # Adjust label as needed

    # Adjust layout and show the figure
    plt.tight_layout()
    plt.show()

def process_track(data, track_id, v, level, reanalysis_directory, space_lon, space_lat, resolution,
                  starting_season_date, file_name, year):
    """
    Processes a single cyclone track to extract reanalysis variables along the track.

    Parameters:
        data (xarray.Dataset): Cyclone track dataset.
        track_id (int): ID of the cyclone track to process.
        v (str): Variable to extract (e.g., 'temperature').
        level (int, optional): Pressure level to process. Default is None.
        reanalysis_directory (str): Path to reanalysis data.
        space_lon (float): Longitudinal range to extract.
        space_lat (float): Latitudinal range to extract.
        resolution (float): Spatial resolution of the data.
        starting_season_date (datetime): Reference start date of the cyclone season.
        file_name (str): Name of the file being processed.
        year (int): Year of the track data.
    """

    print(
        f"Processing {'level-dependent variable' if level else 'surface variable'} {v} "
        f"for track {track_id.values} in file {file_name}"
    )

    # Initialize tensors dictionary for multiple time offsets
    tensors = {0: [], 6: [], 12: [], 18: [], 24: []}
    tensors_indexes = [0, 2, 4, 6, 8]
    x_p, y_p = [], []

    # Filter time indexes and remove zeros
    time_indexes = data.sel(trackid=track_id)['t'].to_numpy()
    time_indexes = time_indexes[time_indexes != 0]
    odd_time_indexes = time_indexes[time_indexes % 2 != 0]

    # Preload dataset if `v` is 'z_geo'
    preloaded_reanalysis = None
    if v == 'z_geo':
        preloaded_reanalysis = xr.open_dataset(reanalysis_directory)


    for i in odd_time_indexes:
        if i == 17.0:
            print('s')
        x = data.sel(trackid=track_id)['lon'].where(data.sel(trackid=track_id)['t'] == i).dropna(dim='points').data[0]
        y = data.sel(trackid=track_id)['lat'].where(data.sel(trackid=track_id)['t'] == i).dropna(dim='points').data[0]
        time = starting_season_date + timedelta(hours=3) * (i - 1)
        x_p.append(x)
        y_p.append(y)

        for offset in tensors_indexes:
            future_index = i + offset
            if future_index > odd_time_indexes[-1]:
                continue

            x_future = data.sel(trackid=track_id)['lon'].where(data.sel(trackid=track_id)['t'] == future_index).dropna(
                dim='points').data[0]
            y_future = data.sel(trackid=track_id)['lat'].where(data.sel(trackid=track_id)['t'] == future_index).dropna(
                dim='points').data[0]

            if v != 'z_geo':
                reanalysis_data, _, _ = get_nc(time.year, time.month, reanalysis_directory)
                result = cut_polygon(reanalysis_data, time, x_future, y_future, space_lon, space_lat, resolution, v,
                                     level)
            else:
                result = cut_polygon(preloaded_reanalysis, '1979-01-01T00:00:00', x_future, y_future, space_lon,
                                     space_lat, resolution, 'z', level)

            tensors[offset * 3].append(result)


    # Save results
    # output_path = f"/data/iacdc/ECMWF/ERA5/Gilad/check/v4/{v}{'a' if level else ''}/{level or ''}/{year}/{file_name}_{int(track_id.values)}.npy"
    # np.save(output_path, np.array(tensors[0]))
    # print(f"Saved result for {v} to {output_path}")

def plot_track(x, y, name, num):
    """
      Plots the cyclone track on a global map.

      Parameters:
          x (list): Longitudes of the cyclone track.
          y (list): Latitudes of the cyclone track.
          name (str): Name of the cyclone or dataset.
          num (int): Cyclone track number.
      """

    # Plotting the cyclone track
    plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='tan')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
    gl.top_labels = False  # Remove top latitude labels
    gl.right_labels = False  # Remove right longitude labels
    gl.xlabel_style = {'size': 10, 'color': 'black'}
    gl.ylabel_style = {'size': 10, 'color': 'black'}

    # Emphasize original points
    plt.plot(x, y, 'ro', markersize=8, transform=ccrs.PlateCarree())
    ax.set_extent([0, 360, -90, 90], crs=ccrs.PlateCarree())

    # Add title and legend
    plt.title(f'Cyclone Track: {name} {num} ')
    # plt.legend()
    plt.show()
    # # Show the plot
    # plt.show()


def pre_tracks(file_path):
    """Load and filter cyclone tracks based on criteria."""

    def filter_tracks(ds):
        nonz = (ds.t.data != 0).sum(axis=0)  # Lifetime of the track
        E = ds.intensity.data
        b = np.argmax(E, axis=0)  # Time of maximum intensity
        lati = np.abs(ds.lat.data[0])  # Latitude at genesis
        loni = ds.lon.data[0]  # Longitude at genesis
        lonm = ds.lon.data[b, np.arange(len(b))]  # Longitude at max intensity
        dlon = np.mod(lonm - loni, 360)  # Distance between genesis and max intensity
        dldt = dlon / (b + 1)  # Speed of the storm
        ind = (lati < 60) & (lati > 20) & (nonz > 16) & (dldt > 0.3) & (dlon < 200)
        return ds.isel(trackid=ind)

    ds = xr.open_dataset(file_path).load().data
    names = ['t', 'lon', 'lat', 'intensity']
    datasets = [ds[i, :, :].to_dataset(name=names[i]).drop('variables') for i in range(4)]
    ds = xr.merge(datasets)
    return filter_tracks(ds)


# Main execution
if __name__ == '__main__':
    data_path = "/data/shreibshtein/OrCyclones"
    start_year, end_year = 1979, 2020
    space_lon, space_lat, resolution = 17.5, 15, 1.25

    # Variables without levels
    surface_variables = ['v10', 'u10', 't2m', 'sp', 'tp', 'psl', 'tcw', 'slhf', 'sshf', 'z_geo']
    surface_directories = [
        "/data/iacdc/ECMWF/ERA5/4xday_1.25_global_1000-200hPa/vas/vas_6hrPlev_reanalysis_ERA5_",
        "/data/iacdc/ECMWF/ERA5/4xday_1.25_global_1000-200hPa/uas/uas_6hrPlev_reanalysis_ERA5_",
        "/data/iacdc/ECMWF/ERA5/4xday_1.25_global_1000-200hPa/tas/tas_6hrPlev_reanalysis_ERA5_",
        "/data/iacdc/ECMWF/ERA5/4xday_1.25_global_1000-200hPa/ps/ps_6hrPlev_reanalysis_ERA5_",
        "/data/iacdc/ECMWF/ERA5/4xday_1.25_global_1000-200hPa/pr/pr_6hrPlev_reanalysis_ERA5_",
        "/data/iacdc/ECMWF/ERA5/4xday_1.25_global_1000-200hPa/psl/psl_6hrPlev_reanalysis_ERA5_",
        "/data/iacdc/ECMWF/ERA5/4xday_1.25_global_1000-200hPa/tpw/tcw_6hrPlev_reanalysis_ERA5_",
        "/data/iacdc/ECMWF/ERA5/4xday_1.25_global_1000-200hPa/hfls/hfls_6hrPlev_reanalysis_ERA5_",
        "/data/iacdc/ECMWF/ERA5/4xday_1.25_global_1000-200hPa/sshf/sshf_6hrPlev_reanalysis_ERA5_",
        '/data/iacdc/ECMWF/ERA5/4xday_1.25_global_1000-200hPa/sfc_geo/correct/sfc_geopotential_6hr_reanalysis_ERA5_1979-01-01_1979-01-31.nc'
    ]


    # Variables with levels
    level_variables = ['t', 'q', 'v', 'u', 'z']
    levels = [250, 300, 500, 850]
    level_directories = [
        "/data/iacdc/ECMWF/ERA5/4xday_1.25_global_1000-200hPa/ta/ta_6hrPlev_reanalysis_ERA5_",
        "/data/iacdc/ECMWF/ERA5/4xday_1.25_global_1000-200hPa/hus/hus_6hrPlev_reanalysis_ERA5_",
        "/data/iacdc/ECMWF/ERA5/4xday_1.25_global_1000-200hPa/va/va_6hrPlev_reanalysis_ERA5_",
        "/data/iacdc/ECMWF/ERA5/4xday_1.25_global_1000-200hPa/ua/ua_6hrPlev_reanalysis_ERA5_",
        "/data/iacdc/ECMWF/ERA5/4xday_1.25_global_1000-200hPa/z/z_6hrPlev_reanalysis_ERA5_"
    ]


    year_pattern = re.compile(r'_(\d{4})_')
    all_files = glob.glob(f"{data_path}/*.nc")
    file_paths = [file for file in all_files if start_year <= int(year_pattern.search(file).group(1)) <= end_year]

    for file in file_paths:

        print(f"Processing file: {file}")
        file_name = file.split('/')[-1].split('.')[0]
        season, year = file_name.split('_')[0], file_name.split('_')[1]
        starting_season_date = datetime(int(year), {'MAM': 3, 'DJF': 12, 'JJA': 6, 'SON': 9}[season], 1)

        data = pre_tracks(file)
        for track_id in data["trackid"]:
            tasks = []

            # Create tasks for surface variables
            for v, reanalysis_directory in zip(surface_variables, surface_directories):
                tasks.append((data, track_id, v, None, reanalysis_directory, space_lon, space_lat, resolution,
                              starting_season_date, file_name, year))

            # Create tasks for level variables
            for v, reanalysis_directory in zip(level_variables, level_directories):
                for level in levels:
                    tasks.append((data, track_id, v, level, reanalysis_directory, space_lon, space_lat, resolution,
                                  starting_season_date, file_name, year))

            # Process all tasks in parallel
            with multiprocessing.Pool(processes=30) as pool:
                pool.starmap(process_track, tasks)

        print(f"Completed processing for file {file_name}")

    print("All files processed.")
