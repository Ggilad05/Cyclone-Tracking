import multiprocessing as mp
from Arrange_data_corrected import get_nc
import numpy as np

def process_month(year, month, reanalysis_directory, v):
    reanalysis_data, longitudes, latitudes = get_nc(year, month, reanalysis_directory)
    if (np.min(longitudes) != 0.0) or (np.max(longitudes) != 358.75) or (np.min(latitudes) != -90) or (np.max(latitudes) != 90) or (reanalysis_data.data_vars._ipython_key_completions_()[0] != v):
        print(f'year: {year}, month: {month}, variable: {v}')
        print(reanalysis_data)


def process_year(v, reanalysis_directory, year):
    for month in range(1, 13):
        process_month(year, month, reanalysis_directory, v)

def lon_lan_check():
    variables = ['v10', 'u10', 't2m', 'sp', 'tp', 'psl', 'tcw', 'hfls', 'ta','q', 'ua', 'ta', 'z']
    start_year, end_year = 1958, 2020
    reanalysis_directories = [
        "/data/iacdc/ECMWF/ERA5/4xday_1.25_global_1000-200hPa/vas/vas_6hrPlev_reanalysis_ERA5_",
        "/data/iacdc/ECMWF/ERA5/4xday_1.25_global_1000-200hPa/uas/uas_6hrPlev_reanalysis_ERA5_",
        "/data/iacdc/ECMWF/ERA5/4xday_1.25_global_1000-200hPa/tas/tas_6hrPlev_reanalysis_ERA5_",
        "/data/iacdc/ECMWF/ERA5/4xday_1.25_global_1000-200hPa/ps/ps_6hrPlev_reanalysis_ERA5_",
        "/data/iacdc/ECMWF/ERA5/4xday_1.25_global_1000-200hPa/pr/pr_6hrPlev_reanalysis_ERA5_",
        "/data/iacdc/ECMWF/ERA5/4xday_1.25_global_1000-200hPa/psl/psl_6hrPlev_reanalysis_ERA5_",
        "/data/iacdc/ECMWF/ERA5/4xday_1.25_global_1000-200hPa/tpw/tcw_6hrPlev_reanalysis_ERA5_",
        "/data/iacdc/ECMWF/ERA5/4xday_1.25_global_1000-200hPa/hfls/hfls_6hrPlev_reanalysis_ERA5_",
        "/data/iacdc/ECMWF/ERA5/4xday_1.25_global_1000-200hPa/ta/ta_6hrPlev_reanalysis_ERA5_",
        "/data/iacdc/ECMWF/ERA5/4xday_1.25_global_1000-200hPa/hus/hus_6hrPlev_reanalysis_ERA5_",
        "/data/iacdc/ECMWF/ERA5/4xday_1.25_global_1000-200hPa/ua/ua_6hrPlev_reanalysis_ERA5_",
        "/data/iacdc/ECMWF/ERA5/4xday_1.25_global_1000-200hPa/ta/ta_6hrPlev_reanalysis_ERA5_",
        "/data/iacdc/ECMWF/ERA5/4xday_1.25_global_1000-200hPa/z/z_6hrPlev_reanalysis_ERA5_"
    ]

    # Create a pool of workers
    pool = mp.Pool(mp.cpu_count())

    for v, reanalysis_directory in zip(variables, reanalysis_directories):
        print(f'var: {v}', f'dir: {reanalysis_directory}')
        # Use the pool to process each year in parallel
        pool.starmap(process_year, [(v, reanalysis_directory, year) for year in range(start_year, end_year + 1)])

    # Close the pool and wait for the work to finish
    pool.close()
    pool.join()

if __name__ == "__main__":
    lon_lan_check()

    ''''v10', 'u10', 't2m', 'sp', 'tp', 'psl', 'tcw', 'hfls', 'ta','q', 'ua', 'ta', 'z' are fine from 1958 - 2020'''
