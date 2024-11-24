import pandas as pd
import datetime
import xarray as xr
import numpy as np


def read_trajectories(year):
    """year  - The list year for which I want the storm tracks"""

    # file_path = 'C:/Users/shrei/PycharmProjects/MasterProject/tracks_ERA5_1979-2020_0.25deg_1hr/' + str(year) + '.txt'
    file_path = '/data/shreibshtein/tracks_ERA5_1979-2020_0.25deg_1hr/' + str(year) + '.txt'

    data = pd.read_table(file_path, sep=" ", names=["Index", "Longitude", "Latitude", "Year", "Month", "Day",
                                                    "Hour", "Lowest MSLP value"])
    data = data.groupby(by="Index")

    index = []
    grouped_data = []
    pressure_at_peak = []
    genesys = []
    peak = []
    lysis = []
    diff_gp = []
    diff_pl = []
    diff_gl = []
    for i in range(1, len(data) + 1):

        group = data.get_group(i)
        index.append(i)
        grouped_data.append(data.get_group(i))

        pressure_at_peak.append(group['Lowest MSLP value'].min())
        genesys.append(group.iloc[0])
        peak_data = group.loc[group["Lowest MSLP value"] == group['Lowest MSLP value'].min()]

        if len(peak_data) > 1:
            peak_data = peak_data.iloc[-1]

        peak.append(peak_data)
        lysis.append(group.iloc[-1])

        genesys_time = datetime.datetime(year=int(year), month=int(group.iloc[0]['Month']),
                                         day=int(group.iloc[0]['Day']), hour=int(group.iloc[0]['Hour']))
        peak_time = datetime.datetime(year=int(year), month=int(peak_data['Month']), day=int(peak_data['Day']),
                                      hour=int(peak_data['Hour']))
        lysis_time = datetime.datetime(year=int(year), month=int(group.iloc[-1]['Month']),
                                       day=int(group.iloc[-1]['Day']), hour=int(group.iloc[-1]['Hour']))

        diff_gp.append(peak_time - genesys_time)
        diff_pl.append(lysis_time - peak_time)
        diff_gl.append(lysis_time - genesys_time)

    data_grouped = pd.DataFrame(
        {"Index": index, "Grouped data": grouped_data, "Pressure at peak": pressure_at_peak, "Genesys": genesys,
         "Peak": peak, "Lysis": lysis, "Diff G-P": diff_gp, "Diff P-L": diff_pl, "Diff G-L": diff_gl})

    # print(data_grouped.where(data_grouped["Index"] ==1).iloc[0]["Diff G-P"])
    # print(data_grouped.where(data_grouped["Index"] == 1).iloc[0])

    return data_grouped


def get_nc(year, month, reanalysis_dir):
    month = float(month)
    number_of_days = pd.Timestamp(year=year, month=int(month), day=1).daysinmonth
    if isinstance(month, float):
        if month == 1.0:
            month = "01"
        if month == 2.0:
            month = "02"
        if month == 3.0:
            month = "03"
        if month == 4.0:
            month = "04"
        if month == 5.0:
            month = "05"
        if month == 6.0:
            month = "06"
        if month == 7.0:
            month = "07"
        if month == 8.0:
            month = "08"
        if month == 9.0:
            month = "09"
        if month == 10:
            month = "10"
        if month == 11:
            month = "11"
        if month == 12:
            month = "12"

    # fn_tp = "/data/iacdc/ECMWF/ERA5/hourly_0.25_global_1000-200hPa/pr/pr_yr_reanalysis_ERA5_" + str(
    #     year) + month + "01_" + str(year) + month + str(number_of_days) + ".nc"
    # fn_tp = reanalysis_dir + str(
    #     year) + str(month) + "01_" + str(year) + str(month) + str(number_of_days) + ".nc"
    # Define file name formats
    fn_tp_format0 = reanalysis_dir + f"{year}-{str(month).zfill(2)}-01_{year}-{str(month).zfill(2)}-{str(number_of_days).zfill(2)}.nc"
    fn_tp_format1 = reanalysis_dir + f"{year}{str(month).zfill(2)}01_{year}{str(month).zfill(2)}{str(number_of_days).zfill(2)}.nc"
    fn_tp_format2 = reanalysis_dir + f"{year}-{str(month).zfill(2)}-01_{year}-{str(month).zfill(2)}-{str(number_of_days).zfill(2)}.nc"
    fn_tp_format3 = reanalysis_dir + f"{year}{str(month).zfill(2)}01_{year}{str(month).zfill(2)}{str(number_of_days).zfill(2)}.nc"
    # fn_tp_format4 = '/home/shreibshtein/Downloads/tp_1940_1957.nc'

    # List of formats to try
    formats = [fn_tp_format0, fn_tp_format1, fn_tp_format2, fn_tp_format3]

    # Try opening the dataset with each format
    for fn in formats:
        # print(fn)
        try:
            tp = xr.open_dataset(fn)
            break
        except FileNotFoundError:
            continue
    else:
        print(print(reanalysis_dir))
        raise FileNotFoundError("None of the file formats were found.")


    # Load lat and lon
    lats = tp.variables['latitude'][:]
    lons = tp.variables['longitude'][:]

    return tp, lons, lats
