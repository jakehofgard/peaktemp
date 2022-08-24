# Plotting and data processing
import numpy as np
import pandas as pd

# Tools for working with CDS API
from geopy.geocoders import Nominatim
from geopy import distance
import os
import cdsapi
import zipfile
import xarray as xr
import re

# Packages for working with NCEI ISD data
from datetime import date
from noaastn import noaastn

# Suppress unnecessary pandas output
pd.options.mode.chained_assignment = None


def download_cmip6_model(model, scenario, location, path, margin=2, start_year=1990, end_year=2050):
    """
    Downloads CMIP6 data for a given climate model, scenario, and location from the Copernicus Climate Data Store.
    Parameters
    ----------
    model : str
        The name of the desired climate model
    scenario: str
        The name of the desired climate scenario
    location: str
        The name of the desired location for climate projections.
    path: str
        A path name for the ZIP files downloaded from the CDS
    margin: float, optional
        The size of the bounding box surrounding the specified location. By default, this is 2, which is
        usually sufficiently coarse for most CMIP6 models
    start_year: int, optional
        The start year of the data download. Cannot be earlier than 1950, and is set to 1990 by default.
    end_year: int, optional
        The end year of the data download. Cannot be later than 2100, and is set to 2050 by default.
    Returns
    -------
    None
        Downloaded files will be located in the specified directory.
    Examples
    --------
    download_cmip6_model(
        model="miroc6",
        scenario="ssp5_8_5",
        location="Phoenix Sky Harbor",
        path="phoenix_miroc6_SSP5-8.5"
    )
    """

    # Retrieve geospatial information from input location
    geolocator = Nominatim(user_agent="CMIP6 Data Download")
    location = geolocator.geocode(location)
    lat, lon = location.latitude, location.longitude

    # If one doesn't exist, create a directory to store zipped CMIP6 files
    if not os.path.exists(path):
        os.makedirs(path)
        print("Created new data directory at " + path)

    # Generate bounding box for downloaded data
    bbox = list([lat + margin, lon - margin, lat - margin, lon + margin])

    # Generate dates for data download
    dates = str(start_year) + "-01-01/" + str(end_year) + "-01-01"
    # Initialize connection to CDS API
    client = cdsapi.Client()

    # Retrieve daily maximums from given model
    client.retrieve(
        'projections-cmip6',
        {
            'temporal_resolution': 'daily',
            'experiment': scenario,
            'level': 'single_levels',
            'variable': 'daily_maximum_near_surface_air_temperature',
            'model': model,
            'date': dates,
            'area': bbox,
            'format': 'zip',
        },
        path + '/daily_max_' + model + '_' + scenario + '.zip'
    )

    # Retrieve daily minimums from given model
    client.retrieve(
        'projections-cmip6',
        {
            'temporal_resolution': 'daily',
            'experiment': scenario,
            'level': 'single_levels',
            'variable': 'daily_minimum_near_surface_air_temperature',
            'model': model,
            'date': dates,
            'area': bbox,
            'format': 'zip',
        },
        path + '/daily_min_' + model + '_' + scenario + '.zip'
    )


# The following are all utility functions for processing and cleaning CMIP6 data.


def _k_to_f(temp):
    return (temp - 273.15) * 9 / 5 + 32


def _open_nc_files(folder, path, lat, lon):
    nc = xr.open_dataset(os.path.join(folder, path))
    ds = nc.sel(lon=lon, lat=lat, method='nearest')
    return ds


def _convert_index_to_datetime(timestamp):
    return pd.to_datetime(str(timestamp)[:10])


def extract_cmip6_data(folder, model, official_name, scenario, location):
    """
    Extracts and processes downloaded CMIP6 model data.
    Parameters
    ----------
    folder: str
        The folder where the downloaded CMIP6 model data is stored
    model : str
        The name of the desired climate model
    official_name: str
        The official name of the downloaded climate scenario (i.e., capitalized correctly, etc.)
    scenario: str
        The name of the desired climate scenario
    location: str
        The name of a specific location (i.e., Denver, Colorado) for timeseries extraction
    Returns
    -------
    pd.DataFrame
        A DataFrame containing all the extracted CMIP6 model data.
    """

    # Format file paths
    files = [
        'daily_min_' + model + '_' + scenario + '.zip', 'daily_max_' + model + '_' + scenario + '.zip',
        'daily_min_' + model + '_historical.zip', 'daily_max_' + model + '_historical.zip'
    ]

    # Unzip downloaded files from CMIP6 climate data store
    for file in files:
        with zipfile.ZipFile(os.path.join(folder, file), 'r') as z:
            z.extractall(folder)

    files = os.listdir(folder)

    # The file paths for historical and projected models are different
    nc_max_path_hist = \
        list(filter(re.compile("tasmax_day_" + official_name + "_historical.*nc").match, files))[0]
    nc_min_path_hist = \
        list(filter(re.compile("tasmin_day_" + official_name + "_historical.*nc").match, files))[0]
    nc_max_path = list(
        filter(re.compile("tasmax_day_" + official_name + "_" + scenario.replace("_", "") + ".*nc").match, files)
    )[0]
    nc_min_path = list(
        filter(re.compile("tasmin_day_" + official_name + "_" + scenario.replace("_", "") + ".*nc").match, files)
    )[0]

    # Retrieve geospatial information from input location
    geolocator = Nominatim(user_agent="CMIP6 Data Extraction")
    location = geolocator.geocode(location)
    lat, lon = location.latitude, location.longitude

    # Only the downloaded .nc files are relevant and contain the desired timeseries
    ds_max = _open_nc_files(folder, nc_max_path, lat, lon)
    ds_max_hist = _open_nc_files(folder, nc_max_path_hist, lat, lon)
    ds_min = _open_nc_files(folder, nc_min_path, lat, lon)
    ds_min_hist = _open_nc_files(folder, nc_min_path_hist, lat, lon)

    # Join all downloaded data into a single dataframe for a given model and scenario
    max_df = pd.concat([ds_max_hist['tasmax'].to_dataframe()[["tasmax"]], ds_max['tasmax'].to_dataframe()[["tasmax"]]])
    min_df = pd.concat([ds_min_hist['tasmin'].to_dataframe()[["tasmin"]], ds_min['tasmin'].to_dataframe()[["tasmin"]]])
    full_df = max_df.join(min_df, how="outer") \
        .assign(
        daily_max=lambda df: _k_to_f(df.tasmax),  # Need to convert everything from Kelvin
        daily_min=lambda df: _k_to_f(df.tasmin)
    )[["daily_max", "daily_min"]]

    full_df.index = full_df.index.map(_convert_index_to_datetime)

    # Close .nc files
    ds_max.close(); ds_max_hist.close(); ds_min.close(); ds_min_hist.close()

    # Delete extra files that result from unzipping
    for f in files:
        if f.endswith(("nc", "png", "json")):
            os.remove(os.path.join(folder, f))

    return full_df


def aggregate_cmip6_data(path, climate_models, scenario, location):
    """
    Extracts and processes downloaded CMIP6 model data for a collection of climate models.
    Parameters
    ----------
    path: str
        The folder where the downloaded CMIP6 model data is stored
    climate_models: str
        A dictionary containing the desired climate models and their 'official' names
    scenario: str
        The name of the desired climate scenario
    location: str
        The name of a specific location (i.e., Denver, Colorado) for timeseries extraction
    Returns
    -------
    dict
        A dictionary containing DataFrames with the extracted data from each model
        Examples
    --------
    aggregate_cmip6_data(
        path="phoenix_miroc6_SSP5-8.5",
        climate_models={"miroc6": "MIROC6", "gfdl_esm4": "GFDL-ESM4", "canesm5: CanESM5"}
        scenario="ssp5_8_5",
        location="Phoenix, AZ"
    )
    """
    data_dict = {}
    for model in climate_models:
        try:
            data_dict[model] = extract_cmip6_data(path, model, climate_models[model], scenario, location)
        except:
            print("Climate scenario " + scenario + " not available for model " + model)

    return data_dict


def _isd_c_to_f(temp_str):
    """
    Reformats ISD default temperature strings into readable Fahrenheit temperatures.
    Parameters
    ----------
    temp_str: str
        A string formatted like the temperature downloads from the ISD.
    Returns
    -------
    float
        A temperature in Fahrenheit
    """
    sign = temp_str[0]
    temp_str = temp_str[1:].lstrip("0").replace(",", ".")

    # Sometimes, there are unpredictable inputs that need to be caught by this blanket exception (sorry)
    try:
        c_temp = float(''.join([n for n in temp_str if not n.isalpha()])) / 10
    except:
        c_temp = float('nan')
    if sign == "-":
        c_temp = -c_temp
    return c_temp * 9 / 5 + 32


def _retrieve_station(location):
    """
    Given the name of a NOAA weather station, retrieves a station number used for ISD downloads.
    Parameters
    ----------
    location: str
        The desired location for ISD data retrieval.
    Returns
    -------
    str
        A station number
    """
    geolocator = Nominatim(user_agent="ISD Data Download")

    # Retrieve stations and process station metadata
    stations = noaastn.get_stations_info(country="US")
    stations["latitude"], stations["longitude"] = stations["latitude"].astype("float"), stations["longitude"].astype(
        "float")
    stations["length"] = stations["end"] - stations["start"]
    stations = stations[stations["length"] > "10000 days"]

    # Given an input, find the closest station with satisfactory data range
    location = geolocator.geocode(location)
    coords = (location.latitude, location.longitude)
    stations["coords"] = list(zip(stations.lat, stations.lon))

    # Calculate closest station, using latitude/longitude inputs
    stations["dist"] = stations.apply(lambda row: distance.distance(coords, (row['lat'], row['lon'])).miles, axis=1)
    station = stations[stations["dist"] == stations["dist"].min()]
    station = station.iloc[0]

    # Retrieve the FIRST viable station
    number = station["usaf"] + station["wban"]
    return number


def _fill_missing_values(input_df, year):
    """
    Fills missing data from ISD downloads, linearly interpolating missing hours.
    Parameters
    ----------
    input_df: pd.DataFrame
        A pandas DataFrame containing a single year of hourly profiles
    year: int
        The year to be interpolated (required due to the leap year case, which must be handled separately)
    Returns
    -------
    None
        The original hourly profile dictionary is updated.
    """
    data = input_df[year]
    start = data["date"][0]

    # Need to handle leap years
    if (year - 2020) % 4 == 0:
        period = 8784
    else:
        period = 8760

    full_timescale = pd.DataFrame(pd.date_range(start=start, periods=period, freq='H'))
    full_timescale = full_timescale.rename(columns={0: "timestamp"})
    full_timescale["hour"] = full_timescale["timestamp"].dt.hour
    full_timescale["date"] = full_timescale["timestamp"].dt.strftime('%Y-%m-%d')
    new_data = data.merge(full_timescale, on=["date", "hour"], how="outer").set_index("timestamp")
    new_data = new_data.interpolate(method="time")
    new_data = new_data.sort_index()

    input_df[year] = new_data


def download_isd_profiles(location, start_year=date.today().year - 30, end_year=date.today().year):
    """
    Downloads and cleans ISD data for a given location, over a given timeframe
    Parameters
    ----------
    location: str
        The desired location of the data download. When possible, format as "CITY, STATE"
    start_year: int
        The start year of the data download, set to be 30 years earlier than the current year by default (in line with NOAA weather normals).
    end_year: int
        The end year of the data download, set to the current year by default.
    Returns
    -------
    dict
        A dictionary containing pd.DataFrames for each year of downloaded, cleaned ISD data.
    """
    all_data = {}
    # Retrieve the closest station with at least 30 years of data
    station_number = _retrieve_station(location)
    # Download all data from NCEI for each year and clean the resulting data
    for year in range(start_year, end_year + 1):
        print(f"Retrieving {location} data for {str(year)}")
        path = "https://www.ncei.noaa.gov/data/global-hourly/access/" + str(year) + "/" + station_number + ".csv"
        data = pd.read_csv(path)
        data = data[["DATE", "TMP"]].rename(columns={"DATE": "timestamp", "TMP": "temp"})
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data["temp"] = data["temp"].map(_isd_c_to_f).apply(lambda val: val if val <= 150 else float('nan'))
        data["hour"] = data["timestamp"].dt.hour
        data["date"] = data["timestamp"].dt.strftime('%Y-%m-%d')
        data = data.drop(columns=["timestamp"])
        data = data.groupby(['date', 'hour']) \
            .agg(
            hourly_temp=pd.NamedAgg(column="temp", aggfunc=np.mean)
        ) \
            .reset_index()

        # Interpolate linearly to fill in missing data when possible
        data = data.interpolate(method='linear', limit_direction='backward')
        all_data[year] = data

    # Interpolate all remaining missing observations to give full hourly profile
    for year in all_data:
        _fill_missing_values(all_data, year)

    # Remove years when no data was measured
    keys = list(all_data.keys())
    for year in keys:
        if len(all_data[year]) > len(all_data[year].dropna()):
            del all_data[year]

    return all_data


def get_daily_extrema(data):
    """
    Produces a DataFrame with daily maxima, minima, and differences for the input data.
    Parameters
    ----------
    data: dict
        A dictionary of pd.DataFrames, likely the output of download_isd_profiles, containing hourly profiles.
    Returns
    -------
    pd.DataFrame
        A DataFrame containing daily minima, maxima, and differences.
    """
    extrema_dict = {}
    for year in data:
        extrema_dict[year] = data[year].groupby(['date']) \
            .agg(
            daily_max=pd.NamedAgg(column="hourly_temp", aggfunc=max),
            daily_min=pd.NamedAgg(column="hourly_temp", aggfunc=min)
        ) \
            .reset_index() \
            .set_index('date')
        extrema_dict[year]["diff"] = extrema_dict[year]["daily_max"] - extrema_dict[year]["daily_min"]

    # Filter out invalid data for the current year
    current_year = date.today().year
    extrema_dict[current_year] = extrema_dict[current_year][extrema_dict[current_year]["diff"] != 0]

    return extrema_dict
