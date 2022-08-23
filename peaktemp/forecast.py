# Plotting and data processing
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Packages for working with NCEI ISD data
from datetime import date

# XGBoost and training metrics
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.model_selection import ShuffleSplit

pd.options.mode.chained_assignment = None


def extract_dt_info(df):
    """
    Reformats DataFrames with necessary DateTime information
    Parameters
    ----------
    df: pd.DataFrame
        The input DataFrame.
    Returns
    -------
    pd.DataFrame
        An updated DataFrame with all desired fields.
    """
    df["month"] = pd.to_datetime(df.index).month
    df["day"] = pd.to_datetime(df.index).day
    return df


def _calculate_coefs(row):
    """
    Calculate coefficients for fitting an hourly profile to model data
    Parameters
    ----------
    row: pd.Series
        A single row from a pd.DataFrame.
    Returns
    -------
    list
        The unique coefficients to perform a linear transformation on a historical hourly profile.
    """
    return np.linalg.solve(
        [
            [row["daily_max_actual"], 1],
            [row["daily_min_actual"], 1]
        ],
        [
            row["daily_max_projected"],
            row["daily_min_projected"]
        ]
    )


def _fit_single_profile(isd_data, daily_extremes, model_data, fit_year, profile_year):
    """
    Fit a single year of model data, given a historical reference year
    Parameters
    ----------
    isd_data: dict
        A dictionary containing hourly profiles; the output of data.download_isd_profiles
    daily_extremes: dict
        A dictionary containing daily extrema; the output of data.get_daily_extrema
    model_data: pd.DataFrame
        A DataFrame containing CMIP6 model data; the output of data.extract_cmip6_data
    fit_year: int
        The year to which an hourly profile will be fit
    profile_year: int
        The year to use as a reference for an hourly profile
    Returns
    -------
    pd.DataFrame
        A DataFrame containing the new hourly temperature profile for fit_year
    """
    # Retrieve all data sources
    profile = isd_data[profile_year].pivot(index="date", columns="hour", values="hourly_temp")
    extremes = daily_extremes[profile_year]
    model = model_data
    model["year"] = pd.to_datetime(model.index).year

    # Find specific year that needs an hourly profile
    model = model[model["year"] == fit_year]

    profile = extract_dt_info(profile)
    extremes = extract_dt_info(extremes)
    model = extract_dt_info(model)

    # Interpolation handles leap year cases
    merged = extremes.merge(model, on=["month", "day"], suffixes=("_actual", "_projected"), how="right").interpolate()
    merged[["alpha", "beta"]] = merged.apply(lambda row: _calculate_coefs(row), axis=1, result_type='expand')
    coef_df = merged[["month", "day", "alpha", "beta"]]
    merged_profile = profile.merge(coef_df, on=["month", "day"], how="right")

    new_profile = merged_profile.copy()

    # Perform the fit using the solved coefficients, unique to each day's profile
    for hr in range(24):
        new_profile[hr] = new_profile[hr] * new_profile["alpha"] + new_profile["beta"]

    return new_profile.drop(["alpha", "beta"], axis=1).merge(model, on=["month", "day"])


def compare_profiles(profile_1, profile_2):
    """
    Compute the annual MAPE between two hourly temperature profiles, according to the Council's methodology
    This method was developed by Daniel Hua at the NWPCC.
    Parameters
    ----------
    profile_1: pd.DataFrame
        A DataFrame containing an hourly profile of temperatures for a single year.
    profile_2: pd.DataFrame
        A DataFrame containing an hourly profile of temperatures for a single year.
    Returns
    -------
    float
        The MAPE between the two hourly profiles.
    """
    return mape(
        profile_1.unstack().to_frame().sort_index(level=1).T.iloc[0],
        profile_2.unstack().to_frame().sort_index(level=1).T.iloc[0]
    )


def _find_best_fit(isd_data, daily_extremes, model_data, profile_year):
    """
    Fit a single year of model data, given a historical reference year
    Parameters
    ----------
    isd_data: dict
        A dictionary containing hourly profiles; the output of data.download_isd_profiles
    daily_extremes: dict
        A dictionary containing daily extrema; the output of data.get_daily_extrema
    model_data: pd.DataFrame
        A DataFrame containing CMIP6 model data; the output of data.extract_cmip6_data
    profile_year: int
        The year to use as a reference for an hourly profile
    Returns
    -------
    pd.DataFrame
        A DataFrame containing the optimal hourly temperature profile
    """
    profiles = []
    profile_scores = []
    model = model_data
    model["year"] = pd.to_datetime(model.index).year
    # Fit all historical back-forecasts to their corresponding hourly profile
    for year in isd_data:
        if year == date.today().year:
            break
        profiles.append(_fit_single_profile(isd_data, daily_extremes, model_data, profile_year, year).interpolate())
    average_profile = pd.concat(profiles).groupby(level=0).mean()
    # Compare all profiles to the average profile and select the one that minimizes MAPE
    for profile in profiles:
        profile_scores.append(compare_profiles(profile, average_profile))
    optimal_profile = profiles[profile_scores.index(min(profile_scores))]

    return optimal_profile


def fit_entire_profile(isd_data, daily_extremes, model_data):
    """
    Given model data, fit hourly profiles to the entire timeframe
    Parameters
    ----------
    isd_data: dict
        A dictionary containing hourly profiles; the output of data.download_isd_profiles
    daily_extremes: dict
        A dictionary containing daily extrema; the output of data.get_daily_extrema
    model_data: pd.DataFrame
        A DataFrame containing CMIP6 model data; the output of data.extract_cmip6_data
    Returns
    -------
    pd.DataFrame
        A DataFrame containing the entire set of hourly temperature profiles for the given model
    """
    profiles = []
    model = model_data
    model["year"] = pd.to_datetime(model.index).year
    years = model["year"].unique()
    years = years[years > 1990]
    for year in years:
        if year % 10 == 0:
            print(f"Finished fitting profiles for {year - 10} -- {year - 1}")
        profiles.append(_find_best_fit(isd_data, daily_extremes, model_data, year))
    full_profile = pd.concat(profiles)
    return full_profile


def create_full_dataset(isd_data, daily_extremes, climate_models):
    """
    Given a set models, fit hourly profiles to each model
    Parameters
    ----------
    isd_data: dict
        A dictionary containing hourly profiles; the output of data.download_isd_profiles
    daily_extremes: dict
        A dictionary containing daily extrema; the output of data.get_daily_extrema
    climate_models: dict
        A dictionary containing CMIP6 model data; the output of data.aggregate_model_data
    Returns
    -------
    dict
        A dictionary containing full hourly profiles for each model in climate_models
    """
    full_dataset = {}
    for model in climate_models:
        print(f'\033[1m Fitting {model} \033[0m')
        try:
            full_dataset[model] = fit_entire_profile(isd_data, daily_extremes, climate_models[model])
        except:
            print("Climate scenario not available for model " + model)
    return full_dataset


def get_seasonal_peaks(df, winter_peak_hrs=[7, 8, 9, 10, 17, 18], summer_peak_hrs=[16, 17, 18, 19, 20]):
    """
    Computes the temperatures each year (both winter and summer) that will result in peak load
    Parameters
    ----------
    df: pd.DataFrame
        A DataFrame containing an hourly temperature profile
    winter_peak_hrs: list
        A list of hours corresponding to winter peaking hours (between 0 and 23)
    summer_peak_hrs: list
        A list of hours corresponding to summer peaking hours (between 0 and 23)
    Returns
    -------
    pd.DataFrame
        A DataFrame containing seasonal peak temperatures for each year
    """
    winter_months = [1, 2, 11, 12]
    summer_months = [6, 7, 8, 9]
    # Rename columns for easier processing
    columns = {hr: "HR" + str(hr + 1) for hr in range(24)}
    # Retrieve peaking temperatures, based on PSEs seasonal definitions and their hourly specifications laid out in
    # FERC Form 1
    df = df.rename(columns=columns)
    df["Timestamp"] = pd.to_datetime(df[["year", "month", "day"]])
    long_df = pd.wide_to_long(df.reset_index(), "HR", i="Timestamp", j="hour") \
        .reset_index() \
        .set_index("Timestamp") \
        .rename(columns={"HR": "hourly_temp"})

    winter_df = long_df[(long_df["month"].isin(winter_months)) & (long_df["hour"].isin(winter_peak_hrs))]

    # Need to match seasons, not years due to seasonal patterns like El Niño, etc.
    winter_df["season_year"] = np.where(winter_df["month"].isin([11, 12]), winter_df["year"] + 1, winter_df["year"])

    winter_peak_df = winter_df.groupby("season_year") \
        .agg(
        winter_peak_temp=pd.NamedAgg(column="hourly_temp", aggfunc=min)
    )

    summer_df = long_df[(long_df["month"].isin(summer_months)) & (long_df["hour"].isin(summer_peak_hrs))]
    summer_peak_df = summer_df.groupby("year") \
        .agg(
        summer_peak_temp=pd.NamedAgg(column="hourly_temp", aggfunc=max)
    )

    forecast_df = winter_peak_df.join(summer_peak_df, how='outer').dropna()

    return forecast_df


def combine_models(isd_data, fitted_models, return_winter=False):
    """
    Computes the temperatures each year (both winter and summer) that will result in peak load
    Parameters
    ----------
    isd_data: dict
        A dictionary containing historical hourly temperature profiles
    fitted_models: dict
        A dictionary containing fit hourly profiles for a set of climate models
    return_winter: bool
        Indicates whether to return winter peaks in addition to summer peaks; False by default
    Returns
    -------
    pd.DataFrame
        A DataFrame containing seasonal peak temperatures for all models
    """
    # Rename columns for easier processing
    columns = {hr: "HR" + str(hr + 1) for hr in range(24)}
    # Process historical station data
    historical_df = pd.concat(isd_data.values()) \
        .pivot(index="date", columns="hour", values="hourly_temp") \
        .rename(columns=columns)

    historical_df = extract_dt_info(historical_df)
    historical_df["year"] = pd.to_datetime(historical_df.index).year

    # Combine all dataframes for final forecast
    combined_peak_df = get_seasonal_peaks(historical_df) \
        .rename(columns={'winter_peak_temp': 'winter_peak_0', 'summer_peak_temp': 'summer_peak_0'})

    for index, model in enumerate(fitted_models):
        combined_peak_df = combined_peak_df.join(
            get_seasonal_peaks(fitted_models[model]).rename(
                columns={'winter_peak_temp': f'winter_peak_{index + 1}', 'summer_peak_temp': f'summer_peak_{index + 1}'}
            ),
            how='outer'
        )

    # Return all columns if argument is passed
    if return_winter:
        return combined_peak_df

    # By default, just return summer peaks
    return combined_peak_df[[col for col in combined_peak_df if col.startswith('summer')]]


def _compute_single_year(stack, year, level, length=30):
    """
    Computes the  1-in-level forecast for a given forecast year
    Parameters
    ----------
    stack: pd.Series
        Aggregated data for a single forecast year
    year: int
        The year for forecasting
    level: float
        The level of the forecast (i.e., 1-in-2, 1-in-10, etc.)
    length: int
        The length of data used for forecasting; 30 years by default

    Returns
    -------
    float
        A single forecast for the given forecast year
    """
    return np.quantile(
        stack[(stack.index >= year - length / 2) & (stack.index <= year + length / 2)],
        q=level,
        method="linear"
    )


def calculate_rolling_quantiles(df, level, length=30, start_year=2005, return_winter=False):
    """
    Computes the  1-in-level forecast for an input DataFrame
    Parameters
    ----------
    df: pd.DataFrame
        The input data for the forecast; the output forecast.combine_models
    level: float
        The level of the forecast (i.e., 1-in-2, 1-in-10, etc.). Must be greater than 1
    length: int
        The length of data used for forecasting; 30 years by default
    start_year: int
        The start year of the forecast (can be in the past); 2005 by default
    return_winter: bool
        Indicates whether to return winter peaks in addition to summer peaks; False by default
    Returns
    -------
    pd.DataFrame
        A DataFrame containing the peak temperature forecast
    """
    summer_stack = df[[col for col in df if col.startswith('summer')]] \
        .stack() \
        .reset_index() \
        .rename(columns={0: "temp", "level_0": "year"}) \
        .set_index("year")[["temp"]]

    final_df = df.assign(
        rolling_summer_peak=lambda data: data.index.map(
            lambda index: _compute_single_year(summer_stack, index, 1 - 1 / level, length)
        )
    )

    if return_winter:
        winter_stack = df[[col for col in df if col.startswith('winter')]] \
            .stack() \
            .reset_index() \
            .rename(columns={0: "temp", "level_0": "year"}) \
            .set_index("year")[["temp"]]

        final_df = final_df.assign(
            rolling_winter_peak=lambda data: data.index.map(
                lambda index: _compute_single_year(winter_stack, index, 1 / level, length)
            )
        )

        final_df = final_df[final_df.index >= start_year]

        return final_df

    final_df = final_df[final_df.index >= start_year]
    return final_df[["rolling_summer_peak"]]

# TODO: Finish plotting functions and XGBoost predictions
# def forecast_xgboost():
# def plot_forecast():
