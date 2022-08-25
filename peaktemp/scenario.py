import forecast
import matplotlib.pyplot as plt
import pandas as pd


def create_scenarios(isd_data, daily_extremes, climate_scenarios):
    """
    Fits all profiles for ALL climate models in a given set of climate scenarios
    Parameters
    ----------
    isd_data: dict
        A dictionary containing historical hourly temperature profiles
    daily_extremes: dict
        A dictionary containing daily extrema; the output of data.get_daily_extrema
    climate_scenarios: dict
        A dictionary containing outputs of data.aggregate_cmip6_data for each climate scenario
    Returns
    -------
    dict
        A dictionary with fitted scenarios for all climate scenarios and models
    """
    fitted_scenarios = {}
    for scenario in climate_scenarios:
        fitted_scenarios[scenario] = forecast.create_full_dataset(isd_data, daily_extremes, climate_scenarios[scenario])
    return fitted_scenarios


def plot_scenarios(isd_data, fitted_scenarios, labels, title, path):
    """
    Plots all climate scenarios, with 95% confidence intervals, against historical peaks
    Parameters
    ----------
    isd_data: dict
        A dictionary containing historical hourly temperature profiles
    fitted_scenarios: dict
        A dictionary containing fitted climate scenarios; the output of scenario.create_scenarios
    labels: list
        The labels, in order, for the different climate scenarios
    title:str
        The title of the output plot
    path: str
        The path of the output plot
    Returns
    -------
    None
        The resulting plot will be saved to the specified path
    """
    # Set plot style and rename columns
    plt.style.use('seaborn-colorblind')
    columns = {hr: "HR" + str(hr + 1) for hr in range(24)}

    # Extract models and historical data
    data_dicts = list(fitted_scenarios.values())
    historical_df = pd.concat(isd_data.values()) \
        .pivot(index="date", columns="hour", values="hourly_temp") \
        .rename(columns=columns)

    historical_df = forecast.extract_dt_info(historical_df)

    # Plot historical data
    forecast.get_seasonal_peaks(historical_df)[["summer_peak_temp"]] \
        .rename(columns={'summer_peak_temp': 'Historical Summer Peak'}) \
        .plot(figsize=(15, 10), title=title, style='k-', zorder=30,
              label="Historical Summer Peak")

    # Compute and plot 95% CI
    for index, data_dict in enumerate(data_dicts):
        ribbon = forecast.calculate_peaks(forecast.combine_models(isd_data, data_dict), level=20 / 19) \
            .rename(columns={"rolling_summer_peak": "lower_bound"}) \
            .join(forecast.calculate_peaks(forecast.combine_models(isd_data, data_dict), level=20)) \
            .rename(columns={"rolling_summer_peak": "upper_bound"})

        plt.fill_between(
            ribbon.index,
            ribbon["lower_bound"],
            ribbon["upper_bound"],
            zorder=25 - 5 * index,
            alpha=0.5 + 0.1 * index,
            label=labels[index]
        )

    plt.legend(loc='upper left')
    plt.ylabel("Temperature (Actual and Forecasted)")

    plt.savefig(path)
