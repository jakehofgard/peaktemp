from peaktemp import forecast
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
    climate_scenarios: list
        A list containing outputs of data.aggregate_cmip6_data for each climate scenario
    Returns
    -------
    list
        A list with fitted scenarios for all climate scenarios and models
    Examples
    --------
    create_scenarios(
        isd_data=isd_data,
        daily_extremes=daily_extremes,
        climate_scenarios=[extracted_models_1_2_6, extracted_models_5_8_5]
    )
    """
    fitted_scenarios = []
    for scenario in climate_scenarios:
        fitted_scenario = forecast.fit_multiple_profiles(isd_data, daily_extremes, scenario)
        fitted_scenarios.append(fitted_scenario)
    return fitted_scenarios


def plot_scenarios(isd_data, fitted_scenarios, labels, title, path):
    """
    Plots all climate scenarios, with 95% confidence intervals, against historical peaks
    Parameters
    ----------
    isd_data: dict
        A dictionary containing historical hourly temperature profiles
    fitted_scenarios: list
        A list containing fitted climate scenarios; the output of scenario.create_scenarios
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
    Examples
    --------
    scenario.plot_scenarios(
        isd_data=isd_data,
        fitted_scenarios=fitted_scenarios,
        labels=["SSP1-2.6", "SSP5-8.5"],
        title="Example Scenario Analysis",
        path="test_scenario.png"
    )
    """
    # Set plot style and rename columns
    plt.style.use('seaborn-colorblind')
    columns = {hr: "HR" + str(hr + 1) for hr in range(24)}

    # Extract models and historical data
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
    for index, data_dict in enumerate(fitted_scenarios):
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
