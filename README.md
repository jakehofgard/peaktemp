# ``peaktemp``
Implementation of WIEB's peak temperature forecasting tool, using Coupled Model Intercomparison Project 6 (CMIP6) climate projection models from the Copernicus Climate Data Store (CDS) and NOAA's Integrated Surface Database (ISD).

This package generalizes the peak temperature forecasting methodologies from the Northwest Power and Conservation Council (NWPCC) and Puget Sound Energy (PSE) so that planners, forecasters, and other interested parties can produce accurate, long-term temperature forecasts for any location in the United States. 

``peaktemp`` can produce peak temperature forecasts (the temperature that results in peak load) for any location in the United States using data from any CMIP6 climate model (e.g., MIROC6, CanESM5, etc.) and any CMIP6-supported climate scenario (e.g., SSP5-8.5). Forecasts are supported out to the year 2100 (although the recommended forecast range is 30 years), and in-depth comparisons between different climate scenarios are also supported. In addition to a variety of statistical forecasts for peak temperatures, an experimental machine learning-based forecast for peak temperatures, using XGBoost, is available as part of ``peaktemp``.

## Installation

``peaktemp`` is available on the Python Package Index (PyPI). First, you'll need to install a few dependencies. First, make sure that you're using Python Version 3.8.0 or higher and pip Version 21.3.1 or higher.

1. cdsapi – To download and process CMIP6 data, you'll need to sign up for a (free) Climate Data Store (CDS) account [here](https://cds.climate.copernicus.eu/user/register). Additionally, you'll need to follow the appropriate [instructions](https://cds.climate.copernicus.eu/api-how-to) to set up the CDS API. Finally, you can download cdsapi via pip.
```commandline
pip install cdsapi
```
2. netcdf4 – Downloads from the CDS are typically in NetCDF4 format, so you'll need to install [netcdf4](https://github.com/Unidata/netcdf4-python), a package for interfacing with this file type.
```commandline
pip install netcdf4
```
3. noaastn – To interface with NOAA's ISD and select the correct set of historical weather data for your forecast, you'll need to install [noaastn](https://github.com/UBC-MDS/noaastn), a package that is currently available on the Test Python Package Index (TestPyPI). To do this using pip, you can run the following command.
```commandline
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple noaastn
```

All other dependencies will be handled automatically by ``peaktemp``. To install ``peaktemp`` using pip, copy and paste the following command. 
```
pip install peaktemp
```
You can also clone this repository and install from source if you want to make significant changes (i.e., use CMIP5 climate models, experiment with other forecasting techniques), but this is not recommended for most users.

## Usage

Currently, ``peaktemp`` supports three main uses: data downloading and processing from CMIP6 and the ISD, peak temperature forecasting, and climate scenario analysis. As a result, ``peaktemp`` comes with three modules: ``data``, ``forecast``, and ``scenario``.

To get started, you'll need to download the necessary CMIP6 data using the ``data`` module using ``data.download_cmip6_model()``:
```python
from peaktemp import data

# Downloading historical data is necessary for validation of model data and XGBoost forecasting
data.download_cmip6_model(
    model="cnrm_esm2_1",
    scenario="historical",
    location="Phoenix Sky Harbor",
    path="test_data"
)

# Next, download the climate scenario you have in mind -- SSP5-8.5 in this case
data.download_cmip6_model(
    model="cnrm_esm2_1",
    scenario="ssp5_8_5",
    location="Phoenix Sky Harbor",
    path="test_data"
)
```

The data module also allows you to download historical hourly temperature profiles from the ISD, a key step in the forecasting process:
```python
# Download ISD data for the last 30 years
isd_data = data.download_isd_profiles(
    location="Phoenix Sky Harbor", 
    start_year=1991, 
    end_year=2022
)

# Extract daily extremes for fitting hourly profiles
daily_extremes = data.get_isd_extremes(isd_data)
```
The final step in the data process involves extracting timeseries from the CMIP6 data – converting complex geospatial data into a single timeseries for forecasting.
```python
# Extract timeseries from the downloaded CMIP6 data 
extracted_models_5_8_5 = data.aggregate_cmip6_data(
    path="test_data",
    climate_models={"cnrm_esm2_1": "CNRM-ESM2-1"},
    scenario="ssp5_8_5",
    location="Phoenix Sky Harbor"
)
```
The example code above only uses data from one climate model, but it is recommended that you use multiple climate models in order to prevent unexpected behavior from a single model dominating your entire forecast. Once all data has been downloaded and processed, you can move on to forecasting. ``peaktemp``'s forecasting module contains 5 main functions, outlined below.
```python
from peaktemp import forecast

# Fit historical hourly profiles to modeled data
fitted_profiles = forecast.fit_multiple_profiles(
    isd_data=isd_data, 
    daily_extremes=daily_extremes, 
    climate_models=extracted_models
)

# Combine all modeled peaks into a single DataFrame
combined_peaks = forecast.combine_models(
    isd_data=isd_data, 
    fitted_models=fitted_profiles
)

# Create a 1-in-2 peak temperature forecast
forecasted_peaks = forecast.calculate_peaks(
    df=combined_peaks, 
    level=2
)

# Forecast peak temperatures using XGBoost (beta version)
ml_forecast = forecast.forecast_xgboost(
    isd_data=isd_data, 
    fitted_models=fitted_profiles, 
    show_importance=True
)

# Plot a 1-in-2, 1-in-10, 1-in-100, and XGBoost forecast against historical data
forecast.plot_full_forecast(
    isd_data=isd_data, 
    fitted_models=fitted_profiles
)
```

Finally, the ``scenario`` module allows users to analyze multiple climate scenarios and plot the 95% confidence intervals associated with each climate scenario. Assuming that data has been downloaded and extracted for at least two climate scenarios into ``extracted_models_1_2_6`` and ``extracted_models_5_8_5``, example usage is as follows.

```python
from peaktemp import scenario

# Conduct full scenario analysis (this step might take quite some time)
climate_scenarios = [extracted_models_1_2_6, extracted_models_5_8_5]
fitted_scenarios = scenario.create_scenarios(
    isd_data=isd_data, 
    daily_extremes=daily_extremes, 
    climate_scenarios=limate_scenarios
)

# Plot 95% CIs for each scenario
scenario.plot_scenarios(
    isd_data=isd_data,
    fitted_scenarios=fitted_scenarios,
    labels=["SSP1-2.6", "SSP5-8.5"],
    title="Example Scenario Analysis",
    path="test_scenario.png"
)
```

Although most of ``peaktemp``'s functionality is abstracted away from the user, you still might experience long runtimes, especially when it comes to working with CMIP6 data.

## Supporting Materials

To learn more about the research behind ``peaktemp`` and the context of the project at the Western Interstate Energy Board, please consult Jake Hofgard's and Evan Savage's [webinar](https://www.westernenergyboard.org/wieb-webinar-incorporating-temperature-and-precipitation-trends-in-long-term-planning/), entitled _Incorporating Temperature and Precipitation Trends in Long-Term Planning_. The webinar and supporting slide deck describe the creation of ``peaktemp`` in addition to policy recommendations that might help planners encourage the adoption of similar techniques for forecasting.

## Acknowledgments
Thank you to:
- Daniel Hua, John Ollis, John Fazio, and the rest of the team at the NWPCC for their advice on peak temperature forecasting and the in-depth presentation of their methodology in the Council's 2021 Power Plan
- The load forecasting and resource adequacy teams at PSE for their input and materials on the statistical side of peak temperature forecasting.
- Trenton Bush and Catalyst Cooperative for their input on data collection and processing.
- Evan Savage, Woori Lee, Maury Galbraith, and the rest of the team at WIEB for input, recommendations, and feedback throughout the development process!

## Contact

For inquiries and recommendations, please contact Jake Hofgard (jhofgard@westernenergyboard.org). If you notice an issue, please submit a pull request on GitHub.