"""
Full unit testing of all of peaktemp's user-facing functionality
"""

from peaktemp import data
from peaktemp import forecast
from peaktemp import scenario

# Test CMIP6 model data download

# Downloading historical data is necessary for XGBoost forecasting
data.download_cmip6_model(
    model="cnrm_esm2_1",
    scenario="historical",
    location="Phoenix Sky Harbor",
    path="test_data"
)

# Repeat for GFDL-ESM4
data.download_cmip6_model(model="gfdl_esm4", scenario="historical", location="Phoenix Sky Harbor", path="test_data")

# Downloading data for SS1-2.6 and SSP5-8.5, although almost all scenarios are available for CNRM-ESM2-1
data.download_cmip6_model(model="cnrm_esm2_1", scenario="ssp1_2_6", location="Phoenix Sky Harbor", path="test_data")
data.download_cmip6_model(model="cnrm_esm2_1", scenario="ssp5_8_5", location="Phoenix Sky Harbor", path="test_data")

# Repeat for GFDL-ESM4
data.download_cmip6_model(model="gfdl_esm4", scenario="ssp1_2_6", location="Phoenix Sky Harbor", path="test_data")
data.download_cmip6_model(model="gfdl_esm4", scenario="ssp5_8_5", location="Phoenix Sky Harbor", path="test_data")

# Test model data extraction for TWO CMIP6 models and TWO climate scenarios, downloaded above
extracted_models_1_2_6 = data.aggregate_cmip6_data(
    path="test_data",
    climate_models={"cnrm_esm2_1": "CNRM-ESM2-1", "gfdl_esm4": "GFDL-ESM4"},
    scenario="ssp1_2_6",
    location="Phoenix Sky Harbor"
)

extracted_models_5_8_5 = data.aggregate_cmip6_data(
    path="test_data",
    climate_models={"cnrm_esm2_1": "CNRM-ESM2-1", "gfdl_esm4": "GFDL-ESM4"},
    scenario="ssp5_8_5",
    location="Phoenix Sky Harbor"
)

# Test ISD data download for Phoenix
isd_data = data.download_isd_profiles(location="Phoenix Sky Harbor", start_year=1991, end_year=2022)

# Test extraction of daily extremes and differences
daily_extremes = data.get_isd_extremes(isd_data)

# Test profile fitting for multiple models
fitted_profiles = forecast.fit_multiple_profiles(isd_data, daily_extremes, extracted_models_5_8_5)

# Test combination of fitted profiles into a single DataFrame for forecasting
combined_peaks = forecast.combine_models(isd_data, fitted_profiles)

# Test 1-in-2 forecast using multiple climate models
forecasted_peaks = forecast.calculate_peaks(combined_peaks, level=2)

# Test XGBoost forecasting
ml_forecast = forecast.forecast_xgboost(isd_data, fitted_profiles, show_importance=True)

# Test plotting functionality for full forecast
forecast.plot_full_forecast(isd_data, fitted_profiles)

# Test scenario analysis
climate_scenarios = [extracted_models_1_2_6, extracted_models_5_8_5]
fitted_scenarios = scenario.create_scenarios(isd_data, daily_extremes, climate_scenarios)

# Test scenario plotting
scenario.plot_scenarios(
    isd_data,
    fitted_scenarios,
    labels=["SSP1-2.6", "SSP5-8.5"],
    title="Example Scenario Analysis",
    path="test_scenario.png"
)
