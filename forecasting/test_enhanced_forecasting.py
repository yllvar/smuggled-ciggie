"""
Test script for the enhanced time series forecasting module
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the forecasting module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forecasting.enhanced_forecasting import EnhancedTimeSeriesForecasting, ForecastingParameters

def create_sample_historical_data():
    """Create sample historical data for testing"""
    # Create sample data similar to the actual data structure
    periods = [
        '2019', '2020', '2021', '2022', '2023', 
        'Sep_2023', 'Nov_2023', 'Jan_2024'
    ]
    
    # Sample incidence data for National level
    incidence_values = [55.3, 55.4, 55.6, 56.6, 57.3, 56.4, 55.3, 56.4]
    
    # Create date mapping
    date_mapping = {
        '2019': '2019-12-31',
        '2020': '2020-12-31',
        '2021': '2021-12-31',
        '2022': '2022-12-31',
        '2023': '2023-12-31',
        'Sep_2023': '2023-09-30',
        'Nov_2023': '2023-11-30',
        'Jan_2024': '2024-01-31'
    }
    
    data = []
    for period, value in zip(periods, incidence_values):
        if period in date_mapping:
            data.append({
                'period': period,
                'date': date_mapping[period],
                'illegal_incidence_percent': value
            })
    
    return pd.DataFrame(data)

def test_enhanced_forecasting():
    """Test the enhanced time series forecasting module"""
    print("Testing Enhanced Time Series Forecasting...")
    
    try:
        # Create sample historical data
        sample_data = create_sample_historical_data()
        print(f"\n1. Created sample data:")
        print(f"   Data shape: {sample_data.shape}")
        print(f"   Date range: {sample_data['date'].min()} to {sample_data['date'].max()}")
        
        # Initialize the enhanced forecasting model
        model = EnhancedTimeSeriesForecasting()
        
        # Prepare time series
        print("\n2. Testing time series preparation...")
        ts = model.prepare_time_series_data(sample_data, 'date', 'illegal_incidence_percent')
        print(f"   Time series prepared: {len(ts)} data points")
        print(f"   Value range: {ts.min():.1f}% to {ts.max():.1f}%")
        
        # Test stationarity
        print("\n3. Testing stationarity analysis...")
        stationarity_results = model.test_stationarity(ts)
        print(f"   Stationarity: {stationarity_results['interpretation']}")
        print(f"   P-value: {stationarity_results['p_value']:.4f}")
        
        # Decompose time series
        print("\n4. Testing time series decomposition...")
        decomposition = model.decompose_time_series(ts)
        print(f"   Decomposition completed")
        
        # Test individual forecasting models
        print("\n5. Testing individual forecasting models...")
        
        # ARIMA forecasting
        arima_forecast = model.forecast_with_arima(ts, steps=6)
        print(f"   ARIMA forecast: order {arima_forecast['order']}, AIC {arima_forecast['aic']:.2f}")
        
        # SARIMA forecasting
        sarima_forecast = model.forecast_with_sarima(ts, steps=6)
        print(f"   SARIMA forecast: AIC {sarima_forecast['aic']:.2f}")
        
        # Prophet forecasting
        prophet_forecast = model.forecast_with_prophet(ts, steps=6)
        print(f"   Prophet forecast: completed")
        
        # Create ensemble forecast
        print("\n6. Testing ensemble forecasting...")
        forecasts = {
            'arima': arima_forecast,
            'sarima': sarima_forecast,
            'prophet': prophet_forecast
        }
        
        ensemble_forecast = model.ensemble_forecast(forecasts)
        print(f"   Ensemble forecast completed")
        print(f"   Weights used: {ensemble_forecast['weights']}")
        
        # Show sample forecast values
        print("\n7. Sample forecast values (next 3 months):")
        for i, value in enumerate(ensemble_forecast['forecast'][:3]):
            print(f"   Month {i+1}: {value:.1f}%")
        
        # Test parameter updates
        print("\n8. Testing parameter updates...")
        original_params = model.get_model_parameters()
        print(f"   Original ARIMA max p: {original_params['arima_max_p']}")
        
        # Update a parameter
        model.update_parameters({'arima_max_p': 5})
        updated_params = model.get_model_parameters()
        print(f"   Updated ARIMA max p: {updated_params['arima_max_p']}")
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_enhanced_forecasting()
