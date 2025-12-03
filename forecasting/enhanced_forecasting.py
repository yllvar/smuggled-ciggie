"""
Enhanced Time Series Forecasting Module for Malaysia Illicit Cigarettes Study

This module provides improved time series forecasting models including enhanced ARIMA,
Prophet, and ensemble methods for better trend prediction.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ForecastingParameters:
    """Data class for forecasting parameters"""
    # ARIMA parameters
    arima_max_p: int = 3
    arima_max_d: int = 2
    arima_max_q: int = 3
    
    # SARIMA parameters
    seasonal_period: int = 12
    
    # Prophet parameters
    prophet_seasonality_mode: str = 'multiplicative'
    prophet_changepoint_prior_scale: float = 0.05
    
    # Ensemble parameters
    ensemble_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.ensemble_weights is None:
            self.ensemble_weights = {'arima': 0.4, 'prophet': 0.4, 'sarima': 0.2}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary"""
        return {
            'arima_max_p': self.arima_max_p,
            'arima_max_d': self.arima_max_d,
            'arima_max_q': self.arima_max_q,
            'seasonal_period': self.seasonal_period,
            'prophet_seasonality_mode': self.prophet_seasonality_mode,
            'prophet_changepoint_prior_scale': self.prophet_changepoint_prior_scale,
            'ensemble_weights': self.ensemble_weights
        }

class EnhancedTimeSeriesForecasting:
    """Enhanced time series forecasting with multiple models and ensemble methods"""
    
    def __init__(self, parameters: Optional[ForecastingParameters] = None):
        """
        Initialize the enhanced time series forecasting model
        
        Args:
            parameters: Forecasting parameters (uses defaults if None)
        """
        self.parameters = parameters or ForecastingParameters()
        
        logger.info("Enhanced Time Series Forecasting initialized successfully")
    
    def prepare_time_series_data(self, historical_data: pd.DataFrame, 
                               date_column: str, value_column: str) -> pd.Series:
        """
        Prepare time series data for forecasting
        
        Args:
            historical_data: DataFrame with historical data
            date_column: Name of the date column
            value_column: Name of the value column
            
        Returns:
            Time series as a pandas Series
        """
        try:
            # Create time series
            ts = historical_data.set_index(date_column)[value_column].sort_index()
            
            # Remove any NaN values
            ts = ts.dropna()
            
            # Ensure values are within reasonable bounds (0-100 for percentages)
            ts = np.clip(ts, 0, 100)
            
            logger.debug(f"Time series prepared: {len(ts)} data points from {ts.index.min()} to {ts.index.max()}")
            return ts
            
        except Exception as e:
            logger.error(f"Error preparing time series data: {str(e)}")
            raise
    
    def test_stationarity(self, ts: pd.Series) -> Dict[str, Any]:
        """
        Test stationarity of time series using Augmented Dickey-Fuller test
        
        Args:
            ts: Time series to test
            
        Returns:
            Dictionary with stationarity test results
        """
        try:
            # Perform ADF test
            adf_result = adfuller(ts.values)
            
            # Extract results
            adf_statistic = adf_result[0]
            p_value = adf_result[1]
            critical_values = adf_result[4]
            
            # Determine stationarity
            is_stationary = p_value < 0.05
            
            results = {
                'adf_statistic': adf_statistic,
                'p_value': p_value,
                'critical_values': critical_values,
                'is_stationary': is_stationary,
                'interpretation': 'Stationary' if is_stationary else 'Non-stationary'
            }
            
            logger.debug(f"Stationarity test: {results['interpretation']} (p-value: {p_value:.4f})")
            return results
            
        except Exception as e:
            logger.error(f"Error in stationarity test: {str(e)}")
            raise
    
    def decompose_time_series(self, ts: pd.Series) -> Dict[str, Any]:
        """
        Decompose time series into trend, seasonal, and residual components
        
        Args:
            ts: Time series to decompose
            
        Returns:
            Dictionary with decomposition results
        """
        try:
            # Perform seasonal decomposition
            decomposition = seasonal_decompose(ts, model='additive', period=min(4, len(ts)//2))
            
            results = {
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid,
                'observed': decomposition.observed
            }
            
            logger.debug(f"Time series decomposition completed with period={min(4, len(ts)//2)}")
            return results
            
        except Exception as e:
            logger.error(f"Error in time series decomposition: {str(e)}")
            raise
    
    def optimize_arima_parameters(self, ts: pd.Series) -> Tuple[int, int, int]:
        """
        Find optimal ARIMA parameters using AIC
        
        Args:
            ts: Time series to optimize for
            
        Returns:
            Tuple of (p, d, q) parameters
        """
        try:
            best_aic = float('inf')
            best_params = (0, 0, 0)
            
            # Test different parameter combinations
            for p in range(self.parameters.arima_max_p + 1):
                for d in range(self.parameters.arima_max_d + 1):
                    for q in range(self.parameters.arima_max_q + 1):
                        try:
                            model = ARIMA(ts, order=(p, d, q))
                            fitted_model = model.fit()
                            
                            if fitted_model.aic < best_aic:
                                best_aic = fitted_model.aic
                                best_params = (p, d, q)
                        except:
                            continue
            
            logger.debug(f"ARIMA parameter optimization: best params {best_params} with AIC {best_aic:.2f}")
            return best_params
            
        except Exception as e:
            logger.error(f"Error in ARIMA parameter optimization: {str(e)}")
            # Return default parameters if optimization fails
            return (1, 1, 1)
    
    def forecast_with_arima(self, ts: pd.Series, steps: int = 12) -> Dict[str, Any]:
        """
        Forecast using ARIMA model
        
        Args:
            ts: Time series to forecast
            steps: Number of steps to forecast
            
        Returns:
            Dictionary with forecast results
        """
        try:
            # Optimize parameters
            p, d, q = self.optimize_arima_parameters(ts)
            
            # Fit ARIMA model
            model = ARIMA(ts, order=(p, d, q))
            fitted_model = model.fit()
            
            # Generate forecasts
            forecast_result = fitted_model.get_forecast(steps=steps)
            forecast = forecast_result.predicted_mean
            conf_int = forecast_result.conf_int()
            
            # Clip forecasts to reasonable bounds
            forecast = np.clip(forecast, 0, 100)
            conf_int = np.clip(conf_int, 0, 100)
            
            results = {
                'model': 'ARIMA',
                'order': (p, d, q),
                'aic': fitted_model.aic,
                'forecast': forecast,
                'confidence_intervals': conf_int,
                'fitted_values': fitted_model.fittedvalues
            }
            
            logger.debug(f"ARIMA forecast completed: order {p,d,q}, AIC {fitted_model.aic:.2f}")
            return results
            
        except Exception as e:
            logger.error(f"Error in ARIMA forecasting: {str(e)}")
            raise
    
    def forecast_with_sarima(self, ts: pd.Series, steps: int = 12) -> Dict[str, Any]:
        """
        Forecast using SARIMA model
        
        Args:
            ts: Time series to forecast
            steps: Number of steps to forecast
            
        Returns:
            Dictionary with forecast results
        """
        try:
            # Optimize parameters
            p, d, q = self.optimize_arima_parameters(ts)
            
            # Fit SARIMA model
            seasonal_order = (1, 1, 1, self.parameters.seasonal_period)
            model = SARIMAX(ts, order=(p, d, q), seasonal_order=seasonal_order)
            fitted_model = model.fit(disp=False, warn_convergence=False)
            
            # Generate forecasts
            forecast_result = fitted_model.get_forecast(steps=steps)
            forecast = forecast_result.predicted_mean
            conf_int = forecast_result.conf_int()
            
            # Clip forecasts to reasonable bounds
            forecast = np.clip(forecast, 0, 100)
            conf_int = np.clip(conf_int, 0, 100)
            
            results = {
                'model': 'SARIMA',
                'order': (p, d, q),
                'seasonal_order': seasonal_order,
                'aic': fitted_model.aic,
                'forecast': forecast,
                'confidence_intervals': conf_int,
                'fitted_values': fitted_model.fittedvalues
            }
            
            logger.debug(f"SARIMA forecast completed: order {p,d,q}, seasonal {seasonal_order}, AIC {fitted_model.aic:.2f}")
            return results
            
        except Exception as e:
            logger.warning(f"SARIMA forecasting failed, falling back to ARIMA: {str(e)}")
            # Fallback to ARIMA if SARIMA fails
            return self.forecast_with_arima(ts, steps)
    
    def forecast_with_prophet(self, ts: pd.Series, steps: int = 12) -> Dict[str, Any]:
        """
        Forecast using Prophet model with fallback mechanisms
        
        Args:
            ts: Time series to forecast
            steps: Number of steps to forecast
            
        Returns:
            Dictionary with forecast results
        """
        try:
            # Prepare data for Prophet
            df = pd.DataFrame({
                'ds': ts.index,
                'y': ts.values
            })
            
            # Check if we have enough data points
            if len(df) < 2:
                raise ValueError("Prophet requires at least 2 data points")
            
            # Initialize Prophet model with more conservative settings
            model = Prophet(
                seasonality_mode=self.parameters.prophet_seasonality_mode,
                changepoint_prior_scale=min(0.01, self.parameters.prophet_changepoint_prior_scale),
                yearly_seasonality=min(2, len(df) // 2),
                weekly_seasonality=False,
                daily_seasonality=False,
                mcmc_samples=0,  # Disable MCMC for faster computation
                interval_width=0.8  # Reduce confidence interval width
            )
            
            # Fit model with error handling
            try:
                model.fit(df)
            except Exception as fit_error:
                logger.warning(f"Prophet fitting failed, trying with simpler model: {str(fit_error)}")
                # Try with even simpler model
                model = Prophet(
                    seasonality_mode='additive',
                    changepoint_prior_scale=0.001,
                    yearly_seasonality=False,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    mcmc_samples=0
                )
                model.fit(df)
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=steps, freq='M')
            
            # Generate forecasts
            forecast = model.predict(future)
            
            # Extract forecast components
            forecast_values = forecast['yhat'].tail(steps).values
            lower_ci = forecast['yhat_lower'].tail(steps).values
            upper_ci = forecast['yhat_upper'].tail(steps).values
            
            # Clip forecasts to reasonable bounds
            forecast_values = np.clip(forecast_values, 0, 100)
            lower_ci = np.clip(lower_ci, 0, 100)
            upper_ci = np.clip(upper_ci, 0, 100)
            
            results = {
                'model': 'Prophet',
                'forecast': forecast_values,
                'confidence_intervals': np.column_stack([lower_ci, upper_ci]),
                'fitted_values': forecast['yhat'].head(len(ts)).values
            }
            
            # Add trend and seasonal if available
            if 'trend' in forecast.columns:
                results['trend'] = forecast['trend'].tail(steps).values
            if 'seasonal' in forecast.columns:
                results['seasonal'] = forecast['seasonal'].tail(steps).values
            
            logger.debug(f"Prophet forecast completed")
            return results
            
        except Exception as e:
            logger.error(f"Error in Prophet forecasting: {str(e)}")
            # Create a simple fallback forecast using last value
            last_value = ts.iloc[-1] if len(ts) > 0 else 50.0
            fallback_forecast = np.full(steps, last_value)
            fallback_ci = np.column_stack([np.full(steps, last_value), np.full(steps, last_value)])
            
            results = {
                'model': 'Prophet-Fallback',
                'forecast': fallback_forecast,
                'confidence_intervals': fallback_ci,
                'fitted_values': np.full(len(ts), last_value) if len(ts) > 0 else np.array([])
            }
            
            logger.warning(f"Prophet failed, using fallback forecast with last value: {last_value}")
            return results
    
    def ensemble_forecast(self, forecasts: Dict[str, Dict[str, Any]], 
                         weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Create ensemble forecast from multiple models
        
        Args:
            forecasts: Dictionary of forecast results from different models
            weights: Weights for each model (uses default if None)
            
        Returns:
            Dictionary with ensemble forecast results
        """
        try:
            if weights is None:
                weights = self.parameters.ensemble_weights
            
            # Get forecast length (assuming all forecasts have same length)
            forecast_length = len(list(forecasts.values())[0]['forecast'])
            
            # Initialize ensemble forecast
            ensemble_forecast = np.zeros(forecast_length)
            
            # Combine forecasts using weights
            for model_name, forecast_result in forecasts.items():
                if model_name in weights:
                    weight = weights[model_name]
                    model_forecast = forecast_result['forecast']
                    ensemble_forecast += weight * model_forecast
            
            # Create ensemble results
            results = {
                'model': 'Ensemble',
                'forecast': ensemble_forecast,
                'weights': weights,
                'individual_forecasts': forecasts
            }
            
            logger.debug(f"Ensemble forecast completed with weights: {weights}")
            return results
            
        except Exception as e:
            logger.error(f"Error in ensemble forecasting: {str(e)}")
            raise
    
    def evaluate_forecast_accuracy(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """
        Evaluate forecast accuracy using multiple metrics
        
        Args:
            actual: Actual values
            predicted: Predicted values
            
        Returns:
            Dictionary with accuracy metrics
        """
        try:
            # Calculate metrics
            mae = mean_absolute_error(actual, predicted)
            mse = mean_squared_error(actual, predicted)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100 if np.all(actual != 0) else float('inf')
            
            results = {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'mape': mape
            }
            
            logger.debug(f"Forecast accuracy evaluation: MAE={mae:.2f}, RMSE={rmse:.2f}")
            return results
            
        except Exception as e:
            logger.error(f"Error in forecast accuracy evaluation: {str(e)}")
            raise
    
    def get_model_parameters(self) -> Dict[str, Any]:
        """
        Get current model parameters
        
        Returns:
            Dictionary with current model parameters
        """
        return self.parameters.to_dict()
    
    def update_parameters(self, new_parameters: Dict[str, Any]):
        """
        Update model parameters
        
        Args:
            new_parameters: Dictionary with new parameter values
        """
        try:
            # Update parameters
            for key, value in new_parameters.items():
                if hasattr(self.parameters, key):
                    setattr(self.parameters, key, value)
                    logger.debug(f"Updated parameter '{key}' to {value}")
                else:
                    logger.warning(f"Unknown parameter '{key}' ignored")
            
            logger.info("Model parameters updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating parameters: {str(e)}")
            raise

def main():
    """Main function to demonstrate the enhanced time series forecasting"""
    try:
        # Create sample data for demonstration
        dates = pd.date_range(start='2020-01-01', periods=20, freq='M')
        values = 50 + 5 * np.sin(np.arange(20) * 2 * np.pi / 12) + np.random.normal(0, 2, 20)
        values = np.clip(values, 0, 100)  # Keep within 0-100 range
        
        sample_data = pd.DataFrame({
            'date': dates,
            'value': values
        })
        
        # Initialize the model
        model = EnhancedTimeSeriesForecasting()
        
        # Prepare time series
        ts = model.prepare_time_series_data(sample_data, 'date', 'value')
        print(f"Time series prepared: {len(ts)} data points")
        
        # Test stationarity
        stationarity_results = model.test_stationarity(ts)
        print(f"Stationarity test: {stationarity_results['interpretation']}")
        
        # Decompose time series
        decomposition = model.decompose_time_series(ts)
        print("Time series decomposition completed")
        
        # Forecast with different models
        print("\nGenerating forecasts...")
        
        arima_forecast = model.forecast_with_arima(ts, steps=6)
        print(f"ARIMA forecast completed: order {arima_forecast['order']}")
        
        sarima_forecast = model.forecast_with_sarima(ts, steps=6)
        print(f"SARIMA forecast completed: AIC {sarima_forecast['aic']:.2f}")
        
        prophet_forecast = model.forecast_with_prophet(ts, steps=6)
        print(f"Prophet forecast completed")
        
        # Create ensemble forecast
        forecasts = {
            'arima': arima_forecast,
            'sarima': sarima_forecast,
            'prophet': prophet_forecast
        }
        
        ensemble_forecast = model.ensemble_forecast(forecasts)
        print(f"Ensemble forecast completed with {len(ensemble_forecast['forecast'])} steps")
        
        # Show sample forecast values
        print("\nSample forecast values (next 3 months):")
        for i, value in enumerate(ensemble_forecast['forecast'][:3]):
            print(f"  Month {i+1}: {value:.1f}%")
        
        print("\n✅ Enhanced Time Series Forecasting demonstration completed!")
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    main()
