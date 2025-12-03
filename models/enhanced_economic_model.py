"""
Enhanced Economic Simulation Model for Malaysia Illicit Cigarettes Study

This module provides an enhanced economic simulation model with more granular parameters,
sensitivity analysis, and improved state-level modeling compared to the original implementation.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import stats
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EconomicParameters:
    """Data class for economic parameters with default values based on Malaysian market context"""
    # Price parameters
    avg_price_per_pack_legal: float = 17.0  # RM per pack (average legal price)
    avg_price_per_pack_illegal: float = 8.0  # RM per pack (average illegal price)
    
    # Tax parameters
    tax_revenue_per_pack_legal: float = 12.0  # RM tax per legal pack
    
    # Enforcement parameters
    enforcement_cost_per_operation: float = 50000  # RM per enforcement operation
    base_seizure_rate_per_operation: float = 0.15  # Base seizure rate per operation
    
    # Market parameters
    total_smokers_malaysia: int = 5_000_000  # Estimated total smokers
    avg_consumption_per_smoker: int = 15  # sticks per day
    
    # Sensitivity analysis parameters
    price_sensitivity: float = 0.3  # Elasticity of demand to price changes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary"""
        return {
            'avg_price_per_pack_legal': self.avg_price_per_pack_legal,
            'avg_price_per_pack_illegal': self.avg_price_per_pack_illegal,
            'tax_revenue_per_pack_legal': self.tax_revenue_per_pack_legal,
            'enforcement_cost_per_operation': self.enforcement_cost_per_operation,
            'base_seizure_rate_per_operation': self.base_seizure_rate_per_operation,
            'total_smokers_malaysia': self.total_smokers_malaysia,
            'avg_consumption_per_smoker': self.avg_consumption_per_smoker,
            'price_sensitivity': self.price_sensitivity
        }

class EnhancedEconomicSimulationModel:
    """Enhanced economic simulation model with sensitivity analysis and granular parameters"""
    
    def __init__(self, state_data: pd.DataFrame, brand_data: pd.DataFrame, 
                 parameters: Optional[EconomicParameters] = None):
        """
        Initialize the enhanced economic simulation model
        
        Args:
            state_data: DataFrame with state-level incidence data
            brand_data: DataFrame with brand market share data
            parameters: Economic parameters (uses defaults if None)
        """
        self.state_data = state_data.copy()
        self.brand_data = brand_data.copy() if not brand_data.empty else pd.DataFrame()
        self.parameters = parameters or EconomicParameters()
        
        # Validate input data
        self._validate_data()
        
        # Calculate derived parameters
        self._calculate_derived_parameters()
        
        logger.info("Enhanced Economic Simulation Model initialized successfully")
    
    def _validate_data(self):
        """Validate input data"""
        required_state_columns = [
            'State', 'Incidence of illegal cigarettes', 'Incidence of legal cigarettes'
        ]
        
        for col in required_state_columns:
            if col not in self.state_data.columns:
                raise ValueError(f"Required column '{col}' not found in state data")
        
        if not self.state_data['State'].is_unique:
            logger.warning("Duplicate state names found in data")
        
        # Validate incidence values are reasonable
        illegal_incidence = self.state_data['Incidence of illegal cigarettes']
        if not ((illegal_incidence >= 0) & (illegal_incidence <= 100)).all():
            raise ValueError("Illegal incidence values must be between 0 and 100")
        
        logger.info("Data validation completed successfully")
    
    def _calculate_derived_parameters(self):
        """Calculate derived parameters"""
        # Calculate total annual consumption
        total_daily_sticks = self.parameters.total_smokers_malaysia * self.parameters.avg_consumption_per_smoker
        self.total_annual_sticks = total_daily_sticks * 365
        self.total_annual_packs = self.total_annual_sticks / 20  # 20 sticks per pack
        
        # Calculate average illegal incidence
        self.avg_illegal_incidence = self.state_data['Incidence of illegal cigarettes'].mean() / 100
        
        logger.debug(f"Derived parameters calculated: total_annual_packs={self.total_annual_packs:,}, "
                    f"avg_illegal_incidence={self.avg_illegal_incidence:.3f}")
    
    def calculate_market_size(self, scenario: str = 'current') -> Dict[str, float]:
        """
        Calculate total market size and economic impact with scenario support
        
        Args:
            scenario: Scenario type ('current', 'post_enforcement', 'sensitivity')
            
        Returns:
            Dictionary with market analysis results
        """
        try:
            # Calculate market segments
            legal_packs = self.total_annual_packs * (1 - self.avg_illegal_incidence)
            illegal_packs = self.total_annual_packs * self.avg_illegal_incidence
            
            # Calculate economic values
            legal_market_value = legal_packs * self.parameters.avg_price_per_pack_legal
            illegal_market_value = illegal_packs * self.parameters.avg_price_per_pack_illegal
            tax_revenue_loss = illegal_packs * self.parameters.tax_revenue_per_pack_legal
            
            # Apply scenario adjustments if needed
            if scenario == 'post_enforcement':
                # Placeholder for post-enforcement adjustments
                pass
            elif scenario == 'sensitivity':
                # Placeholder for sensitivity analysis adjustments
                pass
            
            market_analysis = {
                'total_annual_packs': self.total_annual_packs,
                'legal_packs': legal_packs,
                'illegal_packs': illegal_packs,
                'legal_market_value_rm': legal_market_value,
                'illegal_market_value_rm': illegal_market_value,
                'tax_revenue_loss_rm': tax_revenue_loss,
                'avg_illegal_incidence_pct': self.avg_illegal_incidence * 100,
                'scenario': scenario
            }
            
            logger.debug(f"Market size calculated for scenario '{scenario}': {market_analysis}")
            return market_analysis
            
        except Exception as e:
            logger.error(f"Error calculating market size: {str(e)}")
            raise
    
    def simulate_enforcement_scenarios(self, scenarios: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Simulate different enforcement scenarios and ROI with enhanced modeling
        
        Args:
            scenarios: List of scenario dictionaries with name, intensity, and effectiveness
            
        Returns:
            DataFrame with simulation results
        """
        try:
            results = []
            
            for scenario in scenarios:
                scenario_name = scenario['name']
                enforcement_intensity = scenario['intensity']  # Number of operations
                effectiveness = scenario.get('effectiveness', 1.0)  # Multiplier for seizure rate
                budget_multiplier = scenario.get('budget_multiplier', 1.0)  # Budget adjustment
                
                logger.info(f"Simulating scenario: {scenario_name} with intensity {enforcement_intensity}")
                
                # Calculate enforcement costs with budget multiplier
                total_enforcement_cost = (enforcement_intensity * 
                                        self.parameters.enforcement_cost_per_operation * 
                                        budget_multiplier)
                
                # Calculate effective seizure rate
                effective_seizure_rate = (self.parameters.base_seizure_rate_per_operation * 
                                        effectiveness)
                
                # Calculate seizures by state
                total_seized_packs = 0
                state_seizures = {}
                
                for _, state in self.state_data.iterrows():
                    state_name = state['State']
                    illegal_incidence = state['Incidence of illegal cigarettes'] / 100
                    
                    # Calculate state market size
                    state_market_share = 1 / len(self.state_data)  # Equal distribution assumption
                    state_total_packs = self.total_annual_packs * state_market_share
                    state_illegal_packs = state_total_packs * illegal_incidence
                    
                    # Calculate state seizures
                    state_seizures[state_name] = (state_illegal_packs * 
                                                effective_seizure_rate * 
                                                enforcement_intensity)
                    total_seized_packs += state_seizures[state_name]
                
                # Calculate economic impact
                seized_value = total_seized_packs * self.parameters.avg_price_per_pack_illegal
                recovered_tax = total_seized_packs * self.parameters.tax_revenue_per_pack_legal
                
                # Calculate ROI
                total_benefits = seized_value + recovered_tax
                roi = ((total_benefits - total_enforcement_cost) / 
                      total_enforcement_cost * 100) if total_enforcement_cost > 0 else 0
                
                # Calculate market impact
                market_reduction_pct = ((total_seized_packs / 
                                       (self.total_annual_packs * self.avg_illegal_incidence)) * 
                                      100) if (self.total_annual_packs * self.avg_illegal_incidence) > 0 else 0
                
                # Calculate cost per seizure
                cost_per_seizure = total_enforcement_cost / total_seized_packs if total_seized_packs > 0 else 0
                
                results.append({
                    'scenario': scenario_name,
                    'enforcement_operations': enforcement_intensity,
                    'total_cost_rm': total_enforcement_cost,
                    'seized_packs': total_seized_packs,
                    'seized_value_rm': seized_value,
                    'recovered_tax_rm': recovered_tax,
                    'total_benefits_rm': total_benefits,
                    'roi_pct': roi,
                    'market_reduction_pct': market_reduction_pct,
                    'cost_per_seizure_rm': cost_per_seizure,
                    'effectiveness_multiplier': effectiveness,
                    'budget_multiplier': budget_multiplier
                })
                
                logger.debug(f"Scenario {scenario_name} completed: ROI={roi:.2f}%")
            
            result_df = pd.DataFrame(results)
            logger.info(f"Enforcement scenarios simulated successfully: {len(results)} scenarios")
            return result_df
            
        except Exception as e:
            logger.error(f"Error simulating enforcement scenarios: {str(e)}")
            raise
    
    def generate_state_level_projections(self) -> pd.DataFrame:
        """
        Generate enhanced state-level economic impact projections
        
        Returns:
            DataFrame with state-level projections
        """
        try:
            state_projections = []
            
            for _, state in self.state_data.iterrows():
                state_name = state['State']
                illegal_incidence = state['Incidence of illegal cigarettes'] / 100
                legal_incidence = state['Incidence of legal cigarettes'] / 100
                
                # Calculate state market size (proportional to population)
                state_market_share = 1 / len(self.state_data)  # Simplified equal distribution
                state_total_packs = self.total_annual_packs * state_market_share
                state_illegal_packs = state_total_packs * illegal_incidence
                state_legal_packs = state_total_packs * legal_incidence
                
                # Economic impact
                state_tax_loss = state_illegal_packs * self.parameters.tax_revenue_per_pack_legal
                state_illegal_market = state_illegal_packs * self.parameters.avg_price_per_pack_illegal
                state_legal_market = state_legal_packs * self.parameters.avg_price_per_pack_legal
                
                # Enforcement recommendations with enhanced logic
                # Base recommendations on illegal incidence and market size
                recommended_operations = max(1, int(illegal_incidence * 10))
                
                # Adjust for market size (larger markets need more operations)
                market_size_factor = state_total_packs / (self.total_annual_packs / len(self.state_data))
                adjusted_operations = max(1, int(recommended_operations * market_size_factor))
                
                # Projected seizures
                projected_seizures = (state_illegal_packs * 
                                    self.parameters.base_seizure_rate_per_operation * 
                                    adjusted_operations)
                
                # Projected benefits
                projected_benefits = projected_seizures * (self.parameters.avg_price_per_pack_illegal + 
                                                         self.parameters.tax_revenue_per_pack_legal)
                
                # Enhanced priority scoring (considering both incidence and market size)
                priority_score = illegal_incidence * 100 * market_size_factor
                
                # Calculate potential market shift (if illegal market is reduced)
                potential_legal_increase = projected_seizures * 0.3  # Assume 30% shift to legal
                
                state_projections.append({
                    'state': state_name,
                    'illegal_incidence_pct': illegal_incidence * 100,
                    'legal_incidence_pct': legal_incidence * 100,
                    'estimated_illegal_packs_annual': state_illegal_packs,
                    'estimated_legal_packs_annual': state_legal_packs,
                    'tax_revenue_loss_rm': state_tax_loss,
                    'illegal_market_value_rm': state_illegal_market,
                    'legal_market_value_rm': state_legal_market,
                    'recommended_enforcement_ops': adjusted_operations,
                    'projected_annual_seizures': projected_seizures,
                    'projected_benefits_rm': projected_benefits,
                    'priority_score': priority_score,
                    'potential_legal_market_increase': potential_legal_increase,
                    'market_size_factor': market_size_factor
                })
            
            result_df = pd.DataFrame(state_projections)
            logger.info(f"State-level projections generated successfully: {len(state_projections)} states")
            return result_df
            
        except Exception as e:
            logger.error(f"Error generating state-level projections: {str(e)}")
            raise
    
    def perform_sensitivity_analysis(self, parameter_ranges: Dict[str, Tuple[float, float]], 
                                   num_samples: int = 100) -> Dict[str, Any]:
        """
        Perform sensitivity analysis on key economic parameters
        
        Args:
            parameter_ranges: Dictionary with parameter names and (min, max) tuples
            num_samples: Number of samples for Monte Carlo simulation
            
        Returns:
            Dictionary with sensitivity analysis results
        """
        try:
            logger.info(f"Performing sensitivity analysis with {num_samples} samples")
            
            # Store results
            sensitivity_results = {
                'parameter_samples': {},
                'roi_samples': [],
                'tax_recovery_samples': [],
                'correlations': {}
            }
            
            # Generate random samples for each parameter
            np.random.seed(42)  # For reproducibility
            
            for param_name, (min_val, max_val) in parameter_ranges.items():
                samples = np.random.uniform(min_val, max_val, num_samples)
                sensitivity_results['parameter_samples'][param_name] = samples
                
                logger.debug(f"Generated {len(samples)} samples for parameter '{param_name}'")
            
            # Run simulations for each sample
            for i in range(num_samples):
                # Create temporary parameters with sampled values
                temp_params = EconomicParameters(**self.parameters.to_dict())
                
                for param_name, samples in sensitivity_results['parameter_samples'].items():
                    if hasattr(temp_params, param_name):
                        setattr(temp_params, param_name, samples[i])
                
                # Create temporary model with sampled parameters
                temp_model = EnhancedEconomicSimulationModel(
                    self.state_data, self.brand_data, temp_params)
                
                # Run a basic scenario
                scenarios = [{
                    'name': 'Sample_Scenario',
                    'intensity': 75,  # Targeted operations intensity
                    'effectiveness': 2.0
                }]
                
                results = temp_model.simulate_enforcement_scenarios(scenarios)
                
                if not results.empty:
                    sensitivity_results['roi_samples'].append(results.iloc[0]['roi_pct'])
                    sensitivity_results['tax_recovery_samples'].append(
                        results.iloc[0]['recovered_tax_rm'])
                else:
                    sensitivity_results['roi_samples'].append(0)
                    sensitivity_results['tax_recovery_samples'].append(0)
            
            # Calculate correlations
            roi_samples = np.array(sensitivity_results['roi_samples'])
            
            for param_name, param_samples in sensitivity_results['parameter_samples'].items():
                if len(param_samples) == len(roi_samples):
                    correlation = np.corrcoef(param_samples, roi_samples)[0, 1]
                    sensitivity_results['correlations'][param_name] = correlation
                    
                    logger.debug(f"Correlation between '{param_name}' and ROI: {correlation:.3f}")
            
            logger.info("Sensitivity analysis completed successfully")
            return sensitivity_results
            
        except Exception as e:
            logger.error(f"Error performing sensitivity analysis: {str(e)}")
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
            
            # Recalculate derived parameters
            self._calculate_derived_parameters()
            
            logger.info("Model parameters updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating parameters: {str(e)}")
            raise

def main():
    """Main function to demonstrate the enhanced economic simulation model"""
    try:
        # This would normally load actual data
        # For demonstration, we'll create sample data
        
        # Sample state data
        state_data = pd.DataFrame({
            'State': ['PERLIS', 'KEDAH', 'PENANG'],
            'Incidence of illegal cigarettes': [41.8, 32.4, 47.8],
            'Incidence of legal cigarettes': [58.2, 67.6, 52.2]
        })
        
        # Sample brand data
        brand_data = pd.DataFrame({
            'Brand': ['John', 'Bosston', 'Misto'],
            'Market_Share_Jan2024': [16.3, 13.3, 7.4]
        })
        
        # Initialize the model
        model = EnhancedEconomicSimulationModel(state_data, brand_data)
        
        # Calculate market size
        market_analysis = model.calculate_market_size()
        print("Market Analysis:")
        for key, value in market_analysis.items():
            if isinstance(value, float):
                print(f"  {key}: {value:,.2f}")
            else:
                print(f"  {key}: {value}")
        
        # Define enforcement scenarios
        enforcement_scenarios = [
            {
                'name': 'Current_Level',
                'intensity': 50,
                'effectiveness': 1.0
            },
            {
                'name': 'Targeted_Operations',
                'intensity': 75,
                'effectiveness': 2.0
            }
        ]
        
        # Run enforcement simulations
        simulation_results = model.simulate_enforcement_scenarios(enforcement_scenarios)
        print("\nEnforcement Scenario Analysis:")
        print(simulation_results.to_string(index=False))
        
        # Generate state-level projections
        state_projections = model.generate_state_level_projections()
        print("\nState-Level Projections:")
        print(state_projections[['state', 'illegal_incidence_pct', 'tax_revenue_loss_rm', 'priority_score']].to_string(index=False))
        
        # Perform sensitivity analysis (simplified example)
        parameter_ranges = {
            'avg_price_per_pack_illegal': (6.0, 10.0),
            'tax_revenue_per_pack_legal': (10.0, 14.0)
        }
        
        sensitivity_results = model.perform_sensitivity_analysis(parameter_ranges, num_samples=10)
        print("\nSensitivity Analysis (sample results):")
        for param, correlation in list(sensitivity_results['correlations'].items())[:2]:
            print(f"  {param} correlation with ROI: {correlation:.3f}")
        
        print("\n✅ Enhanced Economic Simulation Model demonstration completed!")
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    main()
