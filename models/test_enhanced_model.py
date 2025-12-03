"""
Test script for the enhanced economic simulation model
"""

import pandas as pd
import sys
import os

# Add the parent directory to the path so we can import the models module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.enhanced_economic_model import EnhancedEconomicSimulationModel, EconomicParameters

def test_enhanced_economic_model():
    """Test the enhanced economic simulation model"""
    print("Testing Enhanced Economic Simulation Model...")
    
    try:
        # Load actual data using our enhanced data processor
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from data_processing.enhanced_data_processor import EnhancedDataProcessor
        
        # Initialize the data processor
        processor = EnhancedDataProcessor()
        
        # Load and clean state data
        state_data = processor.load_and_clean_state_data()
        
        # Load brand data
        brand_data = processor.load_brand_data()
        
        print(f"\n1. Loaded data successfully:")
        print(f"   State data shape: {state_data.shape}")
        print(f"   Brand data shape: {brand_data.shape}")
        
        # Initialize the enhanced model
        model = EnhancedEconomicSimulationModel(state_data, brand_data)
        
        # Test market size calculation
        print("\n2. Testing market size calculation...")
        market_analysis = model.calculate_market_size()
        print(f"   Market analysis completed for scenario: {market_analysis['scenario']}")
        print(f"   Total annual packs: {market_analysis['total_annual_packs']:,.0f}")
        print(f"   Illegal market share: {market_analysis['avg_illegal_incidence_pct']:.1f}%")
        print(f"   Tax revenue loss: RM {market_analysis['tax_revenue_loss_rm']:,.0f}")
        
        # Test enforcement scenarios
        print("\n3. Testing enforcement scenarios...")
        enforcement_scenarios = [
            {
                'name': 'Current_Level',
                'intensity': 50,
                'effectiveness': 1.0
            },
            {
                'name': 'Increased_Enforcement',
                'intensity': 100,
                'effectiveness': 1.2
            },
            {
                'name': 'Targeted_Operations',
                'intensity': 75,
                'effectiveness': 2.0
            }
        ]
        
        simulation_results = model.simulate_enforcement_scenarios(enforcement_scenarios)
        print(f"   Simulated {len(simulation_results)} scenarios successfully")
        
        best_scenario = simulation_results.loc[simulation_results['roi_pct'].idxmax()]
        print(f"   Best ROI scenario: {best_scenario['scenario']} ({best_scenario['roi_pct']:,.0f}%)")
        
        # Test state-level projections
        print("\n4. Testing state-level projections...")
        state_projections = model.generate_state_level_projections()
        print(f"   Generated projections for {len(state_projections)} states")
        
        top_state = state_projections.loc[state_projections['priority_score'].idxmax()]
        print(f"   Highest priority state: {top_state['state']} (score: {top_state['priority_score']:.1f})")
        
        # Test parameter updates
        print("\n5. Testing parameter updates...")
        original_params = model.get_model_parameters()
        print(f"   Original illegal cigarette price: RM {original_params['avg_price_per_pack_illegal']}")
        
        # Update a parameter
        model.update_parameters({'avg_price_per_pack_illegal': 9.0})
        updated_params = model.get_model_parameters()
        print(f"   Updated illegal cigarette price: RM {updated_params['avg_price_per_pack_illegal']}")
        
        # Test sensitivity analysis
        print("\n6. Testing sensitivity analysis...")
        parameter_ranges = {
            'avg_price_per_pack_illegal': (6.0, 10.0),
            'tax_revenue_per_pack_legal': (10.0, 14.0)
        }
        
        # Run with fewer samples for testing
        sensitivity_results = model.perform_sensitivity_analysis(parameter_ranges, num_samples=20)
        print(f"   Sensitivity analysis completed with {len(sensitivity_results['roi_samples'])} samples")
        
        # Show some correlations
        if sensitivity_results['correlations']:
            print("   Sample correlations:")
            for param, correlation in list(sensitivity_results['correlations'].items())[:2]:
                print(f"     {param}: {correlation:.3f}")
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_enhanced_economic_model()
