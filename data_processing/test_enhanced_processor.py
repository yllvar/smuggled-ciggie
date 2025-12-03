"""
Test script for the enhanced data processor
"""

import pandas as pd
import sys
import os

# Add the parent directory to the path so we can import the data_processing module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing.enhanced_data_processor import EnhancedDataProcessor

def test_enhanced_data_processor():
    """Test the enhanced data processor"""
    print("Testing Enhanced Data Processor...")
    
    try:
        # Initialize the processor
        processor = EnhancedDataProcessor()
        
        # Test loading and cleaning state data
        print("\n1. Testing state data loading...")
        state_data = processor.load_and_clean_state_data()
        print(f"   State data loaded successfully: {state_data.shape}")
        print(f"   Columns: {list(state_data.columns)}")
        
        # Show first few rows
        print("\n   First 5 rows of state data:")
        print(state_data.head())
        
        # Test loading brand data
        print("\n2. Testing brand data loading...")
        brand_data = processor.load_brand_data()
        print(f"   Brand data loaded successfully: {brand_data.shape}")
        print(f"   Columns: {list(brand_data.columns)}")
        
        # Show first few rows
        print("\n   First 5 rows of brand data:")
        print(brand_data.head())
        
        # Test data validation
        print("\n3. Testing data validation...")
        validation_results = processor.validate_data_integrity(state_data, brand_data)
        for key, value in validation_results.items():
            print(f"   {key}: {value}")
        
        # Test saving processed data
        print("\n4. Testing data saving...")
        state_file = processor.save_processed_data(state_data, 'test_enhanced_state_data.csv')
        brand_file = processor.save_processed_data(brand_data, 'test_enhanced_brand_data.csv')
        print(f"   State data saved to: {state_file}")
        print(f"   Brand data saved to: {brand_file}")
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_enhanced_data_processor()
