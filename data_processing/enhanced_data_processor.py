"""
Enhanced Data Processing Module for Malaysia Illicit Cigarettes Study

This module provides improved data processing capabilities with better error handling,
validation, and more robust data cleaning compared to the original implementation.
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedDataProcessor:
    """Enhanced data processor with improved error handling and validation"""
    
    def __init__(self, data_dir: str = 'data'):
        """
        Initialize the data processor
        
        Args:
            data_dir: Path to the data directory
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        
        # Create processed directory if it doesn't exist
        self.processed_dir.mkdir(exist_ok=True)
        
        # Validate data directory structure
        if not self.raw_dir.exists():
            raise FileNotFoundError(f"Raw data directory not found: {self.raw_dir}")
    
    def load_and_clean_state_data(self, filename: str = 'page_58_table_1.csv') -> pd.DataFrame:
        """
        Load and clean the state-level incidence data with enhanced error handling
        
        Args:
            filename: Name of the CSV file containing state data
            
        Returns:
            Cleaned DataFrame with state-level data
            
        Raises:
            FileNotFoundError: If the data file is not found
            ValueError: If the data format is unexpected
        """
        try:
            file_path = self.raw_dir / filename
            if not file_path.exists():
                raise FileNotFoundError(f"State data file not found: {file_path}")
            
            logger.info(f"Loading state data from {file_path}")
            
            # Load the data
            state_df = pd.read_csv(file_path)
            
            # Validate basic structure
            if state_df.empty:
                raise ValueError("State data file is empty")
            
            # Clean column names
            state_df.columns = state_df.columns.str.replace('\n', ' ').str.strip()
            
            # Log original columns for debugging
            logger.debug(f"Original columns: {list(state_df.columns)}")
            
            # Remove header rows and clean data
            state_df = state_df.dropna(subset=['State'])
            state_df = state_df[~state_df['State'].isin(['A', 'B', 'C', 'D', 'E'])]
            
            # Validate that we have data after cleaning
            if state_df.empty:
                raise ValueError("No valid state data found after cleaning")
            
            # Clean numeric columns
            numeric_cols = ['Total packs collected (Jan\'24)', 'Number of legal packs collected', 
                           'Number of illegal packs collected']
            
            for col in numeric_cols:
                if col in state_df.columns:
                    # Store original values for logging
                    original_count = len(state_df)
                    original_non_null = state_df[col].notna().sum()
                    
                    # Convert to string, remove commas, then to float
                    state_df[col] = state_df[col].astype(str).str.replace(',', '').astype(float)
                    
                    logger.debug(f"Cleaned column '{col}': {original_non_null}/{original_count} non-null values")
                else:
                    logger.warning(f"Expected column '{col}' not found in data")
            
            # Clean percentage columns
            percentage_cols = ['Incidence of legal cigarettes', 'Incidence of illegal cigarettes']
            for col in percentage_cols:
                if col in state_df.columns:
                    # Store original values for logging
                    original_count = len(state_df)
                    original_non_null = state_df[col].notna().sum()
                    
                    # Remove % symbol and convert to float
                    state_df[col] = state_df[col].astype(str).str.replace('%', '').astype(float)
                    
                    logger.debug(f"Cleaned column '{col}': {original_non_null}/{original_count} non-null values")
                else:
                    logger.warning(f"Expected column '{col}' not found in data")
            
            # Validate state names
            valid_states = self._get_valid_malaysian_states()
            state_df = state_df[state_df['State'].isin(valid_states)]
            
            logger.info(f"Successfully loaded and cleaned state data: {state_df.shape}")
            return state_df
            
        except Exception as e:
            logger.error(f"Error loading state data: {str(e)}")
            raise
    
    def load_brand_data(self, filenames: List[str] = None) -> pd.DataFrame:
        """
        Load and clean brand market share data with enhanced error handling
        
        Args:
            filenames: List of CSV files containing brand data
            
        Returns:
            Cleaned DataFrame with brand market share data
        """
        if filenames is None:
            filenames = ['page_22_table_1.csv', 'page_19_table_1.csv']
        
        brand_data = []
        
        for filename in filenames:
            try:
                file_path = self.raw_dir / filename
                if not file_path.exists():
                    logger.warning(f"Brand data file not found: {file_path}")
                    continue
                
                logger.info(f"Processing brand data from {file_path}")
                
                df = pd.read_csv(file_path)
                
                # Validate basic structure
                if df.empty:
                    logger.warning(f"Brand data file is empty: {file_path}")
                    continue
                
                # Clean and extract brand information
                # Find the column that contains brand names (usually 'Unnamed: 1')
                brand_col = None
                for col in df.columns:
                    if 'brand' in col.lower() or 'illegal' in col.lower():
                        brand_col = col
                        break
                
                if brand_col is None and len(df.columns) > 1:
                    # Use the second column as brand column
                    brand_col = df.columns[1]
                
                if brand_col is None:
                    logger.warning(f"Could not identify brand column in {file_path}")
                    continue
                
                # Find the column that contains percentage data (look for date-like columns)
                value_col = None
                for col in df.columns:
                    if 'jan' in col.lower() and '2024' in col.lower():
                        value_col = col
                        break
                
                if value_col is None:
                    # Try to find any numeric column
                    for col in df.columns:
                        if df[col].dtype in ['int64', 'float64'] or '2024' in str(col):
                            value_col = col
                            break
                
                if value_col is None:
                    logger.warning(f"Could not identify value column in {file_path}")
                    continue
                
                # Extract relevant data
                brands_df = df[[brand_col, value_col]].copy()
                brands_df.columns = ['Brand', 'Market_Share_Jan2024']
                
                # Clean brand names
                brands_df = brands_df.dropna(subset=['Brand'])
                brands_df = brands_df[brands_df['Brand'] != 'Illegal Brand']
                
                # Clean market share values
                brands_df['Market_Share_Jan2024'] = pd.to_numeric(
                    brands_df['Market_Share_Jan2024'], errors='coerce')
                
                # Remove rows with invalid market share values
                brands_df = brands_df.dropna(subset=['Market_Share_Jan2024'])
                
                if not brands_df.empty:
                    brand_data.append(brands_df)
                    logger.info(f"Successfully processed {len(brands_df)} brands from {filename}")
                else:
                    logger.warning(f"No valid brand data found in {filename}")
                    
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
                continue
        
        if brand_data:
            result = pd.concat(brand_data, ignore_index=True).dropna()
            logger.info(f"Successfully loaded brand data: {result.shape}")
            return result
        
        logger.warning("No valid brand data found in any files")
        return pd.DataFrame()
    
    def validate_data_integrity(self, state_df: pd.DataFrame, brand_df: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate data integrity and consistency
        
        Args:
            state_df: State-level data DataFrame
            brand_df: Brand-level data DataFrame
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'state_data_valid': False,
            'brand_data_valid': False,
            'data_consistency': False
        }
        
        try:
            # Validate state data
            if not state_df.empty:
                required_columns = ['State', 'Incidence of illegal cigarettes']
                missing_columns = [col for col in required_columns if col not in state_df.columns]
                
                if not missing_columns:
                    # Check for reasonable values
                    illegal_incidence = state_df['Incidence of illegal cigarettes']
                    if (illegal_incidence >= 0).all() and (illegal_incidence <= 100).all():
                        results['state_data_valid'] = True
            
            # Validate brand data
            if not brand_df.empty:
                required_columns = ['Brand', 'Market_Share_Jan2024']
                missing_columns = [col for col in required_columns if col not in brand_df.columns]
                
                if not missing_columns:
                    # Check for reasonable values
                    market_share = brand_df['Market_Share_Jan2024']
                    if (market_share >= 0).all():
                        results['brand_data_valid'] = True
            
            # Check data consistency
            if results['state_data_valid'] and results['brand_data_valid']:
                # Basic consistency check - this could be expanded
                results['data_consistency'] = True
                
        except Exception as e:
            logger.error(f"Error during data validation: {str(e)}")
        
        return results
    
    def save_processed_data(self, df: pd.DataFrame, filename: str) -> str:
        """
        Save processed data to the processed directory
        
        Args:
            df: DataFrame to save
            filename: Name of the file to save
            
        Returns:
            Path to the saved file
        """
        try:
            file_path = self.processed_dir / filename
            df.to_csv(file_path, index=False)
            logger.info(f"Saved processed data to {file_path}")
            return str(file_path)
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            raise
    
    def _get_valid_malaysian_states(self) -> List[str]:
        """Return a list of valid Malaysian state names"""
        return [
            'PERLIS', 'KEDAH', 'PENANG', 'PERAK', 'SELANGOR', 'WP KL', 
            'N.SEMBILAN', 'MELAKA', 'JOHOR', 'PAHANG', 'T\'GANU', 
            'KELANTAN', 'SABAH', 'SARAWAK'
        ]

def main():
    """Main function to demonstrate the enhanced data processor"""
    try:
        # Initialize the processor
        processor = EnhancedDataProcessor()
        
        # Load and clean state data
        state_data = processor.load_and_clean_state_data()
        
        # Load brand data
        brand_data = processor.load_brand_data()
        
        # Validate data integrity
        validation_results = processor.validate_data_integrity(state_data, brand_data)
        
        print("Data Validation Results:")
        for key, value in validation_results.items():
            print(f"  {key}: {value}")
        
        # Save processed data
        processor.save_processed_data(state_data, 'enhanced_state_data.csv')
        processor.save_processed_data(brand_data, 'enhanced_brand_data.csv')
        
        print(f"\nProcessed data saved successfully!")
        print(f"State data shape: {state_data.shape}")
        print(f"Brand data shape: {brand_data.shape}")
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    main()
