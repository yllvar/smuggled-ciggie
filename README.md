# Malaysia Illicit Cigarettes Study (ICS) Analysis

## Overview
This project implements a comprehensive data science analysis for the Illicit Cigarettes Study (ICS) Malaysia January 2024 Report, focusing on spatio-temporal surveillance and national market simulation.

## Key Features
- **Spatio-Temporal Surveillance**: Detection of illegal cigarette hotspots and forecasting
- **National Market Simulation**: Economic loss estimation and enforcement ROI analysis
- **PDF Data Extraction**: Automated extraction and digitization from the ICS PDF report
- **Nielsen's Weighting Methodology**: Implementation of official statistical methodology
- **Pattern Detection**: Spatial-temporal pattern analysis and hotspot identification
- **Forecasting**: Time series prediction of illegal cigarette incidence by state
- **Economic Simulation**: Enforcement scenario modeling and return on investment analysis

## Project Structure
```
├── malaysia-illicit-cigarettes-analysis.ipynb  # Main analysis notebook
├── data/
│   ├── raw/          # Extracted PDF tables and raw data
│   ├── processed/    # Cleaned and processed datasets
│   └── external/     # External reference data
├── outputs/
│   ├── figures/      # Generated visualizations and charts
│   ├── reports/      # Analysis reports and summaries
│   └── simulations/  # Economic simulation results
├── Illicit-Cigarettes-Study--ICS--In-Malaysia--Jan-2024-Report.pdf  # Source report
└── README.md         # This file
```

## Key Objectives
1. Extract and digitize data from the ICS PDF report
2. Implement Nielsen's weighting methodology for accurate representation
3. Detect spatial-temporal patterns and identify hotspots
4. Forecast illegal cigarette incidence by Malaysian states
5. Simulate enforcement scenarios and calculate ROI
6. Validate results against authoritative report figures

## Data Sources
- **Primary**: Illicit Cigarettes Study (ICS) Malaysia January 2024 Report
- **Coverage**: All 16 Malaysian states and federal territories
- **Time Period**: 2019-2024 with monthly tracking from Sep 2023
- **Metrics**: Illegal cigarette incidence, brand analysis, tax stamp verification

## Analysis Components

### 1. Data Extraction & Processing
- PDF parsing using `pdfplumber` and `tabula-py`
- Automated table extraction and cleaning
- State-level data aggregation

### 2. Spatial Analysis
- Geographic visualization using `geopandas` and `folium`
- Hotspot detection with clustering algorithms
- Regional pattern analysis

### 3. Time Series Analysis
- Trend analysis using `statsmodels` and `Prophet`
- Seasonal decomposition and forecasting
- State-by-state incidence predictions

### 4. Economic Modeling
- Market size estimation
- Enforcement ROI calculation
- Scenario simulation and impact assessment

### 5. Statistical Validation
- Nielsen methodology implementation
- Confidence interval calculations
- Cross-validation against official figures

## Key Findings
The analysis reveals:
- National illegal cigarette incidence trends
- State-level variations and hotspots
- Brand market share dynamics
- Tax stamp compliance patterns
- Economic impact assessments

## Technical Requirements

### Python Packages
```python
# Core data science
pandas, numpy, matplotlib, seaborn

# Spatial analysis
geopandas, folium, shapely

# Time series & forecasting
statsmodels, prophet

# PDF processing
pdfplumber, tabula-py

# Machine learning
scikit-learn

# Visualization
plotly
```

### Installation
```bash
pip install pandas numpy matplotlib seaborn geopandas folium shapely statsmodels prophet pdfplumber tabula-py scikit-learn plotly
```

## Usage

1. **Run the main notebook**:
   ```bash
   jupyter notebook malaysia-illicit-cigarettes-analysis.ipynb
   ```

2. **Data Processing**:
   - Ensure the PDF report is in the root directory
   - Run the data extraction cells first
   - Process extracted tables through cleaning pipeline

3. **Analysis**:
   - Follow the notebook sections sequentially
   - Each section builds on previous data processing
   - Results are saved to the outputs directory

## Methodology Notes

### Nielsen's Weighting Approach
- Implements official Nielsen Consumer LLC methodology
- Accounts for demographic and geographic weighting
- Ensures representative sampling across Malaysia

### Validation Framework
- Cross-checks extracted data against PDF figures
- Validates calculations using report benchmarks
- Ensures statistical significance and confidence intervals

## Output Files
- **Figures**: Interactive maps, trend charts, and statistical visualizations
- **Reports**: Comprehensive analysis summaries and key insights
- **Simulations**: Economic impact models and enforcement scenarios

## Contributing
This analysis follows rigorous data science practices:
- Reproducible code with clear documentation
- Modular functions for easy maintenance
- Comprehensive error handling and validation

## License
Based on the © 2023 Nielsen Consumer LLC report. This analysis code is provided for research and educational purposes.

## Contact
For questions about the analysis methodology or results, please refer to the original ICS Malaysia January 2024 Report for authoritative data and methodology details.
