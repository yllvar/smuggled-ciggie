# Malaysia Illicit Cigarettes Study - Interactive Dashboard

## Overview

This is a simple interactive dashboard for visualizing the results of the Malaysia Illicit Cigarettes Study. The dashboard provides real-time data visualization of key metrics, state-level analysis, enforcement scenarios, and forecasting results.

## Features

- **Key Metrics Display**: Shows total consumption, illegal market share, tax revenue loss, and illegal market value
- **State-Level Analysis**: Visualizes illegal cigarette incidence by state
- **Enforcement Scenarios**: Compares ROI across different enforcement strategies
- **Forecasting Trends**: Displays current vs forecasted illegal incidence
- **Data Tables**: Shows high-risk states and enforcement scenarios in tabular format
- **Real-time Data Reload**: Button to refresh data from source files

## Requirements

- Python 3.8+
- Flask
- Pandas
- Chart.js (loaded via CDN)
- Bootstrap (loaded via CDN)

## Installation

1. Navigate to the dashboard directory:
   ```
   cd dashboard
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Start the Flask application:
   ```
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

3. The dashboard will automatically load data from the main project directories:
   - `data/raw/` for state-level data
   - `outputs/forecasting/` for forecasting results
   - `outputs/simulations/` for simulation data

## Architecture

```
/dashboard/
├── app.py              # Main Flask application
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── templates/
    └── dashboard.html   # Main dashboard template
```

## API Endpoints

- `GET /` - Main dashboard page
- `GET /api/state_data` - State-level data in JSON format
- `GET /api/forecast_data` - Forecasting data in JSON format
- `GET /api/simulation_data` - Simulation data in JSON format
- `GET /api/reload_data` - Reload data from source files

## Customization

The dashboard can be easily customized by modifying the `templates/dashboard.html` file. The JavaScript code handles data loading and chart updates, while the HTML structure defines the layout and styling.

## Limitations

This is a basic implementation designed for demonstration purposes. For production use, consider:

- Adding authentication and authorization
- Implementing caching for better performance
- Adding more sophisticated error handling
- Including additional data visualizations
- Implementing real-time data updates with WebSockets

## Contributing

Feel free to fork this repository and submit pull requests for improvements.
