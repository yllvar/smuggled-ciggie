"""
Simple Interactive Dashboard for Malaysia Illicit Cigarettes Study
"""

from flask import Flask, render_template, jsonify, request
import pandas as pd
import json
import os

app = Flask(__name__)

# Load data
def load_data():
    """Load all relevant data for the dashboard"""
    try:
        # Load state data
        state_file = 'data/raw/page_58_table_1.csv'
        if os.path.exists(state_file):
            state_data = pd.read_csv(state_file)
            # Clean column names
            state_data.columns = state_data.columns.str.replace('\n', ' ').str.strip()
        else:
            state_data = pd.DataFrame()
        
        # Load forecasting data
        forecast_file = 'outputs/forecasting/enhanced_forecasts.json'
        if os.path.exists(forecast_file):
            with open(forecast_file, 'r') as f:
                forecast_data = json.load(f)
        else:
            forecast_data = {}
        
        # Load simulation summary
        simulation_file = 'outputs/simulations/simulation_summary.json'
        if os.path.exists(simulation_file):
            with open(simulation_file, 'r') as f:
                simulation_data = json.load(f)
        else:
            simulation_data = {}
        
        return {
            'state_data': state_data,
            'forecast_data': forecast_data,
            'simulation_data': simulation_data
        }
    except Exception as e:
        print(f"Error loading data: {e}")
        return {
            'state_data': pd.DataFrame(),
            'forecast_data': {},
            'simulation_data': {}
        }

# Global data variable
DATA = load_data()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/state_data')
def get_state_data():
    """API endpoint for state-level data"""
    try:
        if not DATA['state_data'].empty:
            # Convert to dictionary for JSON serialization
            data = DATA['state_data'].to_dict(orient='records')
            return jsonify({
                'status': 'success',
                'data': data
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'No state data available'
            })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/api/forecast_data')
def get_forecast_data():
    """API endpoint for forecasting data"""
    try:
        if DATA['forecast_data']:
            return jsonify({
                'status': 'success',
                'data': DATA['forecast_data']
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'No forecast data available'
            })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/api/simulation_data')
def get_simulation_data():
    """API endpoint for simulation data"""
    try:
        if DATA['simulation_data']:
            return jsonify({
                'status': 'success',
                'data': DATA['simulation_data']
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'No simulation data available'
            })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/api/reload_data')
def reload_data():
    """API endpoint to reload data"""
    global DATA
    DATA = load_data()
    return jsonify({
        'status': 'success',
        'message': 'Data reloaded successfully'
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
