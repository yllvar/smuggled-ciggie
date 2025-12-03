"""
Script to quickly view blog visualizations
"""

import os
import webbrowser
from pathlib import Path

def view_visualizations():
    """Open key visualizations in the default browser"""
    viz_dir = Path(__file__).parent
    
    # Key visualizations to view
    key_files = [
        'market_analysis_visualization.png',
        'enforcement_roi_analysis.png',
        'state_tax_loss_chart.png',
        'interactive_brand_treemap.html',
        'interactive_incidence_map.html'
    ]
    
    print("Opening key visualizations...")
    
    for filename in key_files:
        file_path = viz_dir / filename
        if file_path.exists():
            print(f"Opening {filename}")
            webbrowser.open(f"file://{file_path.absolute()}")
        else:
            print(f"File not found: {filename}")
    
    print("\nAll key visualizations opened in browser tabs.")
    print("Check your browser for the visualization tabs.")

if __name__ == "__main__":
    view_visualizations()
