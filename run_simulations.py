#!/usr/bin/env python3
"""
Economic Simulation Script for Malaysia Illicit Cigarettes Study
Populates outputs/simulations folder with comprehensive analysis results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Create simulation outputs directory
os.makedirs('outputs/simulations', exist_ok=True)

print("🚀 Starting Malaysia Illicit Cigarettes Economic Simulation...")

# 1. Load and Process State Data
def load_and_clean_state_data():
    """Load and clean the state-level incidence data"""
    
    # Load the main state data table
    state_df = pd.read_csv('data/raw/page_58_table_1.csv')
    
    # Clean column names
    state_df.columns = state_df.columns.str.replace('\n', ' ').str.strip()
    
    # Remove header rows and clean data
    state_df = state_df.dropna(subset=['State'])
    state_df = state_df[~state_df['State'].isin(['A', 'B', 'C', 'D', 'E'])]
    
    # Clean numeric columns
    numeric_cols = ['Total packs collected (Jan\'24)', 'Number of legal packs collected', 
                   'Number of illegal packs collected']
    
    for col in numeric_cols:
        if col in state_df.columns:
            state_df[col] = state_df[col].astype(str).str.replace(',', '').astype(float)
    
    # Clean percentage columns
    percentage_cols = ['Incidence of legal cigarettes', 'Incidence of illegal cigarettes']
    for col in percentage_cols:
        if col in state_df.columns:
            state_df[col] = state_df[col].astype(str).str.replace('%', '').astype(float)
    
    return state_df

# Load brand data
def load_brand_data():
    """Load and clean brand market share data"""
    
    brand_files = ['data/raw/page_22_table_1.csv', 'data/raw/page_19_table_1.csv']
    brand_data = []
    
    for file in brand_files:
        try:
            df = pd.read_csv(file)
            # Clean and extract brand information
            df = df.dropna(subset=['Unnamed: 1'])
            df = df[df['Unnamed: 1'] != 'Illegal Brand']
            
            # Extract relevant columns
            if 'Jan, 2024' in df.columns:
                brands_df = df[['Unnamed: 1', 'Jan, 2024']].copy()
                brands_df.columns = ['Brand', 'Market_Share_Jan2024']
                brands_df['Market_Share_Jan2024'] = pd.to_numeric(brands_df['Market_Share_Jan2024'], errors='coerce')
                brand_data.append(brands_df)
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    if brand_data:
        return pd.concat(brand_data, ignore_index=True).dropna()
    return pd.DataFrame()

print("📊 Loading and cleaning data...")
state_data = load_and_clean_state_data()
brand_data = load_brand_data()

print(f"State data shape: {state_data.shape}")
print(f"Brand data shape: {brand_data.shape}")

# 2. Economic Simulation Model
class EconomicSimulationModel:
    """Economic simulation model for illicit cigarette market analysis"""
    
    def __init__(self, state_data, brand_data):
        self.state_data = state_data
        self.brand_data = brand_data
        
        # Economic parameters (based on Malaysian market context)
        self.avg_price_per_pack_legal = 17.0  # RM per pack (average legal price)
        self.avg_price_per_pack_illegal = 8.0  # RM per pack (average illegal price)
        self.tax_revenue_per_pack_legal = 12.0  # RM tax per legal pack
        self.enforcement_cost_per_operation = 50000  # RM per enforcement operation
        self.seizure_rate_per_operation = 0.15  # 15% seizure rate per operation
        
        # Market size parameters
        self.total_smokers_malaysia = 5_000_000  # Estimated total smokers
        self.avg_consumption_per_smoker = 15  # sticks per day
        
    def calculate_market_size(self):
        """Calculate total market size and economic impact"""
        
        # Calculate total packs consumption
        total_daily_sticks = self.total_smokers_malaysia * self.avg_consumption_per_smoker
        total_annual_sticks = total_daily_sticks * 365
        total_annual_packs = total_annual_sticks / 20  # 20 sticks per pack
        
        # Use state data to get illegal incidence
        avg_illegal_incidence = self.state_data['Incidence of illegal cigarettes'].mean() / 100
        
        # Calculate market segments
        legal_packs = total_annual_packs * (1 - avg_illegal_incidence)
        illegal_packs = total_annual_packs * avg_illegal_incidence
        
        # Calculate economic values
        legal_market_value = legal_packs * self.avg_price_per_pack_legal
        illegal_market_value = illegal_packs * self.avg_price_per_pack_illegal
        tax_revenue_loss = illegal_packs * self.tax_revenue_per_pack_legal
        
        market_analysis = {
            'total_annual_packs': total_annual_packs,
            'legal_packs': legal_packs,
            'illegal_packs': illegal_packs,
            'legal_market_value_rm': legal_market_value,
            'illegal_market_value_rm': illegal_market_value,
            'tax_revenue_loss_rm': tax_revenue_loss,
            'avg_illegal_incidence_pct': avg_illegal_incidence * 100
        }
        
        return market_analysis
    
    def simulate_enforcement_scenarios(self, scenarios):
        """Simulate different enforcement scenarios and ROI"""
        
        results = []
        
        for scenario in scenarios:
            scenario_name = scenario['name']
            enforcement_intensity = scenario['intensity']  # Number of operations
            effectiveness = scenario['effectiveness']  # Multiplier for seizure rate
            
            # Calculate enforcement costs
            total_enforcement_cost = enforcement_intensity * self.enforcement_cost_per_operation
            
            # Calculate seizures
            effective_seizure_rate = self.seizure_rate_per_operation * effectiveness
            total_seized_packs = 0
            
            for _, state in self.state_data.iterrows():
                market_analysis = self.calculate_market_size()
                state_illegal_packs = (market_analysis['total_annual_packs'] / len(self.state_data)) * \
                                    (state['Incidence of illegal cigarettes'] / 100)
                state_seizures = state_illegal_packs * effective_seizure_rate * enforcement_intensity
                total_seized_packs += state_seizures
            
            # Calculate economic impact
            seized_value = total_seized_packs * self.avg_price_per_pack_illegal
            recovered_tax = total_seized_packs * self.tax_revenue_per_pack_legal
            
            # Calculate ROI
            total_benefits = seized_value + recovered_tax
            roi = (total_benefits - total_enforcement_cost) / total_enforcement_cost * 100
            
            # Calculate market impact
            market_analysis = self.calculate_market_size()
            market_reduction_pct = (total_seized_packs / (market_analysis['total_annual_packs'] * 
                                 (self.state_data['Incidence of illegal cigarettes'].mean() / 100))) * 100
            
            results.append({
                'scenario': scenario_name,
                'enforcement_operations': enforcement_intensity,
                'total_cost_rm': total_enforcement_cost,
                'seized_packs': total_seized_packs,
                'seized_value_rm': seized_value,
                'recovered_tax_rm': recovered_tax,
                'total_benefits_rm': total_benefits,
                'roi_pct': roi,
                'market_reduction_pct': market_reduction_pct
            })
        
        return pd.DataFrame(results)
    
    def generate_state_level_projections(self):
        """Generate state-level economic impact projections"""
        
        state_projections = []
        
        for _, state in self.state_data.iterrows():
            state_name = state['State']
            illegal_incidence = state['Incidence of illegal cigarettes'] / 100
            
            # Calculate state market size (proportional to population)
            market_analysis = self.calculate_market_size()
            state_market_share = 1 / len(self.state_data)  # Simplified equal distribution
            state_total_packs = market_analysis['total_annual_packs'] * state_market_share
            state_illegal_packs = state_total_packs * illegal_incidence
            
            # Economic impact
            state_tax_loss = state_illegal_packs * self.tax_revenue_per_pack_legal
            state_illegal_market = state_illegal_packs * self.avg_price_per_pack_illegal
            
            # Enforcement recommendations
            recommended_operations = max(1, int(illegal_incidence * 10))
            projected_seizures = state_illegal_packs * self.seizure_rate_per_operation * recommended_operations
            projected_benefits = projected_seizures * (self.avg_price_per_pack_illegal + self.tax_revenue_per_pack_legal)
            
            state_projections.append({
                'state': state_name,
                'illegal_incidence_pct': illegal_incidence * 100,
                'estimated_illegal_packs_annual': state_illegal_packs,
                'tax_revenue_loss_rm': state_tax_loss,
                'illegal_market_value_rm': state_illegal_market,
                'recommended_enforcement_ops': recommended_operations,
                'projected_annual_seizures': projected_seizures,
                'projected_benefits_rm': projected_benefits,
                'priority_score': illegal_incidence * 100  # Simple priority scoring
            })
        
        return pd.DataFrame(state_projections)

# Initialize simulation model
sim_model = EconomicSimulationModel(state_data, brand_data)

# Calculate market size
market_analysis = sim_model.calculate_market_size()
print("\n=== MARKET SIZE ANALYSIS ===")
for key, value in market_analysis.items():
    if isinstance(value, float):
        print(f"{key}: {value:,.2f}")
    else:
        print(f"{key}: {value:,}")

# Define enforcement scenarios
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
        'name': 'High_Intensity',
        'intensity': 200,
        'effectiveness': 1.5
    },
    {
        'name': 'Targeted_Operations',
        'intensity': 75,
        'effectiveness': 2.0
    }
]

# Run enforcement simulations
simulation_results = sim_model.simulate_enforcement_scenarios(enforcement_scenarios)
print("\n=== ENFORCEMENT SCENARIO ANALYSIS ===")
print(simulation_results.to_string(index=False))

# Generate state-level projections
state_projections = sim_model.generate_state_level_projections()
print("\n=== STATE-LEVEL PROJECTIONS (Top 10 by Priority) ===")
top_states = state_projections.nlargest(10, 'priority_score')
print(top_states.to_string(index=False))

# Save simulation results
print("\n💾 Saving simulation results...")

# Save market analysis
with open('outputs/simulations/market_size_analysis.json', 'w') as f:
    json.dump(market_analysis, f, indent=2, default=str)

# Save enforcement scenarios
simulation_results.to_csv('outputs/simulations/enforcement_scenarios.csv', index=False)

# Save state projections
state_projections.to_csv('outputs/simulations/state_level_projections.csv', index=False)

# Create summary report
summary_report = {
    'analysis_date': datetime.now().isoformat(),
    'market_summary': market_analysis,
    'top_5_states_by_illegal_incidence': top_states[['state', 'illegal_incidence_pct', 'tax_revenue_loss_rm']].head(5).to_dict('records'),
    'best_enforcement_scenario': simulation_results.loc[simulation_results['roi_pct'].idxmax()].to_dict(),
    'total_potential_tax_recovery': market_analysis['tax_revenue_loss_rm']
}

with open('outputs/simulations/simulation_summary.json', 'w') as f:
    json.dump(summary_report, f, indent=2, default=str)

# 3. Create Basic Visualizations (without seaborn dependency)
print("\n📈 Creating visualizations...")

# Market Analysis Visualization
plt.style.use('default')
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Malaysia Illicit Cigarette Market Economic Analysis', fontsize=16, fontweight='bold')

# Market breakdown
market_categories = ['Legal Market', 'Illegal Market']
market_values = [market_analysis['legal_market_value_rm'], market_analysis['illegal_market_value_rm']]
colors = ['#2E8B57', '#DC143C']

ax1.pie(market_values, labels=market_categories, colors=colors, autopct='%1.1f%%', startangle=90)
ax1.set_title('Market Value Distribution (RM Billions)')

# Tax revenue loss
ax2.bar(['Tax Revenue Loss'], [market_analysis['tax_revenue_loss_rm']/1e9], color='#FF6B6B')
ax2.set_title('Annual Tax Revenue Loss')
ax2.set_ylabel('RM (Billions)')
ax2.grid(True, alpha=0.3)

# Incidence rates
incidence_data = [market_analysis['avg_illegal_incidence_pct'], 100-market_analysis['avg_illegal_incidence_pct']]
ax3.pie(incidence_data, labels=['Illegal', 'Legal'], colors=['#DC143C', '#2E8B57'], autopct='%1.1f%%')
ax3.set_title('Market Incidence Rates')

# Volume comparison
volumes = [market_analysis['legal_packs']/1e9, market_analysis['illegal_packs']/1e9]
ax4.bar(['Legal Packs', 'Illegal Packs'], volumes, color=['#2E8B57', '#DC143C'])
ax4.set_title('Annual Volume (Billions of Packs)')
ax4.set_ylabel('Packs (Billions)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/simulations/market_analysis_visualization.png', dpi=300, bbox_inches='tight')
plt.close()

# Enforcement ROI Analysis
fig, ax = plt.subplots(figsize=(12, 8))

# Create scatter plot
bubble_size = simulation_results['total_benefits_rm'] / 1e6
scatter = ax.scatter(simulation_results['enforcement_operations'], 
                    simulation_results['roi_pct'],
                    s=bubble_size*10,
                    c=simulation_results['market_reduction_pct'],
                    cmap='RdYlBu_r',
                    alpha=0.7,
                    edgecolors='black')

# Add labels for each scenario
for i, row in simulation_results.iterrows():
    ax.annotate(row['scenario'], 
                (row['enforcement_operations'], row['roi_pct']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=10, fontweight='bold')

ax.set_xlabel('Number of Enforcement Operations')
ax.set_ylabel('ROI (%)')
ax.set_title('Enforcement Scenario Analysis: ROI vs Operations Intensity')
ax.grid(True, alpha=0.3)

plt.colorbar(scatter, label='Market Reduction (%)')
plt.tight_layout()
plt.savefig('outputs/simulations/enforcement_roi_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# State Projections Bar Chart
fig, ax = plt.subplots(figsize=(15, 8))
top_10_states = state_projections.nlargest(10, 'priority_score')

ax.barh(top_10_states['state'], top_10_states['tax_revenue_loss_rm']/1e6, color='#FF6B6B')
ax.set_xlabel('Tax Revenue Loss (RM Millions)')
ax.set_title('Top 10 States: Annual Tax Revenue Loss from Illicit Cigarettes')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/simulations/state_tax_loss_chart.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Generate Executive Summary
executive_summary = f"""
# MALAYSIA ILLICIT CIGARETTE MARKET - ECONOMIC SIMULATION REPORT

## Executive Summary
- **Total Market Size**: {market_analysis['total_annual_packs']:,.0f} packs annually
- **Illegal Market Share**: {market_analysis['avg_illegal_incidence_pct']:.1f}%
- **Annual Tax Revenue Loss**: RM {market_analysis['tax_revenue_loss_rm']:,.2f}
- **Illegal Market Value**: RM {market_analysis['illegal_market_value_rm']:,.2f}

## Key Findings

### 1. Market Impact
The illicit cigarette market represents a significant economic challenge with:
- Tax revenue losses exceeding RM {market_analysis['tax_revenue_loss_rm']/1e9:.1f} billion annually
- Illegal market valued at RM {market_analysis['illegal_market_value_rm']/1e9:.1f} billion
- Average illegal incidence of {market_analysis['avg_illegal_incidence_pct']:.1f}% nationally

### 2. Enforcement Effectiveness
Best performing scenario: {simulation_results.loc[simulation_results['roi_pct'].idxmax(), 'scenario']}
- ROI: {simulation_results['roi_pct'].max():.1f}%
- Market reduction: {simulation_results.loc[simulation_results['roi_pct'].idxmax(), 'market_reduction_pct']:.1f}%

### 3. State-Level Priorities
Top 5 high-priority states for enforcement:
"""

for i, (_, state) in enumerate(top_states.head(5).iterrows()):
    executive_summary += f"""
{i+1}. **{state['state']}** - Incidence: {state['illegal_incidence_pct']:.1f}%, Tax Loss: RM {state['tax_revenue_loss_rm']/1e6:.1f}M
"""

executive_summary += f"""

## Recommendations
1. **Targeted Enforcement**: Focus on high-priority states with {top_states['priority_score'].mean():.1f}%+ illegal incidence
2. **Resource Optimization**: Implement {simulation_results.loc[simulation_results['roi_pct'].idxmax(), 'scenario']} scenario for optimal ROI
3. **Market Intelligence**: Strengthen monitoring in states with highest tax revenue losses
4. **Policy Impact**: Current enforcement could recover up to RM {simulation_results['total_benefits_rm'].max()/1e6:.1f}M annually

## Generated Files
- Market analysis visualizations
- Enforcement scenario comparisons  
- State-level risk assessments
- Comprehensive simulation data

Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

# Save executive summary
with open('outputs/simulations/executive_summary.md', 'w') as f:
    f.write(executive_summary)

print("✅ Economic simulation completed successfully!")
print("\n📁 Files created in outputs/simulations/:")
files_created = []
for file in os.listdir('outputs/simulations'):
    if file.endswith(('.json', '.csv', '.png', '.md')):
        files_created.append(file)
        print(f"  - {file}")

print(f"\n📊 Key Insights:")
print(f"• Annual tax revenue loss: RM {market_analysis['tax_revenue_loss_rm']/1e9:.2f} billion")
print(f"• Best enforcement scenario: {simulation_results.loc[simulation_results['roi_pct'].idxmax(), 'scenario']}")
print(f"• Maximum ROI achievable: {simulation_results['roi_pct'].max():.1f}%")
print(f"• High-priority states identified: {len(top_states[top_states['priority_score'] > 50])}")
print(f"• Total simulation files generated: {len(files_created)}")

print("\n🎯 Simulation outputs ready for analysis and decision-making!")
