"""
Script to verify detailed data from the original PDF report
"""

import pdfplumber
import re

def extract_detailed_data():
    """Extract detailed data from the PDF report"""
    try:
        with pdfplumber.open('Illicit-Cigarettes-Study--ICS--In-Malaysia--Jan-2024-Report.pdf') as pdf:
            print("=== DETAILED DATA EXTRACTION FROM PDF REPORT ===\n")
            
            # Check more pages for key information
            for i, page in enumerate(pdf.pages[:20]):
                text = page.extract_text()
                if text:
                    lines = text.split('\n')
                    
                    # Look for monetary values
                    for line in lines:
                        if re.search(r'\d+\.?\d*\s*(billion|million)', line, re.IGNORECASE):
                            if any(keyword in line.lower() for keyword in ['rm', 'tax', 'revenue', 'loss', 'market', 'value', 'consumption']):
                                print(f"Page {i+1}: {line.strip()}")
                    
                    # Look for percentage values related to illicit market
                    for line in lines:
                        if '%' in line and ('illicit' in line.lower() or 'illegal' in line.lower() or 'market' in line.lower()):
                            print(f"Page {i+1}: {line.strip()}")
                            
            print("\n=== STATE-LEVEL DATA EXTRACTION ===\n")
            
            # Try to find state-level data
            for i, page in enumerate(pdf.pages[:30]):
                text = page.extract_text()
                if text:
                    lines = text.split('\n')
                    state_found = False
                    for line in lines:
                        # Look for state names with percentage data
                        if any(state in line for state in ['PAHANG', 'SARAWAK', 'SABAH', 'TERENGGANU', 'KELANTAN', 'PERLIS', 'KEDAH', 'PENANG', 'PERAK', 'SELANGOR', 'KUALA LUMPUR', 'N.SEMBILAN', 'MELAKA', 'JOHOR']):
                            if '%' in line or re.search(r'\d+\.?\d*', line):
                                print(f"Page {i+1}: {line.strip()}")
                                state_found = True
                    
                    if state_found:
                        print(f"--- Additional context from Page {i+1} ---")
                        # Print a few lines before and after for context
                        for j, line in enumerate(lines[max(0, len(lines)//2-10):min(len(lines), len(lines)//2+10)]):
                            if '%' in line or any(state in line for state in ['PAHANG', 'SARAWAK', 'SABAH', 'TERENGGANU', 'KELANTAN']):
                                print(f"  Context: {line.strip()}")
                        print()
            
            print("=== END OF DETAILED EXTRACTION ===")
            
    except Exception as e:
        print(f"Error extracting data: {e}")

if __name__ == "__main__":
    extract_detailed_data()
