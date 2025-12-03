"""
Script to verify data from the original PDF report
"""

import pdfplumber
import re

def extract_key_data():
    """Extract key data from the PDF report"""
    try:
        with pdfplumber.open('Illicit-Cigarettes-Study--ICS--In-Malaysia--Jan-2024-Report.pdf') as pdf:
            print("=== EXTRACTING KEY DATA FROM PDF REPORT ===\n")
            
            # Check first few pages for key information
            for i, page in enumerate(pdf.pages[:10]):
                text = page.extract_text()
                if text:
                    # Look for total market size and tax loss
                    if 'billion' in text.lower() and ('tax' in text.lower() or 'revenue' in text.lower() or 'loss' in text.lower()):
                        print(f"Page {i+1} - Potential tax loss information:")
                        lines = text.split('\n')
                        for line in lines:
                            if 'billion' in line.lower() and ('tax' in line.lower() or 'revenue' in line.lower() or 'loss' in line.lower()):
                                print(f"  {line.strip()}")
                        print()
                    
                    # Look for market share information
                    if 'illicit' in text.lower() and 'market' in text.lower():
                        print(f"Page {i+1} - Potential market share information:")
                        lines = text.split('\n')
                        for line in lines:
                            if 'illicit' in line.lower() and 'market' in line.lower():
                                print(f"  {line.strip()}")
                        print()
                    
                    # Look for consumption figures
                    if 'consumption' in text.lower() or 'packs' in text.lower():
                        print(f"Page {i+1} - Potential consumption information:")
                        lines = text.split('\n')
                        for line in lines:
                            if ('consumption' in line.lower() or 'packs' in line.lower()) and ('million' in line.lower() or 'billion' in line.lower()):
                                print(f"  {line.strip()}")
                        print()
            
            print("=== END OF EXTRACTION ===")
            
    except Exception as e:
        print(f"Error extracting data: {e}")

if __name__ == "__main__":
    extract_key_data()
