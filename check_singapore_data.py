"""
Script to check for Singapore-related data in the PDF
"""

import pdfplumber
import re

def check_singapore_data():
    """Check for Singapore-related information in the PDF"""
    try:
        with pdfplumber.open('Illicit-Cigarettes-Study--ICS--In-Malaysia--Jan-2024-Report.pdf') as pdf:
            print("=== CHECKING FOR SINGAPORE DATA ===\n")
            
            singapore_mentions = 0
            
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    # Look for Singapore mentions
                    if 'singapore' in text.lower() or 'singapura' in text.lower():
                        singapore_mentions += 1
                        print(f"Page {i+1}: Found Singapore mention")
                        lines = text.split('\n')
                        for line in lines:
                            if 'singapore' in line.lower() or 'singapura' in line.lower():
                                print(f"  Context: {line.strip()}")
                        print()
                    
                    # Look for compliance rates or exact percentages
                    if '99' in text and ('compliance' in text.lower() or 'legal' in text.lower()):
                        print(f"Page {i+1}: Potential compliance data")
                        lines = text.split('\n')
                        for line in lines:
                            if '99' in line and ('compliance' in line.lower() or 'legal' in line.lower()):
                                print(f"  Context: {line.strip()}")
                        print()
            
            print(f"Total pages with Singapore mentions: {singapore_mentions}")
            
            if singapore_mentions == 0:
                print("\nNo Singapore-related data found in the PDF.")
                print("The 99.2% compliance figure appears to be blog commentary, not from the source data.")
            
    except Exception as e:
        print(f"Error checking Singapore data: {e}")

if __name__ == "__main__":
    check_singapore_data()
