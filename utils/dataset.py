import pandas as pd
import requests
import os

def download_csv():
    """Download the ZINC drugs dataset CSV file."""
    csv_url = "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"
    csv_filename = "data/250k_rndm_zinc_drugs_clean_3.csv"
    
    # Download if file doesn't exist
    if not os.path.exists(csv_filename):
        print(f"Downloading {csv_filename}...")
        response = requests.get(csv_url)
        response.raise_for_status()
        with open(csv_filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {csv_filename}")
    else:
        print(f"{csv_filename} already exists")
    
    return csv_filename

csv_path = download_csv()