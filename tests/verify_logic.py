import sys
import os
import pandas as pd
from datetime import datetime
import json

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
sys.path.append(os.getcwd())

# Import app functions
from app import get_cutoff_date, load_system, app
from flask import Flask

def verify():
    print("--- Starting Verification ---")
    
    # 1. Check Cutoff Date
    cutoff = get_cutoff_date()
    print(f"Cutoff Date detected: {cutoff}")
    assert cutoff.date() == datetime.now().date(), "Cutoff date should match system date!"

    # 2. Check Data Loading
    with app.app_context():
        # Manually load system if not loaded (app triggers it on import but context matters)
        # app.py calls load_system() on module level, so it should be loaded.
        pass

    # 3. Test Prediction Logic via API Client
    client = app.test_client()
    
    # HISTORY TEST
    # Pick a date definitely in the past (e.g. 2025-01-01)
    past_date = "2025-01-01"
    print(f"\nTesting Historical Date: {past_date}")
    resp = client.post('/predict', json={'date': past_date})
    data = resp.get_json()
    print(f"Response: {json.dumps(data, indent=2)}")
    
    if data['is_historical'] != True:
        print("FAIL: Should be historical!")
    elif data['status'] == "Future":
        print("FAIL: Status should not be Future!")
    else:
        print("PASS: Historical logic confirmed.")

    # FUTURE TEST
    # Pick a date definitely in the future (e.g. 2027-01-01)
    # Note: 2026-01-30 is today.
    future_date = "2026-05-01"
    print(f"\nTesting Future Date: {future_date}")
    resp = client.post('/predict', json={'date': future_date})
    data = resp.get_json()
    print(f"Response: {json.dumps(data, indent=2)}")
    
    if data['is_historical'] == True:
         print("FAIL: Should be Future!")
    elif data['prediction'] is None:
         print("FAIL: Should yield prediction!")
    else:
         print("PASS: Future prediction logic confirmed.")

    # 4. Graph Data Test
    print(f"\nTesting Graph Endpoint")
    resp = client.get('/api/history_and_forecast')
    data = resp.get_json()
    
    history_len = len(data['history'])
    forecast_len = len(data['forecast'])
    print(f"History Points: {history_len}")
    print(f"Forecast Points: {forecast_len}")
    
    if history_len > 0 and forecast_len > 0:
        print("PASS: Graph endpoint returning data.")
        # Check integrity
        last_hist = data['history'][-1]['Date']
        first_forecast = data['forecast'][0]['Date']
        print(f"Last History: {last_hist}")
        print(f"First Forecast: {first_forecast}")
        
    else:
        print("FAIL: Missing graph data.")

if __name__ == "__main__":
    verify()
