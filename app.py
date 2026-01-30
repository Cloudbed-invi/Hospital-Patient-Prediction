from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
import sys
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor

# Import our pipelines (though we won't run full clustering)
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
# Check if files exist before importing to avoid errors if run from wrong dir
try:
    from clustering_pipeline import process_and_cluster
    from train_model import train_forecasting_model
except ImportError:
    pass

app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'hospital_model.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'processed_data.csv')
RAW_DATA_PATH = os.path.join(BASE_DIR, 'hospital_arrivals.csv')
INCREMENTAL_DATA_PATH = os.path.join(BASE_DIR, 'incremental_feedback.csv')

# Global State
model = None
feature_names = []
base_df = None       # The original loaded static data
current_df = None    # The working dataset (Base + Incremental)
data_lookup = {}
avgs = {}

# --- HELPER: Data Management ---

def load_base_system():
    global model, feature_names, base_df, current_df, data_lookup, avgs
    
    # 1. Load Model (Initial Baseline)
    try:
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
            model = model_data['model']
            feature_names = model_data['features']
        print("Base Model loaded.")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
        feature_names = []

    # 2. Load Base Data
    try:
        base_df = pd.read_csv(DATA_PATH)
        base_df['Date'] = pd.to_datetime(base_df['Date'])
        base_df = base_df.sort_values('Date')
        
        # Calculate globals
        avgs = base_df.mean(numeric_only=True)
        
        # Initial Build
        rebuild_current_dataset()
        
    except Exception as e:
        print(f"Error loading base data: {e}")
        base_df = None

def rebuild_current_dataset():
    """
    Merges Base Data + Incremental Feedback.
    Recalculates Lags.
    """
    global current_df, data_lookup, base_df
    
    if base_df is None:
        return

    # Start with Base
    combined = base_df.copy()
    
    # Load Incremental
    if os.path.exists(INCREMENTAL_DATA_PATH):
        try:
            inc_df = pd.read_csv(INCREMENTAL_DATA_PATH)
            inc_df['Date'] = pd.to_datetime(inc_df['Date'])
            
            # Identify columns to keep/merge
            # We strictly need Date and Arrivals.
            # Other columns (Occupancy etc) we might inherit or use defaults.
            
            for idx, row in inc_df.iterrows():
                d = row['Date']
                arr = row['Actual']
                
                # If date exists in Base, Update it
                mask = combined['Date'] == d
                if mask.any():
                    combined.loc[mask, 'Arrivals'] = arr
                    # We assume Is_FineTuned = 1 for visualization if needed, 
                    # but pure 'Arrivals' is what matters for training
                    combined.loc[mask, 'Is_FineTuned'] = 1
                else:
                    # Append new row
                    # We need to fill missing columns (Bed_Occupancy, etc)
                    # Use AVGS or Defaults
                    new_row = row.to_dict()
                    new_row['Arrivals'] = arr
                    # Fill missing features with averages
                    for col in combined.columns:
                        if col not in new_row:
                            if col in avgs:
                                new_row[col] = avgs[col]
                            else:
                                new_row[col] = 0
                    
                    if 'Cluster_Encoded' not in new_row:
                        new_row['Cluster_Encoded'] = 1 # Default
                        
                    combined = pd.concat([combined, pd.DataFrame([new_row])], ignore_index=True)
            
        except Exception as e:
            print(f"Error loading incremental data: {e}")

    # Sort
    combined = combined.sort_values('Date').reset_index(drop=True)
    
    # RECALCULATE FEATURES (Lags)
    # Because updating yesterday affects today's Lag_1
    combined['Target_NextDay_Arrivals'] = combined['Arrivals'].shift(-1)
    
    for lag in [1, 2, 3, 7]:
        combined[f'Lag_{lag}'] = combined['Arrivals'].shift(lag)
    
    combined['Rolling_Mean_7'] = combined['Arrivals'].rolling(window=7).mean()
    
    # Other features like Day_OfWeek need to be ensured if new rows added
    combined['Day_OfWeek'] = combined['Date'].dt.dayofweek
    combined['Month'] = combined['Date'].dt.month
    
    # Dropna for training purposes (start of dataset)
    # But keep full for lookup
    current_df = combined
    data_lookup = current_df.set_index('Date').to_dict('index')
    print(f"Dataset Rebuilt. Total Records: {len(current_df)}")

def retrain_model_in_memory():
    """
    Performs Rolling Retrain on current_df.
    Constraints: Only Historical Data (<= Today)
    """
    global model
    
    if current_df is None: 
        return

    cutoff = pd.to_datetime(datetime.now().date())
    
    # Filter for Training
    train_df = current_df[current_df['Date'] <= cutoff].copy()
    train_df = train_df.dropna(subset=feature_names + ['Target_NextDay_Arrivals'])
    
    if len(train_df) < 50:
        print("Not enough data to retrain.")
        return

    print(f"Retraining Model on {len(train_df)} records...")
    
    X = train_df[feature_names]
    y = train_df['Target_NextDay_Arrivals']
    
    # Using RandomForest as it's robust and fast for this size
    new_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    new_model.fit(X, y)
    
    model = new_model
    print("Model Retrained Successfully.")

# Initial Load
load_base_system()

def get_cutoff_date():
    return pd.to_datetime(datetime.now().date())

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.json
    date_str = data.get('date')
    try:
        target_date = pd.to_datetime(date_str)
    except:
        return jsonify({'error': 'Invalid date'}), 400
    
    cutoff = get_cutoff_date()
    
    # Response
    response = {
        'date': date_str,
        'prediction': None,
        'actual': None,
        'is_historical': False,
        'status': "Future",
        'color': "gray",
        'message': "No prediction available."
    }

    # Historical
    if target_date <= cutoff:
        response['is_historical'] = True
        if target_date in data_lookup:
            val = int(data_lookup[target_date]['Arrivals'])
            response['actual'] = val
            response['message'] = "Observed Data Point." # Renamed semantics
            
            # Status
            avg_arr = avgs['Arrivals']
            if val > avg_arr * 1.5:
                response['status'] = "High Load (Observed)"
                response['color'] = "red"
            else:
                 response['status'] = "Normal (Observed)"
                 response['color'] = "blue"
        else:
            response['message'] = "No data for this date."
        return jsonify(response)
    
    # Future
    # Generate features for target_date using current_df (which has up-to-date Lags)
    # We need to construct the input vector.
    
    # If target_date is tomorrow, we use today's lags.
    # If target_date is far future, we need to iterate (recursive forecast).
    # For single point predict, we'll try to find it in current_df if it exists (e.g. pre-calculated?), 
    # but current_df only goes up to max date.
    
    # Let's do a quick recursive generation from Cutoff to Target
    # Optimize: Just generate inputs based on latest data if target is close?
    
    # Robust: Simulate forward from Cutoff
    
    curr = cutoff
    # Find the row for (curr) to get its values, then predict next day
    # We need to reach target_date.
    
    # Simple check: Is target_date just 1 day ahead?
    # If so, take Lags from Cutoff.
    
    # For the specific 'predict' button which might be any date...
    # We will just run the simulation loop until we hit target_date
    
    sim_date = cutoff
    current_feats = {} # We need to grab starting state
    
    # We can't easily simulate N days instantly without a loop. 
    # Limit recursion to 30 days.
    days_diff = (target_date - cutoff).days
    if days_diff > 30:
        return jsonify({'error': 'Forecast limited to 30 days ahead.'}), 400
    
    # Instead of simulating here, let's reuse logic from 'history_and_forecast' 
    # but that's inefficient.
    # Simplified: Get features if we can.
    
    # Construct features using 'get_features_dynamic'
    # BUT, to get Lag_1 for T, we need prediction for T-1.
    
    # ... For this specific endpoint, let's just return "Forecast" if it's in the 14 day window
    # or error if too far? 
    # User requirement is "Allow user to enter...".
    # I'll implement a light loop.
    
    # Bootstrap simulation
    # We need the last known real data to start lags.
    last_known = current_df[current_df['Date'] <= cutoff].iloc[-1]
    
    # This is getting complex to do on the fly for a single point. 
    # Let's just assume the user asks for tomorrow or within range.
    # Actually, the user just wants to see the effect.
    
    # Quick Loop
    loop_date = cutoff + timedelta(days=1)
    
    # We need a rolling window of recent 'Arrivals' to compute Lags
    # Get last 10 days from current_df
    recent_history = current_df[current_df['Date'] <= cutoff].tail(14)['Arrivals'].tolist()
    
    pred_val = 0
    
    while loop_date <= target_date:
        # Build features
        # Lag 1 is recent_history[-1]
        
        feats = {
            'Lag_1': recent_history[-1],
            'Lag_2': recent_history[-2],
            'Lag_3': recent_history[-3],
            'Lag_7': recent_history[-7],
            'Rolling_Mean_7': np.mean(recent_history[-7:]),
            'Bed_Occupancy': avgs.get('Bed_Occupancy', 0.5), # Static assumption for future
            'Avg_Wait_Time': avgs.get('Avg_Wait_Time', 10),
            'Cluster_Encoded': 1,
            'Day_OfWeek': loop_date.dayofweek,
            'Month': loop_date.month
        }
        
        inp = pd.DataFrame([feats])
        for col in feature_names:
            if col not in inp.columns: inp[col] = 0
        inp = inp[feature_names]
        
        pred_val = int(model.predict(inp)[0])
        
        recent_history.append(pred_val)
        loop_date += timedelta(days=1)

    response['prediction'] = pred_val
    response['status'] = "Forecast"
    response['color'] = "orange"
    response['message'] = "Future Prediction (Updated Model)"
    
    return jsonify(response)


@app.route('/api/history_and_forecast', methods=['GET'])
def history_and_forecast():
    """
    Returns data for Google Charts.
    Cols: [Date, Observed (Baseline), Forecast]
    """
    if current_df is None:
        return jsonify({'error': 'Data not loaded'}), 500
        
    cutoff = get_cutoff_date()
    
    rows = []
    
    # 1. Observed / Baseline (Past)
    # Using current_df which includes User Corrections
    hist_subset = current_df[current_df['Date'] <= cutoff].tail(60) # Last 60 days
    
    for _, row in hist_subset.iterrows():
        d_str = row['Date'].strftime('%Y-%m-%d')
        val = row['Arrivals']
        rows.append([d_str, val, None]) # Observed, Forecast
        
    # 2. Future Forecast (Loop)
    recent_history = current_df[current_df['Date'] <= cutoff].tail(14)['Arrivals'].tolist()
    sim_date = cutoff + timedelta(days=1)
    
    for i in range(14):
        feats = {
            'Lag_1': recent_history[-1],
            'Lag_2': recent_history[-2],
            'Lag_3': recent_history[-3],
            'Lag_7': recent_history[-7],
            'Rolling_Mean_7': np.mean(recent_history[-7:]),
            'Bed_Occupancy': avgs.get('Bed_Occupancy', 0.5),
            'Avg_Wait_Time': avgs.get('Avg_Wait_Time', 10),
            'Cluster_Encoded': 1,
            'Day_OfWeek': sim_date.dayofweek,
            'Month': sim_date.month
        }
        
        inp = pd.DataFrame([feats])
        for col in feature_names:
            if col not in inp.columns: inp[col] = 0
        inp = inp[feature_names]
        
        pred = int(model.predict(inp)[0])
        recent_history.append(pred)
        
        d_str = sim_date.strftime('%Y-%m-%d')
        rows.append([d_str, None, pred])
        
        sim_date += timedelta(days=1)

    return jsonify({
        'cutoff_date': cutoff.strftime('%Y-%m-%d'),
        'rows': rows
    })

@app.route('/update_data', methods=['POST'])
def update_data():
    """
    Rolling Retrain Endpoint.
    1. Parse Input.
    2. Save to incremental_feedback.csv.
    3. Call rebuild_current_dataset() -> updates Lags.
    4. Call retrain_model_in_memory().
    """
    try:
        data = request.json
        date_str = data.get('date')
        try:
            actual = float(data.get('arrivals'))
        except:
            return jsonify({'error': 'Invalid arrivals'}), 400
            
        target_date = pd.to_datetime(date_str)
        cutoff = get_cutoff_date()
        
        if target_date >= cutoff:
            return jsonify({'error': 'Cannot update future dates.'}), 400
            
        # 1. Save
        new_row = {'Date': date_str, 'Actual': actual}
        # We save minimal info, rebuild handles the rest
        
        if os.path.exists(INCREMENTAL_DATA_PATH):
            df_inc = pd.read_csv(INCREMENTAL_DATA_PATH)
            # Remove old
            df_inc = df_inc[df_inc['Date'] != date_str]
            df_inc = pd.concat([df_inc, pd.DataFrame([new_row])], ignore_index=True)
        else:
            df_inc = pd.DataFrame([new_row])
            
        df_inc.to_csv(INCREMENTAL_DATA_PATH, index=False)
        
        # 2. Rebuild & Retrain
        rebuild_current_dataset()
        retrain_model_in_memory()
        
        return jsonify({
            'status': 'success',
            'message': 'Model retrained with new observed data.'
        })

    except Exception as e:
        print(f"Update failed: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
