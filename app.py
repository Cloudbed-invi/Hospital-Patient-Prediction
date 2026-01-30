from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
import sys
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor

# Import our pipelines
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
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
base_df = None
current_df = None
data_lookup = {}
avgs = {}

# --- HELPER: Data Management ---

def load_base_system():
    global model, feature_names, base_df, current_df, data_lookup, avgs
    
    # 1. Load Model
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
        avgs = base_df.mean(numeric_only=True)
        rebuild_current_dataset()
    except Exception as e:
        print(f"Error loading base data: {e}")
        base_df = None

def rebuild_current_dataset():
    global current_df, data_lookup, base_df
    if base_df is None: return

    combined = base_df.copy()
    
    if os.path.exists(INCREMENTAL_DATA_PATH):
        try:
            inc_df = pd.read_csv(INCREMENTAL_DATA_PATH)
            inc_df['Date'] = pd.to_datetime(inc_df['Date'])
            
            for idx, row in inc_df.iterrows():
                d = row['Date']
                arr = row['Actual']
                mask = combined['Date'] == d
                if mask.any():
                    combined.loc[mask, 'Arrivals'] = arr
                    combined.loc[mask, 'Is_FineTuned'] = 1
                else:
                    new_row = row.to_dict()
                    new_row['Arrivals'] = arr
                    for col in combined.columns:
                        if col not in new_row:
                            new_row[col] = avgs.get(col, 0)
                    if 'Cluster_Encoded' not in new_row: new_row['Cluster_Encoded'] = 1
                    # combined = pd.concat([combined, pd.DataFrame([new_row])], ignore_index=True)
                    # Skip adding new rows for now unless requested, to keep charts aligned
                    pass
        except Exception as e:
            print(f"Error loading incremental data: {e}")

    combined = combined.sort_values('Date').reset_index(drop=True)
    
    # Recalculate Features
    combined['Target_NextDay_Arrivals'] = combined['Arrivals'].shift(-1)
    for lag in [1, 2, 3, 7]:
        combined[f'Lag_{lag}'] = combined['Arrivals'].shift(lag)
    combined['Rolling_Mean_7'] = combined['Arrivals'].rolling(window=7).mean()
    combined['Day_OfWeek'] = combined['Date'].dt.dayofweek
    combined['Month'] = combined['Date'].dt.month
    
    current_df = combined
    data_lookup = current_df.set_index('Date').to_dict('index')
    print(f"Dataset Rebuilt. Total Records: {len(current_df)}")

def retrain_model_in_memory():
    global model
    if current_df is None: return

    cutoff = pd.to_datetime(datetime.now().date())
    train_df = current_df[current_df['Date'] <= cutoff].copy()
    train_df = train_df.dropna(subset=feature_names + ['Target_NextDay_Arrivals'])
    
    if len(train_df) < 50: return

    print(f"Retraining Model on {len(train_df)} records...")
    X = train_df[feature_names]
    y = train_df['Target_NextDay_Arrivals']
    
    new_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    new_model.fit(X, y)
    model = new_model
    print("Model Retrained Successfully.")

load_base_system()

def get_cutoff_date():
    return pd.to_datetime(datetime.now().date())

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model: return jsonify({'error': 'Model not loaded'}), 500

    data = request.json
    date_str = data.get('date')
    try: target_date = pd.to_datetime(date_str)
    except: return jsonify({'error': 'Invalid date'}), 400
    
    cutoff = get_cutoff_date()
    response = {
        'date': date_str,
        'prediction': None,
        'actual': None,
        'is_historical': False,
        'status': "Future",
        'color': "gray",
        'message': "No prediction available."
    }

    if target_date <= cutoff:
        response['is_historical'] = True
        if target_date in data_lookup:
            val = int(data_lookup[target_date]['Arrivals'])
            response['actual'] = val
            response['message'] = "Observed Data Point."
            
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
    
    # Future Simulation Loop
    recent_history = current_df[current_df['Date'] <= cutoff].tail(14)['Arrivals'].tolist()
    loop_date = cutoff + timedelta(days=1)
    
    days_diff = (target_date - cutoff).days
    if days_diff > 30: return jsonify({'error': 'Forecast limited to 30 days ahead.'}), 400
    
    pred_val = 0
    while loop_date <= target_date:
        feats = {
            'Lag_1': recent_history[-1],
            'Lag_2': recent_history[-2],
            'Lag_3': recent_history[-3],
            'Lag_7': recent_history[-7],
            'Rolling_Mean_7': np.mean(recent_history[-7:]),
            'Bed_Occupancy': avgs.get('Bed_Occupancy', 0.5),
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
    Cols: [Date, Baseline, Feedback, Forecast]
    """
    if current_df is None: return jsonify({'error': 'Data not loaded'}), 500
    
    cutoff = get_cutoff_date()
    rows = []
    
    base_lookup = {}
    if base_df is not None:
        for _, r in base_df.iterrows():
            base_lookup[r['Date'].strftime('%Y-%m-%d')] = r['Arrivals']

    # 1. Observed / Baseline (Past)
    hist_subset = current_df[current_df['Date'] <= cutoff].tail(60) 
    
    last_val = None
    for _, row in hist_subset.iterrows():
        d_str = row['Date'].strftime('%Y-%m-%d')
        val_current = row['Arrivals']
        is_tuned = row.get('Is_FineTuned', 0) == 1
        
        # Baseline: original value
        val_base = base_lookup.get(d_str, val_current if not is_tuned else None)
        
        # Feedback: only if tuned
        val_feedback = val_current if is_tuned else None
        
        rows.append([d_str, val_base, val_feedback, None])
        last_val = val_current 
        
    # BRIDGE POINT
    if last_val is not None:
         # Start forecast at the effective last value
         rows[-1][3] = last_val
         
    # 2. Future Forecast
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
        rows.append([d_str, None, None, pred])
        sim_date += timedelta(days=1)

    return jsonify({
        'cutoff_date': cutoff.strftime('%Y-%m-%d'),
        'rows': rows
    })

@app.route('/update_data', methods=['POST'])
def update_data():
    try:
        data = request.json
        date_str = data.get('date')
        try: actual = float(data.get('arrivals'))
        except: return jsonify({'error': 'Invalid arrivals'}), 400
            
        target_date = pd.to_datetime(date_str)
        cutoff = get_cutoff_date()
        if target_date >= cutoff: return jsonify({'error': 'Cannot update future dates.'}), 400
            
        new_row = {'Date': date_str, 'Actual': actual}
        
        if os.path.exists(INCREMENTAL_DATA_PATH):
            df_inc = pd.read_csv(INCREMENTAL_DATA_PATH)
            df_inc = df_inc[df_inc['Date'] != date_str]
            df_inc = pd.concat([df_inc, pd.DataFrame([new_row])], ignore_index=True)
        else:
            df_inc = pd.DataFrame([new_row])
            
        df_inc.to_csv(INCREMENTAL_DATA_PATH, index=False)
        rebuild_current_dataset()
        retrain_model_in_memory()
        
        return jsonify({'status': 'success', 'message': 'Model retrained with new observed data.'})

    except Exception as e:
        print(f"Update failed: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
