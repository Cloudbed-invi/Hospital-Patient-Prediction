from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
import sys
from datetime import datetime, timedelta

# Import our pipelines for retraining
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
# Check if files exist before importing to avoid errors if run from wrong dir
try:
    from clustering_pipeline import process_and_cluster
    from train_model import train_forecasting_model
except ImportError:
    pass

app = Flask(__name__)

# Load Model and Data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'hospital_model.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'processed_data.csv')
RAW_DATA_PATH = os.path.join(BASE_DIR, 'hospital_arrivals.csv')

def load_system():
    global model, feature_names, df, data_lookup, avgs
    try:
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
            model = model_data['model']
            feature_names = model_data['features']
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

    try:
        df = pd.read_csv(DATA_PATH)
        df['Date'] = pd.to_datetime(df['Date'])
        # Ensure strict sorting
        df = df.sort_values('Date')
        data_lookup = df.set_index('Date').to_dict('index')
        avgs = df.mean(numeric_only=True)
        print("Historical data loaded.")
    except Exception as e:
        print(f"Error loading data: {e}")
        df = None
        data_lookup = {}
        avgs = {}

# Initial Load
load_system()

def get_cutoff_date():
    """Returns the current system date as the cutoff for Historical vs Future."""
    return pd.to_datetime(datetime.now().date())

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

    cutoff_date = get_cutoff_date()

    # Default response structure
    response = {
        'date': date_str,
        'prediction': None,
        'actual': None,
        'is_historical': False,
        'data_source_type': 'unknown',
        'status': "Future",
        'color': "gray",
        'message': "No prediction available.",
        'details': {}
    }

    # HISTORICAL CHECK
    if target_date <= cutoff_date:
        response['is_historical'] = True
        if target_date in data_lookup:
            row = data_lookup[target_date]
            actual_val = int(row['Arrivals'])
            response['actual'] = actual_val
            
            # Check source
            is_ft = row.get('Is_FineTuned', 0)
            response['data_source_type'] = 'finetuned' if is_ft == 1 else 'original'
            
            # For historical, we calculate status based on ACTUAL value
            avg_arr = avgs['Arrivals']
            std_arr = df['Arrivals'].std() if df is not None else 10
            
            if actual_val > avg_arr + 1.5 * std_arr:
                response['status'] = "Emergency Surge (Actual)"
                response['color'] = "red"
                response['message'] = "Historical High Load."
            elif actual_val < avg_arr - 1.0 * std_arr:
                response['status'] = "Low Load (Actual)"
                response['color'] = "green"
                response['message'] = "Historical Low Load."
            else:
                response['status'] = "Normal Load (Actual)"
                response['color'] = "blue" # Blue for normal history
                response['message'] = "Historical Normal Operation."
        else:
            response['message'] = "No historical record for this date."
        
        return jsonify(response)

    # FUTURE PREDICTION
    
    # We need inputs (Lags) from previous days.
    
    def get_arrival_for_date(d):
        if d in data_lookup:
            return data_lookup[d]['Arrivals']
        return avgs['Arrivals'] # Fallback
    
    features = {}
    features['Lag_1'] = get_arrival_for_date(target_date - timedelta(days=1))
    features['Lag_2'] = get_arrival_for_date(target_date - timedelta(days=2))
    features['Lag_3'] = get_arrival_for_date(target_date - timedelta(days=3))
    features['Lag_7'] = get_arrival_for_date(target_date - timedelta(days=7))
    features['Rolling_Mean_7'] = avgs['Arrivals'] # Approximation
    
    # Copy latest known context if available for other features
    latest_date = df['Date'].max()
    if latest_date in data_lookup:
        last_row = data_lookup[latest_date]
        features['Bed_Occupancy'] = last_row['Bed_Occupancy']
        features['Avg_Wait_Time'] = last_row['Avg_Wait_Time']
        features['Cluster_Encoded'] = last_row.get('Cluster_Encoded', 1)
    else:
        features['Bed_Occupancy'] = avgs['Bed_Occupancy']
        features['Avg_Wait_Time'] = avgs['Avg_Wait_Time']
        features['Cluster_Encoded'] = 1

    features['Day_OfWeek'] = target_date.dayofweek
    features['Month'] = target_date.month

    # DataFrame creation
    input_df = pd.DataFrame([features])
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_names]

    # Predict
    prediction = model.predict(input_df)[0]
    prediction = int(max(0, prediction))

    # Status Logic
    avg_arr = avgs['Arrivals']
    std_arr = df['Arrivals'].std()
    
    if prediction > avg_arr + 1.5 * std_arr:
        status = "Emergency Surge"
        color = "orange" # Future warning
        msg = "Predicted High Load!"
    elif prediction < avg_arr - 1.0 * std_arr:
        status = "Low Load"
        color = "lightgreen"
        msg = "Predicted Low Load."
    else:
        status = "Normal Load"
        color = "yellow"
        msg = "Predicted Normal."

    response['prediction'] = prediction
    response['status'] = status
    response['color'] = color
    response['message'] = msg
    response['details'] = {
        'day_of_week': target_date.day_name(),
        'input_features': features
    }
    
    return jsonify(response)

@app.route('/api/history_and_forecast', methods=['GET'])
def history_and_forecast():
    """
    Returns:
    1. Historical Data (up to Cutoff)
    2. Forecast Data (Cutoff + 1 to Cutoff + 14 days)
    """
    if df is None:
        return jsonify({'error': 'Data not loaded'}), 500
        
    cutoff_date = get_cutoff_date()
    
    # 1. Historical
    # Filter df <= cutoff
    hist_df = df[df['Date'] <= cutoff_date].sort_values('Date')
    # Limit to last 90 days
    start_history = cutoff_date - timedelta(days=90)
    hist_df = hist_df[hist_df['Date'] >= start_history]
    
    history_data = hist_df[['Date', 'Arrivals']].copy()
    history_data['Date'] = history_data['Date'].dt.strftime('%Y-%m-%d')
    history_list = history_data.to_dict('records')
    
    # 2. Forecast
    # Generate predictions for next 14 days
    forecast_list = []
    current_sim_date = cutoff_date + timedelta(days=1)
    
    occupancy = avgs['Bed_Occupancy']
    wait_time = avgs['Avg_Wait_Time']
    
    for i in range(14):
        target_d = current_sim_date + timedelta(days=i)
        
        feats = {
            'Lag_1': avgs['Arrivals'], 
            'Lag_2': avgs['Arrivals'],
            'Lag_3': avgs['Arrivals'],
            'Lag_7': avgs['Arrivals'],
            'Rolling_Mean_7': avgs['Arrivals'],
            'Bed_Occupancy': occupancy,
            'Avg_Wait_Time': wait_time,
            'Cluster_Encoded': 1,
            'Day_OfWeek': target_d.dayofweek,
            'Month': target_d.month
        }
        
        inp = pd.DataFrame([feats])
        for col in feature_names:
            if col not in inp.columns:
                inp[col] = 0
        inp = inp[feature_names]
        
        pred = model.predict(inp)[0]
        val = int(max(0, pred))
        
        forecast_list.append({
            'Date': target_d.strftime('%Y-%m-%d'),
            'Arrivals': val
        })

    return jsonify({
        'cutoff_date': cutoff_date.strftime('%Y-%m-%d'),
        'history': history_list,
        'forecast': forecast_list
    })

@app.route('/update_data', methods=['POST'])
def update_data():
    """
    Receives user input.
    """
    try:
        data = request.json
        date_str = data.get('date')
        arrivals = int(data.get('arrivals'))
        emergencies = int(data.get('emergencies', 0))
        
        target_date = pd.to_datetime(date_str)
        cutoff_date = get_cutoff_date()
        
        if target_date <= cutoff_date:
            return jsonify({
                'error': f"Cannot modify historical data (Date <= {cutoff_date.date()}). This system enforces strict separation."
            }), 400
            
        # 1. Append to CSV
        new_row = {
            'Date': date_str,
            'Arrivals': arrivals,
            'Emergency_Ratio': round(emergencies / arrivals, 2) if arrivals > 0 else 0,
            'Bed_Occupancy': min(1.0, (arrivals / 80.0)),
            'Avg_Wait_Time': max(5, (arrivals * 0.5)),
            'Day_OfWeek': pd.to_datetime(date_str).dayofweek,
            'Month': pd.to_datetime(date_str).month,
            'Is_Surge_GroundTruth': 0,
            'Is_FineTuned': 1 # Mark as user simulation
        }
        
        df_raw = pd.read_csv(RAW_DATA_PATH)
        
        if date_str in df_raw['Date'].values:
             df_raw = df_raw[df_raw['Date'] != date_str]
        
        df_new = pd.DataFrame([new_row])
        df_final = pd.concat([df_raw, df_new], ignore_index=True).sort_values('Date')
        df_final.to_csv(RAW_DATA_PATH, index=False)
        
        # 2. Trigger Pipeline
        print("Re-running clustering...")
        process_and_cluster() 
        
        print("Re-training model...")
        train_forecasting_model() 
        
        # 3. Reload System
        load_system()
        
        return jsonify({'status': 'success', 'message': 'Future scenario saved. Model updated to reflect new assumptions.'})

    except Exception as e:
        print(f"Update failed: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
