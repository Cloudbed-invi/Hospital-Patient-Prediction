import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
try:
    from xgboost import XGBRegressor
    MODEL_TYPE = 'XGBoost'
except ImportError:
    MODEL_TYPE = 'RandomForest'

def train_forecasting_model():
    # 1. Load Processed Data
    df = pd.read_csv("c:/Users/sriha/Documents/PWA Projects/Hospital Prediction/processed_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    # STRICT FILTERING: TRAIN ONLY ON HISTORICAL DATA <= TODAY
    # System time enforced
    cutoff_date = pd.to_datetime(datetime.now().date())
    print(f"Training Cutoff Date: {cutoff_date}")
    
    df = df[df['Date'] <= cutoff_date]
    print(f"Dataset size after filtering (<= {cutoff_date}): {len(df)}")

    # 2. Feature Engineering
    df['Target_NextDay_Arrivals'] = df['Arrivals'].shift(-1)
    
    for lag in [1, 2, 3, 7]:
        df[f'Lag_{lag}'] = df['Arrivals'].shift(lag)
    
    df['Rolling_Mean_7'] = df['Arrivals'].rolling(window=7).mean()

    df = df.dropna()

    features = [
        'Lag_1', 'Lag_2', 'Lag_3', 'Lag_7', 
        'Rolling_Mean_7', 
        'Day_OfWeek', 'Month',
        'Cluster_Encoded', 
        'Bed_Occupancy',
        'Avg_Wait_Time'
    ]
    
    X = df[features]
    y = df['Target_NextDay_Arrivals']

    if len(X) == 0:
        print("Error: No data available for training after filtering!")
        return

    # 3. Train/Test Split
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Training with {len(X_train)} samples, Testing with {len(X_test)} samples.")
    print(f"Model Type: {MODEL_TYPE}")

    # 4. Model Training
    if MODEL_TYPE == 'XGBoost':
        model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

    model.fit(X_train, y_train)

    # 5. Evaluation
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print(f"Model Performance:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    # 6. Save Model
    model_data = {
        'model': model,
        'features': features,
        'mae': mae
    }
    
    output_path = "c:/Users/sriha/Documents/PWA Projects/Hospital Prediction/models/hospital_model.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    train_forecasting_model()
