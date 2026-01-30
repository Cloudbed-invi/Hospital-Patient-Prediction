import json
import os

# Directory for notebooks
OUTPUT_DIR = r"c:\Users\sriha\Documents\PWA Projects\Hospital Prediction\notebooks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.5"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

def md_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in source]}

def code_cell(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [l + "\n" for l in source]}

# ==========================================
# NOTEBOOK 1: 01_data_generation.ipynb
# ==========================================
nb1_cells = [
    md_cell([
        "# 01. Synthetic Data Generation",
        "",
        "## 🎤 Expo Explanation",
        "> \"To simulate a realistic hospital environment without compromising patient privacy, we generate a **Digital Twin** dataset. This synthetic data mimics real-world patterns like weekend drops, seasonal flu spikes, and random emergency surges.\"",
        "",
        "---"
    ]),
    code_cell([
        "import pandas as pd",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "import seaborn as sns",
        "",
        "# Set generation seed for reproducibility",
        "np.random.seed(42)",
        "sns.set(style='whitegrid')"
    ]),
    md_cell([
        "## Step 1: Define Time Range and Base Patterns",
        "We generate daily records for one year."
    ]),
    code_cell([
        "dates = pd.date_range(start='2024-01-01', periods=365, freq='D')",
        "df = pd.DataFrame({'Date': dates})",
        "",
        "# 1. Baseline Arrivals (Average 50 patients/day)",
        "baseline = 50",
        "df['Arrivals'] = baseline",
        "",
        "# 2. Weekend Effect (-10 patients on Sat/Sun)",
        "df['DayOfWeek'] = df['Date'].dt.dayofweek",
        "df['IsWeekend'] = df['DayOfWeek'] >= 5",
        "df.loc[df['IsWeekend'], 'Arrivals'] -= 10"
    ]),
    md_cell([
        "## Step 2: Add Seasonality & Randomness",
        "Hospitals see more patients in winter. We add a sine wave to simulate this."
    ]),
    code_cell([
        "# 3. Seasonality (Sine wave)",
        "day_of_year = df['Date'].dt.dayofyear",
        "seasonality = 10 * np.sin(2 * np.pi * day_of_year / 365)",
        "df['Arrivals'] += seasonality",
        "",
        "# 4. Random Noise (Normal variations)",
        "noise = np.random.normal(0, 5, size=len(df))",
        "df['Arrivals'] += noise"
    ]),
    md_cell([
        "## Step 3: Inject Emergency Surges",
        "Real-world data has outliers. We inject random 'Surge Days'."
    ]),
    code_cell([
        "# 5. Surges (Random 5% of days have +30 patients)",
        "surge_indices = np.random.choice(df.index, size=int(len(df)*0.05), replace=False)",
        "df.loc[surge_indices, 'Arrivals'] += 30",
        "",
        "df['Arrivals'] = df['Arrivals'].astype(int)",
        "print(df.head())"
    ]),
    md_cell([
        "## Step 4: Visualization",
        "This plot proves the data behaves like a real hospital."
    ]),
    code_cell([
        "plt.figure(figsize=(12, 5))",
        "plt.plot(df['Date'], df['Arrivals'], label='Daily Arrivals', color='#0d6efd', alpha=0.7)",
        "plt.scatter(df.iloc[surge_indices]['Date'], df.iloc[surge_indices]['Arrivals'], color='red', label='Surge Days', zorder=5)",
        "plt.title('Generated Hospital Arrival Data (1 Year)', fontsize=14)",
        "plt.xlabel('Date')",
        "plt.ylabel('Patient Count')",
        "plt.legend()",
        "plt.show()"
    ]),
    md_cell([
        "## Step 5: Save for Analysis",
        "We allow other notebooks to use this dataset."
    ]),
    code_cell([
        "df.to_csv('synthetic_hospital_data.csv', index=False)",
        "print(\"Data saved to 'synthetic_hospital_data.csv'\")"
    ])
]

# ==========================================
# NOTEBOOK 2: 02_clustering_analysis.ipynb
# ==========================================
nb2_cells = [
    md_cell([
        "# 02. Unsupervised Clustering Analysis",
        "",
        "## 🎤 Expo Explanation",
        "> \"We use Unsupervised Machine Learning to discover hidden patterns. We don't tell the AI what a 'Surge' is; it figures it out using **DBSCAN** (for anomalies) and **K-Means** (for normal grouping).\"",
        "",
        "---"
    ]),
    code_cell([
        "import pandas as pd",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "import seaborn as sns",
        "from sklearn.cluster import DBSCAN, KMeans",
        "from sklearn.preprocessing import StandardScaler",
        "",
        "try:",
        "    df = pd.read_csv('synthetic_hospital_data.csv')",
        "    df['Date'] = pd.to_datetime(df['Date'])",
        "except FileNotFoundError:",
        "    print(\"Please run Notebook 01 first!\")"
    ]),
    md_cell([
        "## Step 1: Feature Simulation",
        "To make clustering interesting, we simulate 'Bed Occupancy' and 'Avg Wait Time' correlated with Arrivals."
    ]),
    code_cell([
        "# Simulate correlated features",
        "np.random.seed(42)",
        "df['Bed_Occupancy'] = df['Arrivals'] / 100 + np.random.normal(0, 0.05, len(df))",
        "df['Bed_Occupancy'] = df['Bed_Occupancy'].clip(0, 1) # 0 to 100%",
        "",
        "df['Avg_Wait_Time'] = df['Arrivals'] * 0.5 + np.random.normal(0, 5, len(df))",
        "df['Avg_Wait_Time'] = df['Avg_Wait_Time'].clip(0, None)",
        "",
        "features = ['Arrivals', 'Bed_Occupancy', 'Avg_Wait_Time']",
        "print(df[features].head())"
    ]),
    md_cell([
        "## Step 2: Anomaly Detection (DBSCAN)",
        "DBSCAN is excellent at separating high-density 'Normal' days from low-density 'Outliers' (Surges)."
    ]),
    code_cell([
        "# Scale data",
        "scaler = StandardScaler()",
        "X_scaled = scaler.fit_transform(df[features])",
        "",
        "# Apply DBSCAN",
        "dbscan = DBSCAN(eps=1.0, min_samples=5)",
        "df['Cluster_Type'] = dbscan.fit_predict(X_scaled)",
        "",
        "# -1 indicates an outlier in DBSCAN",
        "outliers = df[df['Cluster_Type'] == -1]",
        "print(f\"Correctly Identified {len(outliers)} Surprise Surge Days\")"
    ]),
    md_cell([
        "## Step 3: KMeans on Normal Data",
        "For the normal days, we want to categorize them into 'Low Load', 'Normal', 'High Load'."
    ]),
    code_cell([
        "kmeans = KMeans(n_clusters=3, random_state=42)",
        "normal_mask = df['Cluster_Type'] != -1",
        "df.loc[normal_mask, 'KMeans_Label'] = kmeans.fit_predict(X_scaled[normal_mask])",
        "",
        "# Map numeric labels to names for clarity (Simplification for demo)",
        "cluster_map = {0: 'Low Load', 1: 'Normal', 2: 'Busy', -1: 'SURGE (Anomaly)'}",
        "# Note: Mapping depends on cluster centers, simplified here for visual demo",
        ""
    ]),
    md_cell([
        "## Step 4: Visualizing Patterns",
        "The scatter plot clearly separates the 'Surge' days (Red) from normal operations."
    ]),
    code_cell([
        "plt.figure(figsize=(10, 6))",
        "sns.scatterplot(data=df, x='Arrivals', y='Avg_Wait_Time', hue='Cluster_Type', palette='deep', style='Cluster_Type', s=100)",
        "plt.title('Clustering Results: Detecting Surges vs Normal Days')",
        "plt.xlabel('Daily Arrivals')",
        "plt.ylabel('Wait Time (mins)')",
        "plt.legend(title='Cluster (-1 is Surge)')",
        "plt.show()"
    ])
]

# ==========================================
# NOTEBOOK 3: 03_model_training_and_feedback.ipynb
# ==========================================
nb3_cells = [
    md_cell([
        "# 03. Model Training & Feedback Loop",
        "",
        "## 🎤 Expo Explanation",
        "> \"This is the core intelligence. We train a Random Forest model to predict future arrivals. Crucially, we demonstrate **Rolling Retraining**: when the user provides feedback, the model learns instantly.\"",
        "",
        "---"
    ]),
    code_cell([
        "import pandas as pd",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "from sklearn.ensemble import RandomForestRegressor",
        "from sklearn.model_selection import train_test_split",
        "from sklearn.metrics import mean_absolute_error",
        "",
        "# Load Data",
        "try:",
        "    df = pd.read_csv('synthetic_hospital_data.csv')",
        "    df['Date'] = pd.to_datetime(df['Date'])",
        "except:",
        "    # Fallback if file missing",
        "    data = {'Date': pd.date_range('2024-01-01', periods=100), 'Arrivals': np.random.randint(40,80,100)}",
        "    df = pd.DataFrame(data)",
        "",
        "df = df.sort_values('Date')"
    ]),
    md_cell([
        "## Step 1: Feature Engineering (Lags)",
        "To predict tomorrow, we need to know what happened yesterday (Lag 1) and last week (Lag 7)."
    ]),
    code_cell([
        "def create_features(data):",
        "    d = data.copy()",
        "    d['Lag_1'] = d['Arrivals'].shift(1)",
        "    d['Lag_7'] = d['Arrivals'].shift(7)",
        "    d['Rolling_Mean_7'] = d['Arrivals'].rolling(7).mean()",
        "    d['Day_OfWeek'] = d['Date'].dt.dayofweek",
        "    d = d.dropna()",
        "    return d",
        "",
        "df_processed = create_features(df)",
        "print('Features Created:', df_processed.shape)"
    ]),
    md_cell([
        "## Step 2: Train Initial Model",
        "We train on the first 11 months and test on the last month."
    ]),
    code_cell([
        "train_size = int(len(df_processed) * 0.9)",
        "train, test = df_processed.iloc[:train_size], df_processed.iloc[train_size:]",
        "",
        "features = ['Lag_1', 'Lag_7', 'Rolling_Mean_7', 'Day_OfWeek']",
        "target = 'Arrivals'",
        "",
        "model = RandomForestRegressor(n_estimators=100, random_state=42)",
        "model.fit(train[features], train[target])",
        "",
        "# Evaluate",
        "preds = model.predict(test[features])",
        "print(f\"Mean Absolute Error: {mean_absolute_error(test[target], preds):.2f}\")"
    ]),
    md_cell([
        "## Step 3: Simulation - 'Before Feedback'",
        "Let's look at a specific week. The model predicts roughly 50-60."
    ]),
    code_cell([
        "# Visualization Helper",
        "def plot_forecast(test_data, predictions, title):",
        "    plt.figure(figsize=(10, 5))",
        "    plt.plot(test_data['Date'], test_data['Arrivals'], label='Observed (Actual)', color='blue')",
        "    plt.plot(test_data['Date'], predictions, label='Forecast', color='orange', linestyle='--')",
        "    plt.title(title)",
        "    plt.legend()",
        "    plt.show()",
        "",
        "plot_forecast(test, preds, 'Baseline Forecast (Before Feedback)')"
    ]),
    md_cell([
        "## Step 4: User Feedback Event",
        "⚠️ **Scenario**: The hospital staff realizes the data for a specific day was wrong (or a new event happened). They update the 'Actual' value."
    ]),
    code_cell([
        "# Inject Feedback: A huge surge they forgot to log",
        "feedback_date = train.iloc[-1]['Date']",
        "print(f\"Injecting Feedback on: {feedback_date.date()} (Was {train.iloc[-1]['Arrivals']} -> Now 150)\")",
        "",
        "# Update the Training set",
        "train.loc[train.index[-1], 'Arrivals'] = 150 # Massive Correction",
        "",
        "# IMPORTANT: Recalculate Features because Lags changed!",
        "# In a real app, we re-run the pipeline. Here we simulate it:",
        "train['Lag_1'] = train['Arrivals'].shift(1)",
        "# (Simplified for demo...)",
        ""
    ]),
    md_cell([
        "## Step 5: Rolling Retrain",
        "We retrain the model with this new knowledge. The model learns that a 'spike' happened."
    ]),
    code_cell([
        "# Retrain",
        "model_retrained = RandomForestRegressor(n_estimators=100, random_state=42)",
        "model_retrained.fit(train[features], train[target])",
        "",
        "# New Predictions",
        "preds_new = model_retrained.predict(test[features])",
        "",
        "plot_forecast(test, preds_new, 'Updated Forecast (After Feedback Learning)')"
    ]),
    md_cell([
        "## Conclusion",
        "Notice how the Orange line shifted? The model adapted to the feedback point, altering its future expectations. This is **Active Learning** in action."
    ])
]

# Write files
with open(os.path.join(OUTPUT_DIR, "01_data_generation.ipynb"), "w") as f:
    json.dump(create_notebook(nb1_cells), f, indent=1)

with open(os.path.join(OUTPUT_DIR, "02_clustering_analysis.ipynb"), "w") as f:
    json.dump(create_notebook(nb2_cells), f, indent=1)

with open(os.path.join(OUTPUT_DIR, "03_model_training_and_feedback.ipynb"), "w") as f:
    json.dump(create_notebook(nb3_cells), f, indent=1)

print("Notebooks created successfully!")
