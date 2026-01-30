# Hospital Patient Arrival Forecasting System
**Expo-Ready Machine Learning Project**

## Overview
This system predicts daily patient arrivals and identifies potential emergency surges using machine learning and clustering techniques. It is designed to help hospital administrators optimize staffing and resource allocation.

## Features
-   **Synthetic Data Generation**: Realistic simulation of daily hospital load (seasonality, weekend dips, surges).
-   **Unsupervised Clustering**: Uses DBSCAN to detect "Emergency Surges" and KMeans to classify days as "Normal" or "Low" load.
-   **Forecasting Model**: Predicts the exact number of patient arrivals for the next day.
-   **Web Dashboard**: A user-friendly interface to view predictions, status alerts (Green/Yellow/Red), and visualization of historical patterns.

## Project Structure
-   `src/data_generator.py`: Generates the synthetic dataset (`hospital_arrivals.csv`).
-   `src/clustering_pipeline.py`: Performs DBSCAN/KMeans and saves `processed_data.csv`.
-   `src/train_model.py`: Trains the XGBoost/RandomForest model (`hospital_model.pkl`).
-   `app.py`: Flask backend for the web dashboard.
-   `templates/index.html`: Dashboard UI.
-   `static/cluster_plot.png`: Visualization of historical clusters.

## How to Run

1.  **Install Dependencies**:
    ```bash
    pip install flask pandas numpy scikit-learn matplotlib seaborn xgboost
    ```

2.  **Run the App**:
    ```bash
    python app.py
    ```

3.  **Use the Dashboard**:
    -   Open your browser to `http://127.0.0.1:5000`
    -   Select a date (e.g., Tomorrow) and click "Predict".

## How it Works (For Expo Presentation)
1.  **Data**: We simulated 2 years of hospital data including seasonal trends (flu season) and random accidents.
2.  **Pattern Detection**: The system first "learns" what a normal day looks like using Clustering. Outliers are flagged as Surges.
3.  **Prediction**: An XGBoost model uses past trends (yesterday models today) to forecast future arrivals.
4.  **Actionable Insight**: The dashboard translates the number into a Color Code (Green/Yellow/Red) for easy decision making.
