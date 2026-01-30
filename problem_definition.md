# Hospital Patient Arrival Forecasting
## Problem Definition

**Problem Statement:**
Hospitals often face unexpected fluctuations in patient arrivals, leading to resource bottlenecks or underutilization. Allocating staff and beds efficiently requires predicting daily patient volumes and distinguishing between normal variations and emergency surges.

**Objectives:**
1.  **Predict Daily Arrivals:** Develop a machine learning model to forecast the number of incoming patients for the next day.
2.  **Identify Load Patterns:** Use unsupervised clustering (DBSCAN/KMeans) to categorize days into "Low Load", "Normal Load", and "Emergency Surge".
3.  **Visualization Dashboard:** Create a user-friendly web interface to display predictions and alert staff about potential surges.

**Assumptions:**
1.  **Synthetic Data:** The model is trained on synthetic data derived from statistical distributions, not real patient records.
2.  **Daily Granularity:** Predictions are made on a daily basis, not hourly.
3.  **Independence:** External factors like large-scale disasters are treated as statistical outliers (surges) without causal modeling.
