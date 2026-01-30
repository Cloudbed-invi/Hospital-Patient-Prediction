import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_hospital_data(start_date, end_date):
    """
    Generates synthetic hospital arrival data with:
    - Weekly patterns (higher on weekdays, lower on weekends)
    - Seasonal patterns (higher in winter)
    - Random noise
    - Outliers (Emergency Surges)
    """
    date_range = pd.date_range(start=start_date, end=end_date)
    data = []

    for date in date_range:
        # 1. Base Demand (Trend)
        # Slight upward trend over time
        day_index = (date - pd.to_datetime(start_date)).days
        base_demand = 50 + (day_index * 0.01)

        # 2. Weekly Pattern
        # Monday(0) is busy, Sunday(6) is quiet
        day_of_week = date.dayofweek
        if day_of_week == 0:  # Mon
            day_factor = 1.2
        elif day_of_week == 5: # Sat
            day_factor = 0.8
        elif day_of_week == 6: # Sun
            day_factor = 0.7
        else:
            day_factor = 1.0
        
        # 3. Seasonal Pattern
        # Higher in Winter (Dec-Feb), lower in Summer
        month = date.month
        if month in [12, 1, 2]:
            season_factor = 1.15
        elif month in [6, 7, 8]:
            season_factor = 0.90
        else:
            season_factor = 1.0

        # Calculate Expected Arrivals
        expected_arrivals = base_demand * day_factor * season_factor

        # 4. Random Noise
        arrivals = int(np.random.normal(expected_arrivals, scale=5))

        # 5. Emergency Surge Injection (Outliers)
        # 2% chance of a major accident/surge
        is_surge = False
        if random.random() < 0.02:
            arrivals += int(np.random.uniform(20, 50))
            is_surge = True

        # Ensure no negative or zero
        arrivals = max(10, arrivals)

        # Additional Features
        # Emergency Ratio: How many of these were emergencies?
        # Correlated with surge
        if is_surge:
            emergency_ratio = np.random.uniform(0.6, 0.9)
        else:
            emergency_ratio = np.random.uniform(0.1, 0.4)
        
        # Bed Occupancy (correlated with arrivals)
        # Assume capacity is around 100
        bed_occupancy = min(1.0, (arrivals / 80.0) * np.random.uniform(0.9, 1.1))

        # Waiting Time (correlated with arrivals/occupancy)
        avg_wait_time = (arrivals * 0.5) * (1 + bed_occupancy) 
        avg_wait_time += np.random.normal(0, 5)
        avg_wait_time = max(5, avg_wait_time)

        data.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Arrivals': arrivals,
            'Emergency_Ratio': round(emergency_ratio, 2),
            'Bed_Occupancy': round(bed_occupancy, 2),
            'Avg_Wait_Time': round(avg_wait_time, 1),
            'Day_OfWeek': day_of_week,
            'Month': month,
            'Is_Surge_GroundTruth': 1 if is_surge else 0,
            'Is_FineTuned': 0 # 0 = Original, 1 = FineTuned
        })

    return pd.DataFrame(data)

if __name__ == "__main__":
    print("Generating synthetic hospital data...")
    # Generate data covering past up to 2026
    start = "2024-01-01"
    end = "2026-02-28"
    
    df = generate_hospital_data(start, end)
    
    # Save to CSV
    output_path = "c:/Users/sriha/Documents/PWA Projects/Hospital Prediction/hospital_arrivals.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Data generated: {len(df)} records.")
    print(df.head())
    print(f"Saved to {output_path}")
