import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans

def process_and_cluster():
    # 1. Load Data
    input_path = "c:/Users/sriha/Documents/PWA Projects/Hospital Prediction/hospital_arrivals.csv"
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} records.")

    # 2. Feature Engineering
    # Features for Clustering: Arrivals, Emergency_Ratio, Bed_Occupancy
    cluster_features = ['Arrivals', 'Emergency_Ratio', 'Bed_Occupancy']
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[cluster_features])

    # 3. Clustering - Step 1: Anomaly Detection (DBSCAN)
    # Detect outliers (potential Emergency Surges that don't fit normal patterns)
    # eps and min_samples need tuning. 
    # Since we added random noise, we expect some density.
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    df['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)

    # -1 is outlier in DBSCAN
    df['Cluster_Label'] = 'Normal'
    df.loc[df['DBSCAN_Cluster'] == -1, 'Cluster_Label'] = 'Emergency Surge'

    print(f"DBSCAN Outliers detected: {len(df[df['Cluster_Label'] == 'Emergency Surge'])}")

    # 4. Clustering - Step 2: KMeans on 'Normal' points
    # Split 'Normal' days into 'Low Load' and 'Normal Load' based on Arrivals
    normal_indices = df[df['Cluster_Label'] == 'Normal'].index
    X_normal = X_scaled[normal_indices]

    # Use KMeans with k=2 (Low, High/Normal)
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_normal)

    # Need to map kmeans labels (0, 1) to "Low" vs "Normal"
    # We check the mean Arrivals for each cluster to identify which is which
    cluster_0_mean = df.loc[normal_indices][kmeans_labels == 0]['Arrivals'].mean()
    cluster_1_mean = df.loc[normal_indices][kmeans_labels == 1]['Arrivals'].mean()

    if cluster_0_mean < cluster_1_mean:
        mapping = {0: 'Low Load', 1: 'Normal Load'}
    else:
        mapping = {0: 'Normal Load', 1: 'Low Load'}
    
    # Assign refined labels
    refined_labels = [mapping[label] for label in kmeans_labels]
    df.loc[normal_indices, 'Cluster_Label'] = refined_labels

    print("Final Cluster Counts:")
    print(df['Cluster_Label'].value_counts())

    # Map text labels to integers for ML (optional, but good for saving)
    label_map = {'Low Load': 0, 'Normal Load': 1, 'Emergency Surge': 2}
    df['Cluster_Encoded'] = df['Cluster_Label'].map(label_map)

    # 5. Visualization
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Date', y='Arrivals', hue='Cluster_Label', 
                    palette={'Low Load': 'green', 'Normal Load': 'blue', 'Emergency Surge': 'red'},
                    s=50)
    plt.title('Daily Hospital Arrivals - Clustered Patterns')
    plt.xticks(np.arange(0, len(df), step=60), rotation=45) # Show x-axis periodically
    plt.tight_layout()
    
    # Save Plot
    plot_path = "c:/Users/sriha/Documents/PWA Projects/Hospital Prediction/static/cluster_plot.png"
    plt.savefig(plot_path)
    print(f"Cluster plot saved to {plot_path}")

    # 6. Save Processed Data
    output_path = "c:/Users/sriha/Documents/PWA Projects/Hospital Prediction/processed_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    process_and_cluster()
