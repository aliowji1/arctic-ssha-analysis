import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
import h5py
from datetime import datetime
import os
import glob

class ArcticMLAnalysis:
    def __init__(self, data_dir="atl21_data"):
        self.data_dir = data_dir
        self.scaler = StandardScaler()
        
    def load_and_preprocess_data(self):
        """Load and preprocess ATL21 data for ML analysis"""
        files = sorted(glob.glob(os.path.join(self.data_dir, 'ATL21-01_*.h5')))
        
        all_data = []
        for file_path in files:
            with h5py.File(file_path, 'r') as f:
                # Extract basic data
                lat = f['grid_lat'][:]
                lon = f['grid_lon'][:]
                ssha = f['monthly/mean_ssha'][:]
                land_mask = f['land_mask_map'][:]
                
                # Get date from filename
                date_str = os.path.basename(file_path).split('_')[1][:8]
                date = datetime.strptime(date_str, '%Y%m%d')
                
                # Create feature matrix
                valid_mask = ~np.isclose(ssha, 3.4028235e+38) & (land_mask == 0)
                valid_points = np.where(valid_mask)
                
                for i, j in zip(*valid_points):
                    # Calculate distance from North Pole
                    dist_from_pole = 90 - abs(lat[i, j])
                    
                    # Calculate proximity to land (simple version)
                    land_proximity = np.min(cdist(
                        [(lat[i, j], lon[i, j])],
                        [(lat[x, y], lon[x, y]) for x, y in zip(*np.where(land_mask == 1))]
                    ))
                    
                    data_point = {
                        'date': date,
                        'latitude': lat[i, j],
                        'longitude': lon[i, j],
                        'ssha': ssha[i, j],
                        'dist_from_pole': dist_from_pole,
                        'land_proximity': land_proximity,
                        'month': date.month,
                        'year': date.year
                    }
                    all_data.append(data_point)
        
        self.df = pd.DataFrame(all_data)
        return self.df
    
    def perform_clustering(self, n_clusters=5):
        """Perform K-means clustering on SSHA patterns"""
        # Prepare features for clustering
        features = self.df[['latitude', 'longitude', 'ssha', 'dist_from_pole', 'land_proximity']].copy()
        scaled_features = self.scaler.fit_transform(features)
        
        # Find optimal number of clusters using elbow method
        inertias = []
        silhouette_scores = []
        k_range = range(2, 11)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(scaled_features)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(scaled_features, kmeans.labels_))
        
        # Plot elbow curve
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(k_range, inertias, 'bo-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal k')
        
        plt.subplot(1, 2, 2)
        plt.plot(k_range, silhouette_scores, 'ro-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score vs Number of Clusters')
        plt.tight_layout()
        plt.show()
        
        # Perform final clustering
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.df['cluster'] = self.kmeans.fit_predict(scaled_features)
        
        # Calculate clustering metrics
        silhouette_avg = silhouette_score(scaled_features, self.df['cluster'])
        calinski_score = calinski_harabasz_score(scaled_features, self.df['cluster'])
        
        print(f"Silhouette Score: {silhouette_avg:.4f}")
        print(f"Calinski-Harabasz Score: {calinski_score:.4f}")
        
        # Visualize clusters
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(self.df['longitude'], self.df['latitude'], 
                            c=self.df['cluster'], cmap='viridis',
                            alpha=0.6, s=50)
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('SSHA Clusters in Arctic Region')
        plt.show()
        
        return self.df['cluster']
    
    def train_random_forest(self, target_months_ahead=1):
        """Train Random Forest model for SSHA prediction"""
        # Prepare features for prediction
        features = ['latitude', 'longitude', 'dist_from_pole', 
                   'land_proximity', 'month', 'year']
        
        # Create target variable (SSHA values n months ahead)
        self.df['target_ssha'] = self.df.groupby(['latitude', 'longitude'])['ssha'].shift(-target_months_ahead)
        
        # Remove rows with NaN targets
        valid_data = self.df.dropna(subset=['target_ssha'])
        
        # Split features and target
        X = valid_data[features]
        y = valid_data['target_ssha']
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train Random Forest model
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.rf_model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nRandom Forest Performance Metrics:")
        print(f"Mean Absolute Error: {mae:.4f} m")
        print(f"Root Mean Square Error: {rmse:.4f} m")
        print(f"R² Score: {r2:.4f}")
        
        # Feature importance plot
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': self.rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title('Feature Importance in SSHA Prediction')
        plt.show()
        
        return self.rf_model, (mae, rmse, r2)

def main():
    # Initialize analysis
    arctic_ml = ArcticMLAnalysis()
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = arctic_ml.load_and_preprocess_data()
    
    # Perform clustering analysis
    print("\nPerforming clustering analysis...")
    clusters = arctic_ml.perform_clustering(n_clusters=5)
    
    # Train prediction model
    print("\nTraining Random Forest model...")
    model, metrics = arctic_ml.train_random_forest(target_months_ahead=1)
    
if __name__ == "__main__":
    main()