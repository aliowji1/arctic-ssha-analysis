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
from carbon_tracker import CarbonTracker
from figure_utils import FigureManager

figure_manager = FigureManager()
carbon_tracker = CarbonTracker()

class ArcticMLAnalysis:
    def __init__(self, data_dir="atl21_data"):
        self.data_dir = data_dir
        self.scaler = StandardScaler()
        
    def load_and_preprocess_data(self):
        """Load and preprocess ATL21 data for ML analysis - optimized version"""
        files = sorted(glob.glob(os.path.join(self.data_dir, 'ATL21-01_*.h5')))
        
        all_data = []
        print(f"Found {len(files)} files to process")
        
        for idx, file_path in enumerate(files):
            print(f"Processing file {idx+1}/{len(files)}")
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
                
                # Get valid points
                valid_lats = lat[valid_mask]
                valid_lons = lon[valid_mask]
                valid_ssha = ssha[valid_mask]
                
                # Calculate distance from North Pole (vectorized)
                dist_from_pole = 90 - np.abs(valid_lats)
                
                # Simplified land proximity (approximate using nearest grid point)
                land_points = np.where(land_mask == 1)
                land_coords = np.column_stack((lat[land_points], lon[land_points]))
                
                # Calculate land proximity for batches of points
                batch_size = 1000
                land_proximity = []
                
                for i in range(0, len(valid_lats), batch_size):
                    batch_points = np.column_stack((
                        valid_lats[i:i+batch_size],
                        valid_lons[i:i+batch_size]
                    ))
                    batch_proximity = np.min(cdist(batch_points, land_coords), axis=1)
                    land_proximity.extend(batch_proximity)
                
                # Create data points
                data_points = pd.DataFrame({
                    'date': [date] * len(valid_ssha),
                    'latitude': valid_lats,
                    'longitude': valid_lons,
                    'ssha': valid_ssha,
                    'dist_from_pole': dist_from_pole,
                    'land_proximity': land_proximity,
                    'month': date.month,
                    'year': date.year
                })
                
                all_data.append(data_points)
        
        self.df = pd.concat(all_data, ignore_index=True)
        return self.df
    
    @carbon_tracker.track("kmeans_clustering")
    def perform_clustering(self, n_clusters=5, max_clusters=15, sample_size=10000):
        """Perform K-means clustering on SSHA patterns with sampling for efficiency"""
        print("Preparing features for clustering...")
        
        # Prepare features for clustering
        features = self.df[['latitude', 'longitude', 'ssha', 
                        'dist_from_pole', 'land_proximity']].copy()
        
        # Add seasonal components
        features['sin_month'] = np.sin(2 * np.pi * self.df['month'] / 12)
        features['cos_month'] = np.cos(2 * np.pi * self.df['month'] / 12)
        
        # Sample the data if it's too large
        if len(features) > sample_size:
            print(f"Sampling {sample_size} points from {len(features)} total points for initial analysis...")
            sample_idx = np.random.choice(len(features), sample_size, replace=False)
            sample_features = features.iloc[sample_idx]
        else:
            sample_features = features
        
        # Standardize features
        print("Standardizing features...")
        scaler = StandardScaler()
        scaled_sample = scaler.fit_transform(sample_features)
        
        # Find optimal number of clusters using sampled data
        print("Finding optimal number of clusters...")
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_clusters + 1)
        
        for k in k_range:
            print(f"Testing k={k}...")
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(scaled_sample)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(scaled_sample, labels))
        
        # Plot elbow curve
        fig = plt.figure(figsize=(12, 5))
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
        
        figure_manager.save_figure(fig, "elbow_curve", "kmeans")
        plt.show()
        plt.close(fig)
        
        # Perform final clustering on full dataset
        print(f"\nPerforming final clustering with {n_clusters} clusters on full dataset...")
        scaled_full = scaler.transform(features)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.df['cluster'] = self.kmeans.fit_predict(scaled_full)
        
        # Calculate clustering metrics using a sample if dataset is large
        if len(features) > sample_size:
            sample_idx = np.random.choice(len(features), sample_size, replace=False)
            silhouette_avg = silhouette_score(scaled_full[sample_idx], 
                                            self.df['cluster'].iloc[sample_idx])
        else:
            silhouette_avg = silhouette_score(scaled_full, self.df['cluster'])
        
        print(f"\nSilhouette Score (on sample): {silhouette_avg:.4f}")
        
        print("\nVisualizing clusters...")
        
        # Sample data for visualization if dataset is too large
        if len(self.df) > sample_size:
            print(f"Sampling {sample_size} points for visualization...")
            viz_sample = self.df.sample(n=sample_size, random_state=42)
        else:
            viz_sample = self.df
        
        # Cluster visualization
        fig = plt.figure(figsize=(12, 8))
        scatter = plt.scatter(viz_sample['longitude'], 
                            viz_sample['latitude'],
                            c=viz_sample['cluster'],
                            cmap='viridis',
                            alpha=0.6,
                            s=50)
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('SSHA Clusters in Arctic Region (Sampled Points)')
        
        figure_manager.save_figure(fig, "cluster_visualization", "kmeans")
        plt.show()
        plt.close(fig)

        
        return self.df['cluster']
        
    @carbon_tracker.track("random_forest_training")
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
        
        # Print statement to debug
        print("Creating importance DataFrame...")
        
        # Create the DataFrame
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': self.rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Create and save the plot
        fig = plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title('Feature Importance in SSHA Prediction')
        figure_manager.save_figure(fig, "feature_importance", "random_forest")
        plt.show()
        plt.close(fig)

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