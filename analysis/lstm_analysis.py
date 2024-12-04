import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy import signal
import seaborn as sns
from datetime import datetime, timedelta
import h5py
import glob
import os
from sklearn.preprocessing import MinMaxScaler
from figure_utils import FigureManager
figure_manager = FigureManager()

class ArcticLSTMAnalysis:
    def __init__(self, model_path="best_lstm_autoencoder.keras", data_dir="atl21_data"):
        self.model = tf.keras.models.load_model(model_path)
        self.data_dir = data_dir
        self.scaler = MinMaxScaler()
        self.start_date = datetime(2018, 1, 1)
    
    def prepare_data(self, sequence_length=12):
        """Prepare data for LSTM analysis"""
        files = sorted(glob.glob(os.path.join(self.data_dir, 'ATL21-01_*.h5')))
        print(f"Found {len(files)} files to process")
        
        # First collect all valid data for scaling
        print("Collecting valid data for scaling...")
        all_valid_data = []
        for file_path in files:
            with h5py.File(file_path, 'r') as f:
                ssha = f['monthly/mean_ssha'][:]
                land_mask = f['land_mask_map'][:]
                valid_mask = ~np.isclose(ssha, 3.4028235e+38) & (land_mask == 0)
                valid_data = ssha[valid_mask]
                all_valid_data.extend(valid_data)
        
        # Fit scaler
        self.scaler.fit(np.array(all_valid_data).reshape(-1, 1))
        
        # Extract time series for each grid point
        grid_series = {}
        for file_path in files:
            with h5py.File(file_path, 'r') as f:
                ssha = f['monthly/mean_ssha'][:]
                land_mask = f['land_mask_map'][:]
                date_str = os.path.basename(file_path).split('_')[1][:8]
                date = datetime.strptime(date_str, '%Y%m%d')
                
                valid_mask = ~np.isclose(ssha, 3.4028235e+38) & (land_mask == 0)
                for i, j in zip(*np.where(valid_mask)):
                    key = (i, j)
                    if key not in grid_series:
                        grid_series[key] = []
                    grid_series[key].append((date, ssha[i, j]))
        
        # Create sequences
        X = []
        locations = []
        
        for (i, j), series in grid_series.items():
            if len(series) >= sequence_length:
                series.sort(key=lambda x: x[0])
                values = [v for _, v in series]
                
                # Normalize values
                normalized_values = self.scaler.transform(np.array(values).reshape(-1, 1)).flatten()
                
                for idx in range(len(values) - sequence_length + 1):
                    sequence = normalized_values[idx:idx + sequence_length]
                    X.append(sequence)
                    locations.append((i, j))
        
        X = np.array(X)
        locations = np.array(locations)
        
        # Reshape X to include feature dimension
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        print(f"Prepared {len(X)} sequences from {len(grid_series)} grid points")
        return X, locations

    def visualize_temporal_patterns(self, X_data, num_patterns=5):
        """Visualize main temporal patterns with enhanced formatting"""
        encoder = tf.keras.Model(
            inputs=self.model.input,
            outputs=self.model.layers[2].output
        )
        
        encoded_data = encoder.predict(X_data)
        pca = PCA(n_components=num_patterns)
        patterns = pca.fit_transform(encoded_data)
        
        # Create figure with enhanced size
        fig = plt.figure(figsize=(20, 15))
        time_points = np.arange(X_data.shape[1])
        
        for i in range(num_patterns):
            plt.subplot(num_patterns, 1, i+1)
            pattern = X_data[np.argmax(np.abs(patterns[:, i]))]
            reconstruction = self.model.predict(pattern[np.newaxis, ...])[0]
            
            pattern_orig = self.scaler.inverse_transform(pattern)
            recon_orig = self.scaler.inverse_transform(reconstruction)
            
            # Enhanced line plots
            plt.plot(time_points, pattern_orig, 'b-', label='Original',
                    linewidth=3, color='blue')
            plt.plot(time_points, recon_orig, 'r--', label='Reconstructed',
                    linewidth=3, color='red', dashes=(5, 5))
            
            # Enhanced formatting
            plt.title(f'Pattern {i+1} (Explains {pca.explained_variance_ratio_[i]:.1%} of variance)',
                    fontsize=16, fontweight='bold', pad=15)
            plt.xlabel('Time (months)', fontsize=14, fontweight='bold')
            plt.ylabel('SSHA (m)', fontsize=14, fontweight='bold')
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend(fontsize=12, frameon=True)
            plt.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout(h_pad=3.0)
        figure_manager.save_figure(fig, "temporal_patterns", "lstm_analysis")
        plt.show()
        plt.close(fig)

    def analyze_arctic_processes(self, X_data):
        """Enhanced visualization of SSHA patterns vs sea ice cycle"""
        if len(X_data.shape) > 2:
            X_data = X_data.reshape(X_data.shape[0], -1)
            X_data = np.mean(X_data, axis=1)
        
        X_orig = self.scaler.inverse_transform(X_data.reshape(-1, 1)).flatten()
        
        # Create figure with white background
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 8))
        ax = fig.add_subplot(111)
        
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        monthly_means = []
        for i in range(12):
            month_data = X_orig[i::12]
            monthly_means.append(np.mean(month_data))
        monthly_means = np.array(monthly_means)
        
        # Enhanced line plot
        plt.plot(months, monthly_means, '-o', linewidth=4, markersize=10,
                color='blue', markerfacecolor='blue', markeredgecolor='blue')
        
        # Enhanced formatting
        plt.title('SSHA Pattern vs Sea Ice Cycle', 
                fontsize=24, fontweight='bold', pad=20)
        plt.xlabel('Month', fontsize=18, fontweight='bold')
        plt.ylabel('Mean SSHA (m)', fontsize=18, fontweight='bold')
        
        # Enhanced ticks
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        
        # Enhanced sea ice context
        plt.axvspan(8, 9, color='lightblue', alpha=0.3, label='Min Sea Ice (Sep)')
        plt.axvspan(2, 3, color='lightgray', alpha=0.3, label='Max Sea Ice (Mar)')
        
        # Enhanced grid
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # Enhanced legend
        plt.legend(fontsize=14, frameon=True, loc='upper right')
        
        plt.tight_layout()
        figure_manager.save_figure(fig, "sea_ice_cycle", "lstm_analysis")
        plt.show()
        plt.close(fig)
        
   
def main():
    # Initialize analyzer
    analyzer = ArcticLSTMAnalysis()
    
    # Prepare data
    print("Preparing data...")
    X_data, locations = analyzer.prepare_data()
    
    # Analyze temporal patterns
    print("\nAnalyzing temporal patterns...")
    analyzer.visualize_temporal_patterns(X_data)
    

    analyzer.analyze_arctic_processes(X_data)  # Add this line

if __name__ == "__main__":
    main()