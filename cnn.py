import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
import h5py
from datetime import datetime
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from carbon_tracker import CarbonTracker
from figure_utils import FigureManager

figure_manager = FigureManager()

carbon_tracker = CarbonTracker()

class ArcticCNN:
    def __init__(self, data_dir="atl21_data"):
        self.data_dir = data_dir
        self.scaler = MinMaxScaler()
        
    def prepare_cnn_data(self, window_size=3):
        """Prepare data for CNN analysis with consistent scaling"""
        files = sorted(glob.glob(os.path.join(self.data_dir, 'ATL21-01_*.h5')))
        print(f"Found {len(files)} files to process")
        
        # First pass: collect all valid data for fitting the scaler
        print("Collecting valid data for scaling...")
        all_valid_data = []
        for file_path in files:
            with h5py.File(file_path, 'r') as f:
                ssha = f['monthly/mean_ssha'][:]
                land_mask = f['land_mask_map'][:]
                valid_mask = ~np.isclose(ssha, 3.4028235e+38) & (land_mask == 0)
                valid_data = ssha[valid_mask]
                all_valid_data.extend(valid_data)
        
        # Fit scaler once on all valid data
        print("Fitting scaler...")
        self.scaler.fit(np.array(all_valid_data).reshape(-1, 1))
        print(f"Data range for scaling: {np.min(all_valid_data):.3f} to {np.max(all_valid_data):.3f}")
        
        # Second pass: create sequences using consistent scaling
        all_samples = []
        all_labels = []
        
        for i in range(len(files) - window_size):
            print(f"Processing time window {i+1}/{len(files)-window_size}")
            sequence = []
            for j in range(window_size):
                with h5py.File(files[i + j], 'r') as f:
                    ssha = f['monthly/mean_ssha'][:]
                    land_mask = f['land_mask_map'][:]
                    
                    valid_mask = ~np.isclose(ssha, 3.4028235e+38) & (land_mask == 0)
                    masked_ssha = np.where(valid_mask, ssha, 0)
                    
                    # Transform using pre-fitted scaler
                    normalized_ssha = self.scaler.transform(
                        masked_ssha.reshape(-1, 1)
                    ).reshape(masked_ssha.shape)
                    
                    sequence.append(normalized_ssha)
            
            with h5py.File(files[i + window_size], 'r') as f:
                target_ssha = f['monthly/mean_ssha'][:]
                target_mask = ~np.isclose(target_ssha, 3.4028235e+38) & (land_mask == 0)
                target_ssha = np.where(target_mask, target_ssha, 0)
                normalized_target = self.scaler.transform(
                    target_ssha.reshape(-1, 1)
                ).reshape(target_ssha.shape)
            
            all_samples.append(np.stack(sequence, axis=-1))
            all_labels.append(normalized_target[..., np.newaxis])  # Add channel dimension
        
        X = np.array(all_samples)
        y = np.array(all_labels)
        print(f"Final data shapes - X: {X.shape}, y: {y.shape}")
        return X, y
    
    def build_cnn_model(self, input_shape):
        """Build CNN model for SSHA prediction with improved architecture"""
        model = models.Sequential([
            # Encoder path
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', 
                        input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Bridge
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            
            # Decoder path
            layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            
            layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            
            # Final prediction
            layers.Conv2D(1, (1, 1), activation='linear')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        return model
    @carbon_tracker.track("cnn_training")
    def train_cnn(self, X_train, y_train, epochs=50, batch_size=32):
        """Train CNN model with enhanced visualization"""
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        model = self.build_cnn_model(X_train.shape[1:])
        
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5
            ),
            callbacks.ModelCheckpoint(
                filepath='best_cnn_model.keras',
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks_list
        )
        
        # Create enhanced training plots
        fig = plt.figure(figsize=(20, 8))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], 'b-', label='Training Loss', 
                linewidth=3)
        plt.plot(history.history['val_loss'], 'r-', label='Validation Loss', 
                linewidth=3)
        plt.title('Model Loss', fontsize=20, fontweight='bold', pad=15)
        plt.xlabel('Epoch', fontsize=16, fontweight='bold')
        plt.ylabel('Loss', fontsize=16, fontweight='bold')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=14, frameon=True)
        plt.grid(True, linestyle='--', alpha=0.3)

        # MAE plot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], 'b-', label='Training MAE', 
                linewidth=3)
        plt.plot(history.history['val_mae'], 'r-', label='Validation MAE', 
                linewidth=3)
        plt.title('Model MAE', fontsize=20, fontweight='bold', pad=15)
        plt.xlabel('Epoch', fontsize=16, fontweight='bold')
        plt.ylabel('MAE', fontsize=16, fontweight='bold')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=14, frameon=True)
        plt.grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()
        figure_manager.save_figure(fig, "training_metrics", "cnn")
        plt.show()
        plt.close(fig)
        
        return model, history
    def visualize_predictions(self, model, X_test, y_test, num_samples=None):
        """Visualize CNN predictions with enhanced formatting and clear documentation"""
        if num_samples is None:
            num_samples = len(X_test)
        
        # Make predictions
        predictions = model.predict(X_test[:num_samples])
        
        # Convert normalized values back to original scale
        predictions_original = self.scaler.inverse_transform(
            predictions.reshape(-1, 1)).reshape(predictions.shape)
        y_test_original = self.scaler.inverse_transform(
            y_test[:num_samples].reshape(-1, 1)).reshape(y_test[:num_samples].shape)
        
        # Process each sample
        for i in range(num_samples):
            # Create mask for valid data points
            mask = np.abs(y_test_original[i, :, :, 0]) > 1e-6
            
            # Create masked arrays
            pred_masked = np.where(mask, predictions_original[i, :, :, 0], np.nan)
            actual_masked = np.where(mask, y_test_original[i, :, :, 0], np.nan)
            
            # Create figure with enhanced size and formatting
            fig = plt.figure(figsize=(20, 6))
            
            # Plot actual SSHA
            ax1 = plt.subplot(1, 3, 1)
            im1 = plt.imshow(actual_masked, cmap='coolwarm', vmin=-0.4, vmax=0.4)
            cbar1 = plt.colorbar(im1)
            cbar1.set_label('SSHA (m)', fontsize=14, fontweight='bold')
            cbar1.ax.tick_params(labelsize=12)
            plt.title('Actual SSHA - Sample {}'.format(i+1), 
                    fontsize=16, fontweight='bold', pad=15)
            plt.xlabel('Longitude', fontsize=14, fontweight='bold')
            plt.ylabel('Latitude', fontsize=14, fontweight='bold')
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            
            # Plot predicted SSHA
            ax2 = plt.subplot(1, 3, 2)
            im2 = plt.imshow(pred_masked, cmap='coolwarm', vmin=-0.4, vmax=0.4)
            cbar2 = plt.colorbar(im2)
            cbar2.set_label('SSHA (m)', fontsize=14, fontweight='bold')
            cbar2.ax.tick_params(labelsize=12)
            plt.title('Predicted SSHA - Sample {}'.format(i+1), 
                    fontsize=16, fontweight='bold', pad=15)
            plt.xlabel('Longitude', fontsize=14, fontweight='bold')
            plt.ylabel('Latitude', fontsize=14, fontweight='bold')
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            
            # Plot difference
            diff = pred_masked - actual_masked
            ax3 = plt.subplot(1, 3, 3)
            im3 = plt.imshow(diff, cmap='RdBu', vmin=-0.2, vmax=0.2)
            cbar3 = plt.colorbar(im3)
            cbar3.set_label('Difference (m)', fontsize=14, fontweight='bold')
            cbar3.ax.tick_params(labelsize=12)
            mean_error = np.nanmean(np.abs(diff))
            plt.title('Prediction Error - Sample {}\nMean Error: {:.3f}m'.format(i+1, mean_error), 
                    fontsize=16, fontweight='bold', pad=15)
            plt.xlabel('Longitude', fontsize=14, fontweight='bold')
            plt.ylabel('Latitude', fontsize=14, fontweight='bold')
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            
            # Adjust layout
            plt.tight_layout(w_pad=3.0)
            
            # Save and show figure
            figure_manager.save_figure(fig, f"prediction_sample_{i+1}", "cnn")
            plt.show()
            plt.close(fig)
            
            # Calculate and print statistics
            valid_diff = diff[~np.isnan(diff)]
            print(f"\nSample {i+1} Statistics:")
            print(f"Mean Absolute Error: {np.nanmean(np.abs(diff)):.3f} m")
            print(f"Root Mean Square Error: {np.sqrt(np.nanmean(valid_diff**2)):.3f} m")
            print(f"Maximum Error: {np.nanmax(np.abs(valid_diff)):.3f} m")

def main():
    arctic_cnn = ArcticCNN()
    
    print("Preparing data for CNN analysis...")
    X_data, y_data = arctic_cnn.prepare_cnn_data()
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=42
    )
    
    # Train the model
    print("\nTraining CNN model...")
    model, history = arctic_cnn.train_cnn(X_train, y_train)
    
    # Generate predictions and visualizations
    print("\nGenerating prediction visualizations...")
    arctic_cnn.visualize_predictions(model, X_test, y_test)
    
    print("\nVisualization complete!")

if __name__ == "__main__":
    main()