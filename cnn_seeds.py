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

class ArcticCNN:
    def __init__(self, data_dir="atl21_data"):
        self.data_dir = data_dir
        self.scaler = MinMaxScaler()
    
    def prepare_cnn_data(self, window_size=3):
        """Prepare data for CNN analysis"""
        files = sorted(glob.glob(os.path.join(self.data_dir, 'ATL21-01_*.h5')))
        print(f"Found {len(files)} files to process")
        
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
                    
                    normalized_ssha = self.scaler.fit_transform(
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
            all_labels.append(normalized_target)
        
        return np.array(all_samples), np.array(all_labels)
    
    def build_cnn_model(self, input_shape):
        """Build CNN model for SSHA prediction"""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                         input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            
            layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same'),
            layers.BatchNormalization(),
            
            layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same'),
            layers.BatchNormalization(),
            
            layers.Conv2D(1, (1, 1), activation='linear')
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def train_with_multiple_seeds(self, X_train, y_train, seeds=[42, 123, 456, 789, 101112], 
                                epochs=20, batch_size=32):
        """Train CNN model with multiple random seeds for train/val split"""
        plt.figure(figsize=(15, 10))
        
        for idx, seed in enumerate(seeds, 1):
            print(f"\nTraining with seed {seed}")
            
            # Split data with current seed
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train, y_train, test_size=0.2, random_state=seed
            )
            
            # Build and train model
            model = self.build_cnn_model(X_train.shape[1:])
            
            callbacks_list = [
                callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5
                ),
                callbacks.ModelCheckpoint(
                    filepath=f'best_cnn_model_seed_{seed}.keras',
                    monitor='val_loss',
                    save_best_only=True
                )
            ]
            
            history = model.fit(
                X_train_split, y_train_split,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val_split, y_val_split),
                callbacks=callbacks_list,
                verbose=1
            )
            
            # Plot loss
            plt.subplot(2, len(seeds), idx)
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title(f'Model Loss (Seed {seed})')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            # Plot MAE
            plt.subplot(2, len(seeds), len(seeds) + idx)
            plt.plot(history.history['mae'], label='Training MAE')
            plt.plot(history.history['val_mae'], label='Validation MAE')
            plt.title(f'Model MAE (Seed {seed})')
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
            plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        return model, history

def main():
    arctic_cnn = ArcticCNN()
    
    print("Preparing CNN data...")
    X_cnn, y_cnn = arctic_cnn.prepare_cnn_data()
    
    print("\nTraining CNN model with multiple random seeds...")
    model, history = arctic_cnn.train_with_multiple_seeds(X_cnn, y_cnn)
    
    print("\nTraining complete. Models saved as 'best_cnn_model_seed_*.keras'")

if __name__ == "__main__":
    main()