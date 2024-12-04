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

class ArcticDeepLearning:
    def __init__(self, data_dir="atl21_data"):
        self.data_dir = data_dir
        self.scaler = MinMaxScaler()
        
    def prepare_cnn_data(self, window_size=3):
        """Prepare data for CNN analysis"""
        files = sorted(glob.glob(os.path.join(self.data_dir, 'ATL21-01_*.h5')))
        
        all_samples = []
        all_labels = []
        
        for i in range(len(files) - window_size):
            sequence = []
            for j in range(window_size):
                with h5py.File(files[i + j], 'r') as f:
                    ssha = f['monthly/mean_ssha'][:]
                    land_mask = f['land_mask_map'][:]
                    
                    # Mask invalid data
                    valid_mask = ~np.isclose(ssha, 3.4028235e+38) & (land_mask == 0)
                    masked_ssha = np.where(valid_mask, ssha, 0)
                    
                    # Normalize the data
                    normalized_ssha = self.scaler.fit_transform(
                        masked_ssha.reshape(-1, 1)
                    ).reshape(masked_ssha.shape)
                    
                    sequence.append(normalized_ssha)
            
            # Get target (next month's SSHA)
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
    
    def prepare_lstm_data(self, sequence_length=12):
        """Prepare data for LSTM analysis"""
        files = sorted(glob.glob(os.path.join(self.data_dir, 'ATL21-01_*.h5')))
        
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
        y = []
        
        for series in grid_series.values():
            if len(series) >= sequence_length + 1:
                series.sort(key=lambda x: x[0])
                values = [v for _, v in series]
                
                for i in range(len(values) - sequence_length):
                    sequence = values[i:i + sequence_length]
                    target = values[i + sequence_length]
                    X.append(sequence)
                    y.append(target)
        
        return np.array(X), np.array(y)
    
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
    
    def build_lstm_autoencoder(self, input_shape):
        """Build LSTM autoencoder for temporal pattern analysis"""
        # Encoder
        encoder_inputs = layers.Input(shape=input_shape)
        x = layers.LSTM(128, return_sequences=True)(encoder_inputs)
        x = layers.LSTM(64, return_sequences=False)(x)
        encoded = layers.Dense(32)(x)
        
        # Decoder
        x = layers.RepeatVector(input_shape[0])(encoded)
        x = layers.LSTM(64, return_sequences=True)(x)
        x = layers.LSTM(128, return_sequences=True)(x)
        decoded = layers.TimeDistributed(
            layers.Dense(input_shape[1])
        )(x)
        
        # Combine encoder and decoder
        autoencoder = models.Model(encoder_inputs, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder
    
    def train_cnn(self, X_train, y_train, epochs=50, batch_size=32):
        """Train CNN model"""
        # Split into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        # Build and train model
        model = self.build_cnn_model(X_train.shape[1:])
        
        # Add callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5
            ),
            callbacks.ModelCheckpoint(
                filepath='best_cnn_model.h5',
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks_list
        )
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        return model, history
    
    def train_lstm_autoencoder(self, X_train, epochs=50, batch_size=32):
        """Train LSTM autoencoder"""
        # Split into train and validation
        X_train, X_val = train_test_split(X_train, test_size=0.2, random_state=42)
        
        # Build and train model
        model = self.build_lstm_autoencoder(X_train.shape[1:])
        
        # Add callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5
            ),
            callbacks.ModelCheckpoint(
                filepath='best_lstm_autoencoder.h5',
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # Train model
        history = model.fit(
            X_train, X_train,  # Autoencoder reconstructs its input
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, X_val),
            callbacks=callbacks_list
        )
        
        # Plot training history
        plt.figure(figsize=(8, 4))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Autoencoder Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        return model, history

def main():
    # Initialize
    arctic_dl = ArcticDeepLearning()
    
    # Prepare data for CNN
    print("Preparing CNN data...")
    X_cnn, y_cnn = arctic_dl.prepare_cnn_data()
    
    # Train CNN
    print("\nTraining CNN model...")
    cnn_model, cnn_history = arctic_dl.train_cnn(X_cnn, y_cnn)
    
    # Prepare data for LSTM
    print("\nPreparing LSTM data...")
    X_lstm, _ = arctic_dl.prepare_lstm_data()
    
    # Train LSTM autoencoder
    print("\nTraining LSTM autoencoder...")
    lstm_model, lstm_history = arctic_dl.train_lstm_autoencoder(X_lstm)
    
    print("\nTraining complete. Models saved as 'best_cnn_model.h5' and 'best_lstm_autoencoder.h5'")

if __name__ == "__main__":
    main()