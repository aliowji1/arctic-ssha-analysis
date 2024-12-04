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
from figure_utils import FigureManager
from carbon_tracker import CarbonTracker

figure_manager = FigureManager()
carbon_tracker = CarbonTracker()

class ArcticLSTM:
    def __init__(self, data_dir="atl21_data"):
        self.data_dir = data_dir
        self.scaler = MinMaxScaler()
    
    def prepare_lstm_data(self, sequence_length=12):
        """Prepare data for LSTM analysis"""
        files = sorted(glob.glob(os.path.join(self.data_dir, 'ATL21-01_*.h5')))
        print(f"Found {len(files)} files to process")
        
        grid_series = {}
        
        for idx, file_path in enumerate(files):
            print(f"Processing file {idx+1}/{len(files)}")
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
        
        print("Creating sequences...")
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
        
        X = np.array(X)
        y = np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        return X, y
    
    def build_lstm_autoencoder(self, input_shape):
        """Build LSTM autoencoder for temporal pattern analysis"""
        encoder_inputs = layers.Input(shape=input_shape)
        x = layers.LSTM(128, return_sequences=True)(encoder_inputs)
        x = layers.LSTM(64, return_sequences=False)(x)
        encoded = layers.Dense(32)(x)
        
        x = layers.RepeatVector(input_shape[0])(encoded)
        x = layers.LSTM(64, return_sequences=True)(x)
        x = layers.LSTM(128, return_sequences=True)(x)
        decoded = layers.TimeDistributed(
            layers.Dense(input_shape[1])
        )(x)
        
        autoencoder = models.Model(encoder_inputs, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder
    @carbon_tracker.track("lstm_training")
    def train_lstm_autoencoder(self, X_train, epochs=20, batch_size=32):
        """Train LSTM autoencoder"""
        X_train, X_val = train_test_split(X_train, test_size=0.2, random_state=42)
        
        model = self.build_lstm_autoencoder(X_train.shape[1:])
        
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5
            ),
            callbacks.ModelCheckpoint(
                filepath='best_lstm_autoencoder.keras',
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        history = model.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, X_val),
            callbacks=callbacks_list
        )
        
        # Create figure first
        fig = plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
        plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        plt.title('LSTM Autoencoder Loss', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Epoch', fontsize=12, fontweight='bold')
        plt.ylabel('Loss', fontsize=12, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10, frameon=True, facecolor='white', edgecolor='black')
        plt.tight_layout()

        # Save the figure
        figure_manager.save_figure(fig, "training_loss", "lstm")
        plt.show()
        plt.close(fig)
        
        return model, history

def main():
    arctic_lstm = ArcticLSTM()
    
    print("Preparing LSTM data...")
    X_lstm, _ = arctic_lstm.prepare_lstm_data()
    
    print("\nTraining LSTM autoencoder...")
    lstm_model, lstm_history = arctic_lstm.train_lstm_autoencoder(X_lstm)
    
    print("\nTraining complete. Model saved as 'best_lstm_autoencoder.keras'")

if __name__ == "__main__":
    main()