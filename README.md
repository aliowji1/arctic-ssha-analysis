
# Arctic SSHA Analysis: Code Implementation and Structure

## Code Organization
This repository contains the implementation for analyzing Arctic Sea Surface Height Anomalies (SSHA) using machine learning and deep learning approaches. The codebase is organized into preprocessing and analysis components.

## Data Processing Scripts

### atl.py
The `atl.py` script handles the retrieval and initial processing of ICESat-2 ATL21 data through NASA's Earthdata API. The script implements the following functionality:
- Authentication with Earthdata Login using netrc credentials
- Automated download of ATL21 data for specified years using CMR (Common Metadata Repository) queries
- Processing of .h5 files containing gridded sea surface height measurements (448 x 304 grid)
- Conversion of raw measurements into analyze-able data formats
- Implementation of temporal statistics calculations for SSHA patterns

### arctic.py
`arctic.py` focuses on generating standardized visualizations and processed data files for analysis. Key functionalities include:
- Generation of standardized SSHA visualization maps using Basemap
- Creation of data distribution plots for quality assessment
- Production of NPZ files containing processed SSHA data, masks, and dates
- Implementation of consistent data scaling and normalization
- Generation of both visual outputs and numerical data files for CNN training

## Utility Scripts

### figure_utils.py
The `figure_utils.py` module provides a centralized figure management system through the `FigureManager` class. This utility:
- Maintains consistent figure saving across different analysis types (CNN, LSTM, K-means, Random Forest)
- Implements automatic directory structure creation
- Handles timestamp-based figure naming
- Ensures consistent DPI and format settings across visualizations
- Organizes outputs by analysis type in separate directories

### carbon_tracker.py
`carbon_tracker.py` implements emissions tracking for model training using the CodeCarbon library. Features include:
- Decorator-based tracking for easy integration with model training functions
- Automatic tracking of computational emissions
- Logging of emissions data to files
- Project-specific tracking configurations
- Integration with both traditional ML and deep learning implementations

## Analysis Scripts

### ml.py
Implements traditional machine learning approaches through the `ArcticMLAnalysis` class:
- K-means clustering for pattern identification (parameterized with n_clusters=5, max_clusters=15)
- Random Forest regression for SSHA prediction
- Feature importance analysis
- Includes data sampling for efficient processing of large datasets
- Implements both spatial and temporal feature engineering

### cnn.py
Contains the CNN implementation through the `ArcticCNN` class:
- U-Net style architecture for spatial pattern recognition
- Custom data preparation pipeline for spatial analysis
- Implements batch processing for large spatial datasets
- Features comprehensive visualization of predictions
- Includes model checkpointing and early stopping
- Handles both training and inference pipelines

### lstm.py
Implements the LSTM autoencoder through the `ArcticLSTM` class:
- Sequence-to-sequence prediction for temporal patterns
- Custom sequence preparation for temporal analysis
- Implements sliding window approach for data preparation
- Features model architecture with encoder-decoder structure
- Includes comprehensive logging of training metrics

### lstm_analysis.py
Provides detailed analysis tools for LSTM outputs through the `ArcticLSTMAnalysis` class:
- Implementation of PCA for temporal pattern identification
- Analysis of sea ice cycle relationships
- Visualization of temporal patterns and reconstructions
- Statistical analysis of identified patterns
- Investigation of seasonal and annual trends

## Usage

Each script can be run independently but follows a sequential pipeline:

1. Data Retrieval and Processing:
```python
from preprocessing.atl import download_atl21_data, process_atl21_file
from preprocessing.arctic import generate_ssha_images

# Download and process data
download_atl21_data(years=range(2018, 2025))
generate_ssha_images()
```

2. Machine Learning Analysis:
```python
from analysis.ml import ArcticMLAnalysis

analysis = ArcticMLAnalysis()
analysis.load_and_preprocess_data()
analysis.perform_clustering()
analysis.train_random_forest()
```

3. Deep Learning Analysis:
```python
from analysis.cnn import ArcticCNN
from analysis.lstm import ArcticLSTM
from analysis.lstm_analysis import ArcticLSTMAnalysis

# CNN Training and Analysis
cnn = ArcticCNN()
X_data, y_data = cnn.prepare_cnn_data()
model, history = cnn.train_cnn(X_data, y_data)

# LSTM Training
lstm = ArcticLSTM()
X_lstm, _ = lstm.prepare_lstm_data()
lstm_model, history = lstm.train_lstm_autoencoder(X_lstm)

# Pattern Analysis
analyzer = ArcticLSTMAnalysis(model_path="best_lstm_autoencoder.keras")
analyzer.visualize_temporal_patterns(X_lstm)
analyzer.analyze_arctic_processes(X_lstm)
```

## Dependencies
Required Python packages are listed in requirements.txt. Key dependencies include:
- tensorflow >= 2.8.0
- scikit-learn >= 0.24.0
- h5py >= 3.6.0
- matplotlib >= 3.4.0
- basemap >= 1.2.0
- codecarbon >= 2.1.0

## Data Requirements
- ICESat-2 ATL21 dataset access requires NASA Earthdata Login credentials
- Minimum storage requirement: 50GB for raw data
- Processed data storage: ~10GB
- RAM requirement: 16GB minimum recommended

## Project Status
This codebase is actively maintained and updated. Current focus areas include:
- Optimization of CNN architecture for better spatial prediction
- Extension of LSTM sequence analysis capabilities
- Implementation of additional validation metrics
- Integration of new environmental variables

For questions or contributions, please open an issue or contact the repository maintainers.

