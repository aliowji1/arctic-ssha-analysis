import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import os
from datetime import datetime
import pandas as pd
import seaborn as sns
from scipy import stats
import glob
from atl import process_atl21_file  # Import the processing function

def analyze_atl21_data(data_dir="atl21_data"):
    """Analyze all ATL21 files and create temporal visualizations"""
    # Find all ATL21-01 files
    files = sorted(glob.glob(os.path.join(data_dir, 'ATL21-01_*.h5')))
    
    if not files:
        print("No ATL21 files found")
        return
    
    print(f"Found {len(files)} files to analyze")
    
    # Process each file
    stats_list = []
    for file_path in files:
        try:
            # Get date from filename
            date_str = os.path.basename(file_path).split('_')[1][:8]
            date = datetime.strptime(date_str, '%Y%m%d')
            
            # Process file
            with h5py.File(file_path, 'r') as f:
                lat = f['grid_lat'][:]
                lon = f['grid_lon'][:]
                monthly_ssha = f['monthly/mean_ssha'][:]
                land_mask = f['land_mask_map'][:]
                
                # Mask invalid data
                fill_value = 3.4028235e+38
                invalid_mask = np.isclose(monthly_ssha, fill_value, rtol=1e-6)
                water_mask = (land_mask == 0)
                valid_data_mask = ~invalid_mask & water_mask
                valid_ssha = np.ma.masked_array(monthly_ssha, ~valid_data_mask)
                
                # Calculate statistics
                stats = {
                    'date': date,
                    'year': date.year,
                    'month': date.month,
                    'mean_ssha': np.ma.mean(valid_ssha),
                    'median_ssha': np.ma.median(valid_ssha),
                    'std_ssha': np.ma.std(valid_ssha),
                    'min_ssha': np.ma.min(valid_ssha),
                    'max_ssha': np.ma.max(valid_ssha),
                    'valid_points': np.sum(valid_data_mask)
                }
                stats_list.append(stats)
                
                print(f"Processed {date.strftime('%Y-%m')}")
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(stats_list)
    
    # Create visualizations
    create_temporal_plots(df)
    create_seasonal_analysis(df)
    print_statistics(df)

def create_temporal_plots(df):
    """Create time series plots"""
    plt.figure(figsize=(15, 10))
    
    # Plot mean SSHA with uncertainty
    plt.plot(df['date'], df['mean_ssha'], 'b-', label='Mean SSHA')
    plt.fill_between(df['date'], 
                    df['mean_ssha'] - df['std_ssha'],
                    df['mean_ssha'] + df['std_ssha'],
                    alpha=0.2, color='blue')
    
    plt.xlabel('Date')
    plt.ylabel('Sea Surface Height Anomaly (m)')
    plt.title('Sea Surface Height Anomaly Time Series')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def create_seasonal_analysis(df):
    """Create seasonal analysis plots"""
    # Monthly patterns
    plt.figure(figsize=(15, 10))
    sns.boxplot(data=df, x='month', y='mean_ssha', hue='year')
    plt.xlabel('Month')
    plt.ylabel('Mean SSHA (m)')
    plt.title('Monthly SSHA Distribution by Year')
    plt.legend(title='Year')
    plt.tight_layout()
    plt.show()
    
    # Yearly trends
    plt.figure(figsize=(12, 8))
    yearly_means = df.groupby('year')['mean_ssha'].agg(['mean', 'std'])
    plt.errorbar(yearly_means.index, yearly_means['mean'], 
                yerr=yearly_means['std'], 
                fmt='o-', capsize=5)
    plt.xlabel('Year')
    plt.ylabel('Mean SSHA (m)')
    plt.title('Annual Mean Sea Surface Height Anomaly')
    plt.grid(True)
    plt.show()

def print_statistics(df):
    """Print summary statistics"""
    print("\nSummary Statistics:")
    print("-" * 40)
    print(f"Overall Mean SSHA: {df['mean_ssha'].mean():.4f} m")
    print(f"Overall Standard Deviation: {df['mean_ssha'].std():.4f} m")
    
    # Calculate trend
    X = (df['date'] - df['date'].min()).dt.total_seconds()
    slope, intercept, r_value, p_value, std_err = stats.linregress(X, df['mean_ssha'])
    trend_per_year = slope * 365.25 * 24 * 3600
    
    print(f"\nTrend Analysis:")
    print(f"Rate of change: {trend_per_year:.6f} m/year")
    print(f"R-squared value: {r_value**2:.4f}")
    print(f"P-value: {p_value:.4e}")
    
    print("\nSeasonal Statistics:")
    seasonal_stats = df.groupby('month')['mean_ssha'].agg(['mean', 'std']).round(4)
    print(seasonal_stats)

if __name__ == "__main__":
    analyze_atl21_data()