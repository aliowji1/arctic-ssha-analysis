import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import os
from datetime import datetime
import glob
import requests
import netrc
import subprocess

def download_atl21_data(years, output_dir="atl21_data"):
    """Download ATL21 data for specified years"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up Earthdata authentication
    netrc_path = os.path.expanduser("~/.netrc")
    if not os.path.exists(netrc_path):
        username = input("Earthdata Login username: ")
        password = getpass.getpass("Earthdata Login password: ")
        with open(netrc_path, 'w') as f:
            f.write(f'machine urs.earthdata.nasa.gov login {username} password {password}\n')
        os.chmod(netrc_path, 0o600)
    
    # Search and download for each year
    session = requests.Session()
    
    for year in years:
        print(f"\nProcessing year {year}")
        
        # Search for granules
        search_url = "https://cmr.earthdata.nasa.gov/search/granules.json"
        params = {
            'short_name': 'ATL21',
            'version': '003',
            'temporal': f'{year}-01-01T00:00:00Z,{year}-12-31T23:59:59Z',
            'page_size': 2000,
            'provider': 'NSIDC_ECS'
        }
        
        response = session.get(search_url, params=params)
        response.raise_for_status()
        
        results = response.json()
        granules = results.get('feed', {}).get('entry', [])
        
        print(f"Found {len(granules)} granules for {year}")
        
        for granule in granules:
            download_urls = [l['href'] for l in granule.get('links', [])
                           if 'href' in l and 'data#' in l.get('rel', '')]
            
            for url in download_urls:
                if '.h5' in url:
                    filename = os.path.join(output_dir, os.path.basename(url))
                    if not os.path.exists(filename):
                        print(f"Downloading {os.path.basename(url)}")
                        subprocess.run([
                            "curl",
                            "-b", "~/.urs_cookies",
                            "-c", "~/.urs_cookies",
                            "-L",
                            "-n",
                            "-o", filename,
                            url
                        ], check=True)
                        print(f"Successfully downloaded {filename}")

def process_atl21_file(file_path):
    """Process a single ATL21 file and return processed data"""
    # [Previous process_atl21_file function code remains the same]
    
def plot_monthly_data(data):
    """Plot SSHA map, distribution, and extreme values for a single month"""
    # [Previous plot_monthly_data function code remains the same]

def main():
    # Define years to process
    years = range(2018, 2025)
    
    # Download data if needed
    download_atl21_data(years)
    
    # Process all files
    data_dir = 'atl21_data'
    files = sorted(glob.glob(os.path.join(data_dir, 'ATL21-01_*.h5')))
    
    if not files:
        print("No ATL21 files found in the directory")
        return
    
    print(f"\nFound {len(files)} files to process")
    
    # Process and plot each file
    for file_path in files:
        print(f"\nProcessing {os.path.basename(file_path)}")
        try:
            data = process_atl21_file(file_path)
            plot_monthly_data(data)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

if __name__ == "__main__":
    main()

def analyze_seasonal_trends(df):
    """Analyze seasonal trends with detrending"""
    plt.figure(figsize=(15, 12))
    
    # Create subplots
    plt.subplot(2, 1, 1)
    
    # Calculate monthly averages by year
    monthly_means = df.pivot_table(
        index='month',
        columns='year',
        values='mean_ssha',
        aggfunc='mean'
    )
    
    # Plot monthly patterns for each year with different colors
    for year in monthly_means.columns:
        plt.plot(monthly_means.index, monthly_means[year], 
                marker='o', label=str(year), linewidth=2)
    
    plt.grid(True)
    plt.xlabel('Month')
    plt.ylabel('Mean SSHA (m)')
    plt.title('Monthly SSHA Patterns by Year')
    plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Subplot 2: Detrended Seasonal Pattern
    plt.subplot(2, 1, 2)
    
    # Calculate overall seasonal pattern
    seasonal_pattern = df.groupby('month')['mean_ssha'].agg(['mean', 'std'])
    
    # Plot seasonal pattern with error bars
    plt.errorbar(seasonal_pattern.index, seasonal_pattern['mean'],
                yerr=seasonal_pattern['std'],
                fmt='o-', capsize=5, linewidth=2)
    
    plt.grid(True)
    plt.xlabel('Month')
    plt.ylabel('Mean SSHA (m)')
    plt.title('Average Seasonal Pattern (2018-2024)')
    
    # Add seasonal statistics
    stats_text = "Peak Months: {}\nTrough Months: {}".format(
        seasonal_pattern['mean'].nlargest(3).index.tolist(),
        seasonal_pattern['mean'].nsmallest(3).index.tolist()
    )
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Print seasonal statistics
    print("\nSeasonal Statistics:")
    print("-------------------")
    print("Monthly Averages:")
    print(seasonal_pattern.round(4))
    
    # Calculate and print seasonal amplitude
    seasonal_amplitude = seasonal_pattern['mean'].max() - seasonal_pattern['mean'].min()
    print(f"\nSeasonal Amplitude: {seasonal_amplitude:.4f} m")
    
    # Perform seasonal decomposition
    # Group by month and calculate the mean deviation from annual mean
    annual_mean = df.groupby('year')['mean_ssha'].mean()
    df['annual_mean'] = df['year'].map(annual_mean)
    df['seasonal_component'] = df['mean_ssha'] - df['annual_mean']
    
    print("\nSeasonal Component Statistics:")
    seasonal_stats = df.groupby('month')['seasonal_component'].agg(['mean', 'std']).round(4)
    print(seasonal_stats)

def analyze_arctic_patterns(df):
    """Analyze Arctic-specific patterns"""
    # Calculate Arctic seasons
    df['arctic_season'] = pd.cut(df['month'],
                                bins=[0, 3, 6, 9, 12],
                                labels=['Winter', 'Spring', 'Summer', 'Fall'],
                                include_lowest=True)
    
    plt.figure(figsize=(12, 6))
    
    # Plot Arctic seasonal patterns
    sns.boxplot(data=df, x='arctic_season', y='mean_ssha', hue='year')
    plt.title('Arctic Seasonal Patterns')
    plt.xlabel('Season')
    plt.ylabel('Mean SSHA (m)')
    plt.legend(title='Year', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()
    
    # Print Arctic seasonal statistics
    print("\nArctic Seasonal Statistics:")
    print("-------------------------")
    arctic_stats = df.groupby(['year', 'arctic_season'])['mean_ssha'].agg(['mean', 'std']).round(4)
    print(arctic_stats)

if __name__ == "__main__":
    # Load your data from the previous analysis
    df = pd.read_csv('atl21_temporal_stats.csv')  # If you saved it, or get it from your previous analysis
    
    # Run analyses
    analyze_seasonal_trends(df)
    analyze_arctic_patterns(df)