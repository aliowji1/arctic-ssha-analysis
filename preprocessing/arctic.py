import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import os
from datetime import datetime
import glob

def generate_ssha_images(file_path, output_dir="ssha_images"):
    """Generate standardized SSHA visualization images for CNN"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get date from filename
    date_str = os.path.basename(file_path).split('_')[1][:8]
    date = datetime.strptime(date_str, '%Y%m%d')
    
    with h5py.File(file_path, 'r') as f:
        # Extract data
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
        
        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 1. SSHA Map
        m = Basemap(projection='npstere', boundinglat=60, lon_0=0, resolution='l', ax=ax1)
        x, y = m(lon, lat)
        m.drawcoastlines()
        m.fillcontinents(color='gray', lake_color='aqua')
        m.drawparallels(np.arange(-80., 81., 20.))
        m.drawmeridians(np.arange(-180., 181., 20.))
        
        # Use consistent color scale for all images
        vmin, vmax = -0.5, 0.5
        cs = m.pcolormesh(x, y, valid_ssha, cmap='coolwarm', latlon=False, vmin=vmin, vmax=vmax)
        plt.colorbar(cs, ax=ax1, label='SSHA (m)')
        ax1.set_title(f'Sea Surface Height Anomaly - {date.strftime("%B %Y")}')
        
        # 2. Distribution Plot
        valid_data = valid_ssha.compressed()
        ax2.hist(valid_data, bins=50, edgecolor='black')
        ax2.set_xlabel('SSHA (m)')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'SSHA Distribution - {date.strftime("%B %Y")}')
        
        # Add statistics
        stats_text = f'Mean: {np.mean(valid_data):.4f} m\n'
        stats_text += f'Median: {np.median(valid_data):.4f} m\n'
        stats_text += f'Std Dev: {np.std(valid_data):.4f} m'
        ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save figure
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'ssha_{date.strftime("%Y%m")}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Generated image for {date.strftime('%Y-%m')}")
        
        # Also save the raw data for CNN training
        np.savez(os.path.join(output_dir, f'ssha_data_{date.strftime("%Y%m")}'),
                ssha=valid_ssha.data,
                mask=valid_ssha.mask,
                date=date.strftime('%Y%m'))

def main():
    data_dir = 'atl21_data'
    files = sorted(glob.glob(os.path.join(data_dir, 'ATL21-01_*.h5')))
    
    if not files:
        print("No ATL21 files found")
        return
    
    print(f"Found {len(files)} files to process")
    
    for file_path in files:
        try:
            generate_ssha_images(file_path)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

if __name__ == "__main__":
    main()