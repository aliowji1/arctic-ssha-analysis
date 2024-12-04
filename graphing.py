import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# Open the HDF5 filex
file_path = 'ATL21-01_20190101005132_00550201_003_01.h5'
f = h5py.File(file_path, 'r')

# Extract data
lat = f['grid_lat'][:]
lon = f['grid_lon'][:]
monthly_ssha = f['monthly/mean_ssha'][:]
land_mask = f['land_mask_map'][:]

# Mask invalid data and land areas
ocean_mask = (monthly_ssha != 3.4028235e+38) & (land_mask == 0)
valid_ssha = np.ma.masked_array(monthly_ssha, ~ocean_mask)

# Visualization 1: Map of SSHA
plt.figure(figsize=(12, 8))
m = Basemap(projection='npstere', boundinglat=60, lon_0=0, resolution='l')
x, y = m(lon, lat)
m.drawcoastlines()
m.fillcontinents(color='gray', lake_color='aqua')
m.drawparallels(np.arange(-80., 81., 20.))
m.drawmeridians(np.arange(-180., 181., 20.))
cs = m.pcolormesh(x, y, valid_ssha, cmap='coolwarm', latlon=False)
plt.colorbar(cs, label='SSHA (m)')
plt.title('Sea Surface Height Anomaly')
plt.show()

# Visualization 2: Histogram of SSHA
plt.figure(figsize=(10, 6))
plt.hist(valid_ssha.compressed(), bins=50, edgecolor='black')
plt.xlabel('SSHA (m)')
plt.ylabel('Frequency')
plt.title('Distribution of Sea Surface Height Anomalies')
plt.show()

# Calculate some statistics
mean_ssha = np.mean(valid_ssha)
median_ssha = np.median(valid_ssha)
std_ssha = np.std(valid_ssha)

print(f"Mean SSHA: {mean_ssha:.4f} m")
print(f"Median SSHA: {median_ssha:.4f} m")
print(f"Standard Deviation of SSHA: {std_ssha:.4f} m")

# Visualization 3: Identify areas with extreme SSHA
threshold = 2 * std_ssha  # 2 standard deviations from the mean
extreme_mask = (valid_ssha > mean_ssha + threshold) | (valid_ssha < mean_ssha - threshold)

plt.figure(figsize=(12, 8))
m = Basemap(projection='npstere', boundinglat=60, lon_0=0, resolution='l')
x, y = m(lon, lat)
m.drawcoastlines()
m.fillcontinents(color='gray', lake_color='aqua')
m.drawparallels(np.arange(-80., 81., 20.))
m.drawmeridians(np.arange(-180., 181., 20.))
m.pcolormesh(x, y, extreme_mask, cmap='RdYlBu', latlon=False)
plt.title('Areas with Extreme Sea Surface Height Anomalies')
plt.show()


f.close()

