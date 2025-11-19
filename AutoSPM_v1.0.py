"""
AutoSPM Analysis v1.0
- proof of concept
"""

from spym.process import Level, Filters
import nanonispy as nap
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import xarray as xr

input_path = Path("/Users/isaactsai/Downloads/B46HTG_029.sxm")
output_dir = Path("/Users/isaactsai/STMoutput")
output_dir.mkdir(exist_ok=True, parents=True)

try:
    scan = nap.read.Scan(str(input_path))
    print(f"Loaded scan: {scan}")
    print(f"Available channels: {scan.signals.keys()}")
    
    channel_name = list(scan.signals.keys())[0]
    channel_dict = scan.signals[channel_name]
    
    print(f"Using channel: {channel_name}")
    
    if isinstance(channel_dict, dict):
        if 'forward' in channel_dict:
            channel_data = channel_dict['forward']
            print(f"Using 'forward' scan direction")
        elif 'backward' in channel_dict:
            channel_data = channel_dict['backward']
            print(f"Using 'backward' scan direction")
        else:
            channel_data = list(channel_dict.values())[0]
    else:
        channel_data = channel_dict
    
    print(f"Data shape: {channel_data.shape}")
    print(f"Data type: {type(channel_data)}")
    
    if hasattr(scan, 'header'):
        scan_range = scan.header.get('scan_range', None)
        if scan_range is not None and len(scan_range) >= 2:
            x_range = float(scan_range[0]) * 1e-9
            y_range = float(scan_range[1]) * 1e-9
        else:
            x_range = 1.0
            y_range = 1.0
    else:
        x_range = 1.0
        y_range = 1.0
    
    ny, nx = channel_data.shape
    x_coords = np.linspace(0, x_range, nx)
    y_coords = np.linspace(0, y_range, ny)
    
    data_array = xr.DataArray(
        channel_data,
        dims=['y', 'x'],
        coords={'x': x_coords, 'y': y_coords},
        name=channel_name
    )
    
    print(f"Converted to xarray DataArray: {data_array}")
    
except Exception as e:
    print(f"Error loading .sxm file: {e}")
    print("\nTrying alternative method...")
    # Fallback: try to read as raw data
    raise

data_array.spym.plane()
data_flat = data_array.copy()  # Work with a copy

# Align rows
data_flat.spym.align()
data_aligned = data_flat.copy()

# Median filter to remove scars)
data_aligned.spym.Filters.median()
data_clean = data_aligned.copy()

# 2D FFT using numpy
fft_image = np.fft.fft2(data_clean.values)
fft_image = np.fft.fftshift(fft_image)  # Shift zero frequency to center
fft_magnitude = np.abs(fft_image)

# Save as NetCDF
data_clean.to_netcdf(output_dir / "processed_data_v1.0.nc")

# Save FFT as image using matplotlib
plt.imsave(output_dir / "fft_v1.0.png", np.log(fft_magnitude + 1), cmap='gray')

# Save processed data as image
plt.imsave(output_dir / "processed_data_v1.0.png", data_clean.values, cmap='gray')

# plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(data_clean.values, cmap='gray')
plt.title("Processed SPM Data")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(np.log(fft_magnitude + 1), cmap='gray')
plt.title("2D FFT (log scale)")
plt.colorbar()

plt.tight_layout()
plt.savefig(output_dir / "analysis_v1.0.png", dpi=150)
plt.show()
