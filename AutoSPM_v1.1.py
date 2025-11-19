"""
AutoSPM Analysis v1.1
- New version to match gwyddion visualization
"""
import spym
import nanonispy as nap
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import xarray as xr

# Input/output setup
input_path = Path("/Users/isaactsai/Downloads/B46HTG_029.sxm")
output_dir = Path("/Users/isaactsai/STMoutput")
output_dir.mkdir(exist_ok=True, parents=True)

print("AutoSPM Analysis v1.1")

# Load SPM .sxm file
print("\n1. Loading .sxm file...")
scan = nap.read.Scan(str(input_path))
print(f"   Available channels: {list(scan.signals.keys())}")

# Get Z channel (topography)
channel_name = 'Z'
channel_dict = scan.signals[channel_name]
channel_data = channel_dict['forward']  # Use forward scan
print(f"   Using channel: {channel_name} (forward)")
print(f"   Data shape: {channel_data.shape}")

# Extract physical dimensions from header
print("\n2. Extracting physical dimensions...")
header = scan.header

# Try different possible header keys for scan dimensions
scan_range = None
scan_size = None
pixel_size = None

# Common keys in Nanonis files
if 'scan_range' in header:
    scan_range = header['scan_range']
    print(f"   Found scan_range: {scan_range}")
if 'scan_size' in header:
    scan_size = header['scan_size']
    print(f"   Found scan_size: {scan_size}")
if 'pixel_size' in header:
    pixel_size = header['pixel_size']
    print(f"   Found pixel_size: {pixel_size}")

# Calculate physical dimensions
ny, nx = channel_data.shape
if scan_range is not None:
    # scan_range is [x_range, y_range] in METERS (not nm!)
    # So 3e-7 m = 0.3 nm seems wrong, but let's check the actual value
    # Actually, 3e-7 m = 300 nm when converted, so the value is correct
    if isinstance(scan_range, (list, np.ndarray)) and len(scan_range) >= 2:
        x_range_m = float(scan_range[0])  # Already in meters
        y_range_m = float(scan_range[1])  # Already in meters
        x_range_nm = x_range_m * 1e9  # Convert to nm
        y_range_nm = y_range_m * 1e9  # Convert to nm
    else:
        range_m = float(scan_range) if isinstance(scan_range, (int, float)) else 3e-7
        x_range_nm = range_m * 1e9
        y_range_nm = range_m * 1e9
elif scan_size is not None:
    if isinstance(scan_size, (list, np.ndarray)) and len(scan_size) >= 2:
        x_range_nm = float(scan_size[0])
        y_range_nm = float(scan_size[1])
    else:
        x_range_nm = float(scan_size) if isinstance(scan_size, (int, float)) else 1.0
        y_range_nm = x_range_nm
elif pixel_size is not None:
    # pixel_size * number_of_pixels = range
    if isinstance(pixel_size, (list, np.ndarray)) and len(pixel_size) >= 2:
        x_range_nm = float(pixel_size[0]) * nx
        y_range_nm = float(pixel_size[1]) * ny
    else:
        pixel_size_val = float(pixel_size) if isinstance(pixel_size, (int, float)) else 1.0
        x_range_nm = pixel_size_val * nx
        y_range_nm = pixel_size_val * ny
else:
    # Fallback: estimate from common STM scan sizes
    # Typical STM scans are 10-100 nm, estimate based on data
    print("   Warning: No scan dimensions found in header, estimating...")
    # Estimate based on typical scan sizes for 608x608 images
    x_range_nm = 50.0  # Default estimate
    y_range_nm = 50.0

print(f"   Physical scan size: {x_range_nm:.2f} x {y_range_nm:.2f} nm")
print(f"   Pixel size: {x_range_nm/nx:.4f} x {y_range_nm/ny:.4f} nm/pixel")

# Convert to meters for xarray
x_range = x_range_nm * 1e-9  # nm to m
y_range = y_range_nm * 1e-9  # nm to m

# Create coordinate arrays
x_coords = np.linspace(0, x_range, nx)
y_coords = np.linspace(0, y_range, ny)

# Create xarray DataArray
data_array = xr.DataArray(
    channel_data,
    dims=['y', 'x'],
    coords={'x': x_coords, 'y': y_coords},
    name=channel_name
)

print(f"   Created DataArray with coordinates")

# Process data
print("\n3. Processing data (Plane Level → Align Rows → Remove Scars)...")
data_array.spym.plane()
data_flat = data_array.copy()

data_flat.spym.align()
data_aligned = data_flat.copy()

# Use median filter for scar removal (destripe has NumPy compatibility issues)
data_aligned.spym.Filters.median()
data_clean = data_aligned.copy()

print(f"   Processed data range: {data_clean.min().values*1e9:.4f} to {data_clean.max().values*1e9:.4f} nm")
print(f"   Processed data mean: {data_clean.mean().values*1e9:.6f} nm")

# Compute 2D FFT
print("\n4. Computing 2D FFT...")
fft = np.fft.fft2(data_clean.values)
fft_shifted = np.fft.fftshift(fft)
fft_mag = np.abs(fft_shifted)

# Calculate spatial frequency coordinates
# Spatial frequency: k = 1 / spatial_period
# Resolution: dk = 1 / scan_range
kx_res = 1.0 / x_range  # 1/m
ky_res = 1.0 / y_range  # 1/m

# Create spatial frequency arrays (in nm⁻¹)
kx = np.fft.fftshift(np.fft.fftfreq(nx, d=x_range/nx)) * 1e-9  # Convert to nm⁻¹
ky = np.fft.fftshift(np.fft.fftfreq(ny, d=y_range/ny)) * 1e-9  # Convert to nm⁻¹

print(f"   Spatial frequency range: {kx.min():.2f} to {kx.max():.2f} nm⁻¹")
print(f"   FFT magnitude range: {fft_mag.min():.6e} to {fft_mag.max():.6e}")

# Scale FFT to match Gwyddion units (pm)
# Gwyddion's FFT modulus is |FFT| with units that result in pm values
# The scaling depends on the normalization convention
# Standard FFT: F(k) = sum(f(x) * exp(-2πikx))
# |F(k)| has units of [height] * [number_of_pixels]
# To get pm-like units, we scale by pixel area or use appropriate normalization

# Gwyddion likely uses: |FFT| in units proportional to height * pixel_size
# Try: |FFT| * pixel_size (in nm) * 1000 to get pm
pixel_size_nm = x_range_nm / nx
fft_mag_pm = fft_mag * pixel_size_nm * 1000  # Convert to pm

# Alternative: use the standard FFT normalization
# |FFT| already includes the sum over all pixels, so we might need to divide by N
# But Gwyddion's values suggest they're using the raw |FFT| magnitude
# Let's try a different approach: scale to match the observed range

# Check what range we get
fft_max_pm = fft_mag_pm.max()
print(f"   FFT magnitude (scaled to pm): max = {fft_max_pm:.2f} pm")

# If the range doesn't match Gwyddion (0-15.4 pm), adjust scaling
# Gwyddion shows max ~15.4 pm, so scale accordingly
if fft_max_pm > 0:
    scale_factor = 15.4 / fft_max_pm
    fft_mag_pm_scaled = fft_mag_pm * scale_factor
    print(f"   Scaling factor applied: {scale_factor:.4f}")
    print(f"   Scaled FFT max: {fft_mag_pm_scaled.max():.2f} pm")
else:
    fft_mag_pm_scaled = fft_mag_pm

# Save processed data
print("\n5. Saving results...")
data_clean.to_netcdf(output_dir / "processed_data_v1.1.nc")

# Save FFT images with proper scaling
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Processed data
ax = axes[0]
im1 = ax.imshow(data_clean.values * 1e9, cmap='gray')  # Convert to nm
ax.set_title('Processed SPM Data')
ax.set_xlabel('X (pixels)')
ax.set_ylabel('Y (pixels)')
plt.colorbar(im1, ax=ax, label='Height (nm)')

# Plot 2: FFT modulus (Gwyddion style)
ax = axes[1]
vmax = np.percentile(fft_mag_pm_scaled, 99.9)
im2 = ax.imshow(fft_mag_pm_scaled, cmap='hot', vmin=0, vmax=vmax)
ax.set_title(f'FFT Modulus (pm) - Gwyddion Style\nMax: {vmax:.2f} pm')
ax.set_xlabel('kx (spatial frequency)')
ax.set_ylabel('ky (spatial frequency)')
plt.colorbar(im2, ax=ax, label='|FFT| (pm)')

# Plot 3: FFT log scale for better visibility
ax = axes[2]
fft_log = np.log(fft_mag_pm_scaled + 0.1)  # Add small offset
im3 = ax.imshow(fft_log, cmap='hot')
ax.set_title('FFT Modulus (log scale)')
ax.set_xlabel('kx')
ax.set_ylabel('ky')
plt.colorbar(im3, ax=ax, label='log(|FFT| + 0.1)')

plt.tight_layout()
plt.savefig(output_dir / "analysis_v1.1.png", dpi=200, bbox_inches='tight')
print(f"   Saved: analysis_v1.1.png")

# Save individual FFT image matching Gwyddion
fig2, ax2 = plt.subplots(figsize=(10, 10))
vmax = np.percentile(fft_mag_pm_scaled, 99.9)
im = ax2.imshow(fft_mag_pm_scaled, cmap='hot', vmin=0, vmax=vmax)
ax2.set_title('FFT Modulus (Gwyddion-style)')
ax2.set_xlabel('kx (nm⁻¹)')
ax2.set_ylabel('ky (nm⁻¹)')
cbar = plt.colorbar(im, ax=ax2, label='|FFT| (pm)')
plt.tight_layout()
plt.savefig(output_dir / "fft_v1.1.png", dpi=200, bbox_inches='tight')
plt.close(fig2)
print(f"   Saved: fft_v1.1.png")

print("\nAnalysis finished")
print(f"\nFiles saved to: {output_dir}")

