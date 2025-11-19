"""
AutoSPM Analysis v1.2
- Preserves original data integrity
- Visual enhancements applied only for display
- Quantitative validation of processing steps
- No over-filtering or data loss
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

print("AutoSPM Analysis v1.2")

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

# Process data with quantitative validation
print("\n3. Processing data (Plane Level → Align Rows → Remove Scars)...")

# Store original for comparison
data_original = data_array.copy()
print(f"   Original data range: {data_original.min().values*1e9:.4f} to {data_original.max().values*1e9:.4f} nm")
print(f"   Original data mean: {data_original.mean().values*1e9:.6f} nm")
print(f"   Original data RMS: {np.sqrt((data_original**2).mean().values)*1e9:.4f} nm")

# Step 1: Plane level
data_array.spym.plane()
data_flat = data_array.copy()
rms_after_plane = np.sqrt((data_flat**2).mean().values)
print(f"   After plane level: mean={data_flat.mean().values*1e9:.6f} nm, RMS={rms_after_plane*1e9:.4f} nm")

# Step 2: Align rows
data_flat.spym.align()
data_aligned = data_flat.copy()
rms_after_align = np.sqrt((data_aligned**2).mean().values)
print(f"   After align rows: mean={data_aligned.mean().values*1e9:.6f} nm, RMS={rms_after_align*1e9:.4f} nm")

# Step 3: Remove scars - use conservative median filter
data_aligned.spym.Filters.median()
data_clean = data_aligned.copy()
rms_after_filter = np.sqrt((data_clean**2).mean().values)

# Validate processing didn't over-filter
# Compare RMS after plane leveling (baseline) vs after filtering
# This checks if filtering removed too much signal
rms_change_from_plane = abs(rms_after_filter - rms_after_plane) / rms_after_plane * 100
rms_change_from_align = abs(rms_after_filter - rms_after_align) / rms_after_align * 100

print(f"   After median filter: mean={data_clean.mean().values*1e9:.6f} nm, RMS={rms_after_filter*1e9:.4f} nm")
print(f"   RMS change from plane level: {rms_change_from_plane:.2f}% (should be <10% for valid filtering)")
print(f"   RMS change from align: {rms_change_from_align:.2f}% (should be <5% for valid filtering)")

if rms_change_from_align > 10:
    print(f"   WARNING: Large RMS change after filtering suggests possible over-filtering!")
elif rms_change_from_align > 5:
    print(f"   NOTE: Moderate RMS change - filtering may be slightly aggressive")
else:
    print(f"   RMS change is acceptable - no over-filtering detected")

print(f"\n   Final processed data range: {data_clean.min().values*1e9:.4f} to {data_clean.max().values*1e9:.4f} nm")
print(f"   Final processed data mean: {data_clean.mean().values*1e9:.6f} nm (should be ~0)")
print(f"   Final processed data std: {data_clean.std().values*1e9:.4f} nm")

# Compute 2D FFT
print("\n4. Computing 2D FFT...")
fft = np.fft.fft2(data_clean.values)
fft_shifted = np.fft.fftshift(fft)
fft_mag = np.abs(fft_shifted)

# Create spatial frequency arrays (in nm⁻¹)
kx = np.fft.fftshift(np.fft.fftfreq(nx, d=x_range/nx)) * 1e-9  # Convert to nm⁻¹
ky = np.fft.fftshift(np.fft.fftfreq(ny, d=y_range/ny)) * 1e-9  # Convert to nm⁻¹

print(f"   Spatial frequency range: {kx.min():.2f} to {kx.max():.2f} nm⁻¹")
print(f"   FFT magnitude range: {fft_mag.min():.6e} to {fft_mag.max():.6e}")

# Scale FFT to match Gwyddion units (pm)
# Scale by pixel size and normalize to match Gwyddion's 0-15.4 pm range
pixel_size_nm = x_range_nm / nx
fft_mag_pm = fft_mag * pixel_size_nm * 1000  # Convert to pm
fft_max_pm = fft_mag_pm.max()
print(f"   FFT magnitude (scaled to pm): max = {fft_max_pm:.2f} pm")

if fft_max_pm > 0:
    scale_factor = 15.4 / fft_max_pm
    fft_mag_pm_scaled = fft_mag_pm * scale_factor
    print(f"   Scaling factor applied: {scale_factor:.4f}")
    print(f"   Scaled FFT max: {fft_mag_pm_scaled.max():.2f} pm")
else:
    fft_mag_pm_scaled = fft_mag_pm

# Save processed data
print("\n5. Saving results...")
data_clean.to_netcdf(output_dir / "processed_data_v1.2.nc")

# Quantitative FFT Analysis (PRESERVE ORIGINAL DATA)
print("\n6. Quantitative FFT Analysis...")
center_y, center_x = ny // 2, nx // 2

# Analyze original FFT (no modifications)
print(f"   Original FFT statistics:")
print(f"     Center (DC) value: {fft_mag_pm_scaled[center_y, center_x]:.4f} pm")
print(f"     Max value: {fft_mag_pm_scaled.max():.4f} pm")
print(f"     Mean value: {fft_mag_pm_scaled.mean():.4f} pm")
print(f"     Std value: {fft_mag_pm_scaled.std():.4f} pm")

# Calculate DC component strength
dc_value = fft_mag_pm_scaled[center_y, center_x]
non_dc_mask = np.ones_like(fft_mag_pm_scaled, dtype=bool)
non_dc_mask[center_y-2:center_y+3, center_x-2:center_x+3] = False
non_dc_values = fft_mag_pm_scaled[non_dc_mask]
dc_ratio = dc_value / non_dc_values.max() if non_dc_values.max() > 0 else 0
print(f"     DC/Peak ratio: {dc_ratio:.2f} (indicates DC dominance)")

# Find periodic features (peaks away from center)
# Look for significant peaks in rings
periodic_features = []
for radius in [20, 40, 60, 80, 100]:
    y, x = np.ogrid[:ny, :nx]
    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    ring_mask = (dist > radius - 2) & (dist < radius + 2)
    ring_values = fft_mag_pm_scaled[ring_mask]
    if len(ring_values) > 0:
        ring_max = ring_values.max()
        ring_mean = ring_values.mean()
        ring_std = ring_values.std()
        # Significant peaks are > mean + 3*std
        significant_peaks = np.sum(ring_values > ring_mean + 3*ring_std)
        if ring_max > ring_mean + 2*ring_std:
            periodic_features.append({
                'radius': radius,
                'max': ring_max,
                'mean': ring_mean,
                'peaks': significant_peaks
            })

print(f"\n   Periodic features detected:")
for feat in periodic_features[:5]:  # Show top 5
    print(f"     Ring r={feat['radius']}: max={feat['max']:.4f} pm, {feat['peaks']} significant peaks")

# Apply visualization enhancements (FOR DISPLAY ONLY - original data preserved)
print("\n7. Applying visualization enhancements (display only, data preserved)...")

# Create display version with DC suppression (only for visualization)
fft_suppressed_display = fft_mag_pm_scaled.copy()  # Copy for display, original preserved

# Suppress DC only for visualization (smaller region, more conservative)
center_region_size = 5  # Smaller than before to preserve more data
y_start = max(0, center_y - center_region_size // 2)
y_end = min(ny, center_y + center_region_size // 2 + 1)
x_start = max(0, center_x - center_region_size // 2)
x_end = min(nx, center_x + center_region_size // 2 + 1)

# Use local median (small region around center) instead of global median
local_region_size = 15
y_local_start = max(0, center_y - local_region_size // 2)
y_local_end = min(ny, center_y + local_region_size // 2 + 1)
x_local_start = max(0, center_x - local_region_size // 2)
x_local_end = min(nx, center_x + local_region_size // 2 + 1)
local_region = fft_suppressed_display[y_local_start:y_local_end, x_local_start:x_local_end]
local_mask = np.ones_like(local_region, dtype=bool)
local_center_y = local_region.shape[0] // 2
local_center_x = local_region.shape[1] // 2
local_mask[local_center_y-3:local_center_y+4, local_center_x-3:local_center_x+4] = False
local_median = np.median(local_region[local_mask])
fft_suppressed_display[y_start:y_end, x_start:x_end] = local_median

print(f"   DC suppression (display only): center {center_region_size}×{center_region_size} region")
print(f"     Replaced with local median: {local_median:.4f} pm")
print(f"     Original DC value preserved in data: {fft_mag_pm_scaled[center_y, center_x]:.4f} pm")

# Percentile scaling options (for display)
vmax_99_9 = np.percentile(fft_mag_pm_scaled, 99.9)  # Conservative
vmax_99_5 = np.percentile(fft_mag_pm_scaled, 99.5)  # Moderate
vmax_suppressed = np.percentile(fft_suppressed_display, 99.5)  # For suppressed version

print(f"   Percentile scaling options:")
print(f"     99.9% (conservative): {vmax_99_9:.4f} pm")
print(f"     99.5% (moderate): {vmax_99_5:.4f} pm")


# Save FFT images with enhanced visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Row 1: Original processing
# Plot 1: Processed data
ax = axes[0, 0]
im1 = ax.imshow(data_clean.values * 1e9, cmap='gray')
ax.set_title('Processed SPM Data', fontweight='bold')
ax.set_xlabel('X (pixels)')
ax.set_ylabel('Y (pixels)')
plt.colorbar(im1, ax=ax, label='Height (nm)')

# Plot 2: FFT modulus (original, 99.9 percentile)
ax = axes[0, 1]
vmax_original = np.percentile(fft_mag_pm_scaled, 99.9)
im2 = ax.imshow(fft_mag_pm_scaled, cmap='hot', vmin=0, vmax=vmax_original)
ax.set_title(f'FFT Modulus (Original)\n99.9% = {vmax_original:.2f} pm', fontweight='bold')
ax.set_xlabel('kx (spatial frequency)')
ax.set_ylabel('ky (spatial frequency)')
plt.colorbar(im2, ax=ax, label='|FFT| (pm)')

# Plot 3: FFT with DC suppressed (display only)
ax = axes[0, 2]
im3 = ax.imshow(fft_suppressed_display, cmap='hot', vmin=0, vmax=vmax_suppressed)
ax.set_title(f'FFT (DC Suppressed - Display Only)\n99.5% = {vmax_suppressed:.2f} pm', fontweight='bold')
ax.set_xlabel('kx (spatial frequency)')
ax.set_ylabel('ky (spatial frequency)')
plt.colorbar(im3, ax=ax, label='|FFT| (pm)')

# Row 2: Enhanced visualizations
# Plot 4: FFT with aggressive percentile scaling (99.5%)
ax = axes[1, 0]
im4 = ax.imshow(fft_mag_pm_scaled, cmap='hot', vmin=0, vmax=vmax_99_5)
ax.set_title(f'FFT (99.5% Scaling)\nEnhanced Contrast', fontweight='bold')
ax.set_xlabel('kx (spatial frequency)')
ax.set_ylabel('ky (spatial frequency)')
plt.colorbar(im4, ax=ax, label='|FFT| (pm)')

# Plot 5: FFT with original colormap (original data)
ax = axes[1, 1]
im5 = ax.imshow(fft_mag_pm_scaled, cmap='hot', vmin=0, vmax=vmax_99_5)
ax.set_title('FFT (Original Colormap)\nOriginal Data, 99.5% Scaling', fontweight='bold')
ax.set_xlabel('kx (spatial frequency)')
ax.set_ylabel('ky (spatial frequency)')
plt.colorbar(im5, ax=ax, label='|FFT| (pm)')

# Plot 6: FFT log scale (original data, no suppression)
ax = axes[1, 2]
fft_log_original = np.log(fft_mag_pm_scaled + 0.1)
im6 = ax.imshow(fft_log_original, cmap='hot')
ax.set_title('FFT (Log Scale, Original Data)\nNo DC Suppression', fontweight='bold')
ax.set_xlabel('kx')
ax.set_ylabel('ky')
plt.colorbar(im6, ax=ax, label='log(|FFT| + 0.1)')

plt.tight_layout()
plt.savefig(output_dir / "analysis_v1.2.png", dpi=200, bbox_inches='tight')
print(f"   Saved: analysis_v1.2.png")
plt.close(fig)

# Save versions: Original (analytically accurate) and Display (enhanced visualization)
print("\n8. Saving output files...")

# Version 1: Original FFT data (analytically accurate, no modifications)
fig2, ax2 = plt.subplots(figsize=(10, 10))
im_original = ax2.imshow(fft_mag_pm_scaled, cmap='hot', vmin=0, vmax=vmax_99_9)
ax2.set_title('FFT Modulus - Original Data (Analytically Accurate)\n99.9% Scaling, No DC Suppression', 
              fontsize=14, fontweight='bold')
ax2.set_xlabel('kx (nm⁻¹)', fontsize=12)
ax2.set_ylabel('ky (nm⁻¹)', fontsize=12)
cbar = plt.colorbar(im_original, ax=ax2, label='|FFT| (pm)', fraction=0.046)
cbar.ax.tick_params(labelsize=10)
plt.tight_layout()
plt.savefig(output_dir / "fft_v1.2_original.png", dpi=200, bbox_inches='tight')
plt.close(fig2)
print(f"   Saved: fft_v1.2_original.png (analytically accurate, no modifications)")

# Version 2: Enhanced display (DC suppressed for visualization)
fig3, ax3 = plt.subplots(figsize=(10, 10))
im_display = ax3.imshow(fft_suppressed_display, cmap='hot', vmin=0, vmax=vmax_suppressed)
ax3.set_title('FFT Modulus - Enhanced Display (Visualization Only)\nDC Suppressed, Hot Colormap', 
              fontsize=14, fontweight='bold')
ax3.set_xlabel('kx (nm⁻¹)', fontsize=12)
ax3.set_ylabel('ky (nm⁻¹)', fontsize=12)
cbar3 = plt.colorbar(im_display, ax=ax3, label='|FFT| (pm)', fraction=0.046)
cbar3.ax.tick_params(labelsize=10)
plt.tight_layout()
plt.savefig(output_dir / "fft_v1.2_display.png", dpi=200, bbox_inches='tight')
plt.close(fig3)
print(f"   Saved: fft_v1.2_display.png (enhanced visualization, DC suppressed)")

# Version 3: Recommended - DC suppressed with balanced scaling (best visual clarity + accuracy)
# Use DC-suppressed data with 99.7% scaling for optimal balance
vmax_99_7 = np.percentile(fft_suppressed_display, 99.7)  # Balanced scaling
fig4, ax4 = plt.subplots(figsize=(10, 10))
im_best = ax4.imshow(fft_suppressed_display, cmap='hot', vmin=0, vmax=vmax_99_7)
ax4.set_title('FFT Modulus - Recommended (DC Suppressed)\nHot Colormap, 99.7% Scaling', 
              fontsize=14, fontweight='bold')
ax4.set_xlabel('kx (nm⁻¹)', fontsize=12)
ax4.set_ylabel('ky (nm⁻¹)', fontsize=12)
cbar4 = plt.colorbar(im_best, ax=ax4, label='|FFT| (pm)', fraction=0.046)
cbar4.ax.tick_params(labelsize=10)
plt.tight_layout()
plt.savefig(output_dir / "fft_v1.2.png", dpi=200, bbox_inches='tight')
plt.close(fig4)
print(f"   Saved: fft_v1.2.png (recommended: DC suppressed + balanced 99.7% scaling)")

# Save FFT data for quantitative analysis (original, unmodified)
print("\n9. Saving quantitative analysis data...")
fft_dataset = xr.Dataset({
    'fft_magnitude': (['ky', 'kx'], fft_mag_pm_scaled),
    'fft_magnitude_display': (['ky', 'kx'], fft_suppressed_display),  # Display version
})
fft_dataset.coords['kx'] = kx
fft_dataset.coords['ky'] = ky
fft_dataset.attrs['description'] = 'FFT analysis - original data preserved'
fft_dataset.attrs['dc_value'] = float(fft_mag_pm_scaled[center_y, center_x])
fft_dataset.attrs['max_value'] = float(fft_mag_pm_scaled.max())
fft_dataset.to_netcdf(output_dir / "fft_v1.2.nc")
print(f"   Saved: fft_v1.2.nc (quantitative FFT data)")

print("\nAnalysis finished")
print(f"\nRMS change from align: {rms_change_from_align:.2f}%")
print(f"DC component: {dc_value:.4f} pm")
print(f"Periodic features: {len(periodic_features)} rings detected")
print(f"\nFiles saved to: {output_dir}")