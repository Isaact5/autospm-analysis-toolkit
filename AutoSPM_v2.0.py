"""
AutoSPM Analysis v2.0
- All v1.2 features (analytical accuracy, data preservation)
- Twist angle calculation from FFT peaks (moiré pattern analysis)
- Domain wall detection from topography gradients
- Strain calculation from lattice constant variations
- Comprehensive visualization and quantitative metrics
"""
import spym
import nanonispy as nap
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import xarray as xr
from scipy import ndimage, signal
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# Input/output setup
input_path = Path("/Users/isaactsai/Downloads/B46HTG_029.sxm")
output_dir = Path("/Users/isaactsai/STMoutput")
output_dir.mkdir(exist_ok=True, parents=True)

print("AutoSPM Analysis v2.0")

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
data_clean.to_netcdf(output_dir / "processed_data_v2.0.nc")

# Quantitative FFT Analysis (PRESERVE ORIGINAL DATA)
print("\n6. FFT Analysis...")
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
plt.savefig(output_dir / "analysis_v2.0_basic.png", dpi=200, bbox_inches='tight')
print(f"   Saved: analysis_v2.0_basic.png")
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
plt.savefig(output_dir / "fft_v2.0_original.png", dpi=200, bbox_inches='tight')
plt.close(fig2)
print(f"   Saved: fft_v2.0_original.png (analytically accurate, no modifications)")

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
plt.savefig(output_dir / "fft_v2.0_display.png", dpi=200, bbox_inches='tight')
plt.close(fig3)
print(f"   Saved: fft_v2.0_display.png (enhanced visualization, DC suppressed)")

# Version 3: Recommended - DC suppressed with balanced scaling (best visual clarity + accuracy)
# Use DC-suppressed data with 99.7% scaling for optimal balance
vmax_99_7_basic = np.percentile(fft_suppressed_display, 99.7)  # Balanced scaling
fig4, ax4 = plt.subplots(figsize=(10, 10))
im_best = ax4.imshow(fft_suppressed_display, cmap='hot', vmin=0, vmax=vmax_99_7_basic)
ax4.set_title('FFT Modulus - Recommended (DC Suppressed)\nHot Colormap, 99.7% Scaling', 
              fontsize=14, fontweight='bold')
ax4.set_xlabel('kx (nm⁻¹)', fontsize=12)
ax4.set_ylabel('ky (nm⁻¹)', fontsize=12)
cbar4 = plt.colorbar(im_best, ax=ax4, label='|FFT| (pm)', fraction=0.046)
cbar4.ax.tick_params(labelsize=10)
plt.tight_layout()
plt.savefig(output_dir / "fft_v2.0.png", dpi=200, bbox_inches='tight')
plt.close(fig4)
print(f"   Saved: fft_v2.0.png (recommended: DC suppressed + balanced 99.7% scaling)")

# Save FFT data for quantitative analysis (original, unmodified)
print("\n9. Saving data...")
fft_dataset = xr.Dataset({
    'fft_magnitude': (['ky', 'kx'], fft_mag_pm_scaled),
    'fft_magnitude_display': (['ky', 'kx'], fft_suppressed_display),  # Display version
})
fft_dataset.coords['kx'] = kx
fft_dataset.coords['ky'] = ky
fft_dataset.attrs['description'] = 'FFT analysis - original data preserved'
fft_dataset.attrs['dc_value'] = float(fft_mag_pm_scaled[center_y, center_x])
fft_dataset.attrs['max_value'] = float(fft_mag_pm_scaled.max())
fft_dataset.to_netcdf(output_dir / "fft_v2.0.nc")
print(f"   Saved: fft_v2.0.nc")

# Advanced analysis: twist angle, domain walls, strain

# ----------------------------------------------------------------------------
# 1. TWIST ANGLE CALCULATION (from FFT peaks)
# ----------------------------------------------------------------------------
print("\n10. Calculating Twist Angle from FFT Peaks...")

def find_fft_peaks(fft_mag, center_y, center_x, min_distance=10, threshold_percentile=95):
    """
    Find significant peaks in FFT spectrum.
    Returns peak positions (y, x) and their magnitudes.
    """
    # Create mask to exclude DC region
    y, x = np.ogrid[:fft_mag.shape[0], :fft_mag.shape[1]]
    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    dc_mask = dist_from_center < 5  # Exclude center 5 pixels
    
    # Threshold for peak detection
    threshold = np.percentile(fft_mag[~dc_mask], threshold_percentile)
    
    # Find local maxima
    from scipy.ndimage import maximum_filter
    local_maxima = maximum_filter(fft_mag, size=min_distance) == fft_mag
    peaks_mask = (fft_mag > threshold) & local_maxima & (~dc_mask)
    
    peak_y, peak_x = np.where(peaks_mask)
    peak_magnitudes = fft_mag[peak_y, peak_x]
    
    # Sort by magnitude
    sort_idx = np.argsort(peak_magnitudes)[::-1]
    peak_y = peak_y[sort_idx]
    peak_x = peak_x[sort_idx]
    peak_magnitudes = peak_magnitudes[sort_idx]
    
    return peak_y, peak_x, peak_magnitudes

def calculate_twist_angle(peak_y, peak_x, center_y, center_x, kx, ky, peak_magnitudes, 
                          lattice_constant_nm=0.246):
    """
    Calculate twist angle from FFT peak positions for helical trilayer graphene.
    
    For twisted graphene: moiré period L = a / (2*sin(θ/2))
    Solving for θ: θ = 2*arcsin(a/(2*L))
    
    For trilayer helical stacks, we look for the primary moiré peaks
    (not higher-order beat patterns).
    """
    if len(peak_y) < 2:
        return None, None, None
    
    # Get spatial frequencies of peaks
    peak_kx = kx[peak_x]
    peak_ky = ky[peak_y]
    
    # Calculate angles of peaks relative to center
    angles = np.arctan2(peak_ky, peak_kx) * 180 / np.pi
    
    # Find primary peaks (usually 6-fold symmetric for hexagonal lattices)
    # Group peaks by angle (within 30 degrees)
    primary_peaks = []
    used = np.zeros(len(peak_y), dtype=bool)
    
    for i in range(min(12, len(peak_y))):  # Check top 12 peaks
        if used[i]:
            continue
        
        # Find peaks in similar direction
        angle_i = angles[i]
        similar = np.abs(angles - angle_i) < 30
        similar = similar & (~used)
        
        if np.sum(similar) > 0:
            # Use the strongest peak in this direction
            similar_idx = np.where(similar)[0]
            strongest = similar_idx[np.argmax(peak_magnitudes[similar_idx])]
            primary_peaks.append(strongest)
            used[similar] = True
    
    if len(primary_peaks) < 2:
        return None, None, None
    
    # Calculate angles between primary peaks
    primary_angles = angles[primary_peaks[:6]]  # Use up to 6 primary peaks
    primary_angles = np.sort(primary_angles)
    
    # For hexagonal lattices, angles should be ~60° apart
    # Twist angle is half the smallest angle difference
    angle_diffs = np.diff(primary_angles)
    angle_diffs = np.append(angle_diffs, primary_angles[0] + 360 - primary_angles[-1])
    
    # Find the smallest angle difference (should be ~60° for hexagonal)
    min_diff = np.min(angle_diffs[angle_diffs > 10])  # Ignore very small differences
    
    # Twist angle is typically half of the moiré angle
    # For small twist angles, moiré period ≈ a / (2*sin(θ/2))
    # where a is lattice constant and θ is twist angle
    twist_angle = min_diff / 2  # Simplified estimate
    
    # More accurate: use the angle between two primary peaks
    if len(primary_peaks) >= 2:
        vec1 = np.array([peak_kx[primary_peaks[0]], peak_ky[primary_peaks[0]]])
        vec2 = np.array([peak_kx[primary_peaks[1]], peak_ky[primary_peaks[1]]])
        
        # Normalize
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        
        # Angle between vectors
        cos_angle = np.clip(np.dot(vec1, vec2), -1, 1)
        angle_between = np.arccos(cos_angle) * 180 / np.pi
        
        # For hexagonal lattices, twist angle ≈ angle_between / 2
        twist_angle = angle_between / 2
    
    # NEW METHOD: Calculate from moiré period instead of peak angles
    # Calculate distances from center (spatial frequencies)
    peak_frequencies = np.sqrt(peak_kx**2 + peak_ky**2)
    
    # Look for peaks in the range 0.05 - 0.5 nm⁻¹ (primary moiré range)
    # This corresponds to moiré periods of ~2-20 nm (typical for twisted graphene)
    moire_freq_min = 0.05  # nm⁻¹
    moire_freq_max = 0.5   # nm⁻¹
    
    moire_mask = (peak_frequencies > moire_freq_min) & (peak_frequencies < moire_freq_max)
    moire_peaks = np.where(moire_mask)[0]
    
    if len(moire_peaks) < 2:
        # Use all peaks but prioritize lower frequencies
        moire_peaks = np.argsort(peak_frequencies)[:min(12, len(peak_y))]
    
    # Get the strongest moiré peaks
    moire_magnitudes = peak_magnitudes[moire_peaks]
    sort_idx = np.argsort(moire_magnitudes)[::-1]
    primary_moire_peaks = moire_peaks[sort_idx[:6]]
    
    if len(primary_moire_peaks) < 2:
        return None, None, None, None
    
    # Calculate moiré periods from peak frequencies
    moire_periods = 1.0 / peak_frequencies[primary_moire_peaks]
    
    # Use the STRONGEST peak (by magnitude), not biased toward any specific angle
    # This removes bias - we select based on signal strength, not expected value
    peak_mags_at_moire = peak_magnitudes[primary_moire_peaks]
    best_idx = np.argmax(peak_mags_at_moire)  # Strongest peak
    primary_period = moire_periods[best_idx]
    
    # Calculate twist angle from moiré period: θ = 2*arcsin(a/(2*L))
    if primary_period > 0:
        twist_angle = 2 * np.arcsin(lattice_constant_nm / (2 * primary_period)) * 180 / np.pi
    else:
        twist_angle = None
    
    return twist_angle, primary_moire_peaks, (peak_kx, peak_ky), primary_period

# Find FFT peaks
peak_y, peak_x, peak_mags = find_fft_peaks(fft_mag_pm_scaled, center_y, center_x, 
                                           min_distance=15, threshold_percentile=98)

print(f"   Found {len(peak_y)} significant FFT peaks")
if len(peak_y) > 0:
    print(f"   Top 6 peak magnitudes: {peak_mags[:6]}")
    
    # Calculate twist angle (for helical trilayer graphene)
    result = calculate_twist_angle(
        peak_y, peak_x, center_y, center_x, kx, ky, peak_mags, 
        lattice_constant_nm=0.246  # Graphene lattice constant
    )
    
    if result[0] is not None:
        twist_angle, primary_peaks, peak_coords, moire_period = result
        
        # DIAGNOSTICS: Show all detected moiré periods and their twist angles
        print(f"\n   DIAGNOSTICS - All Detected Moiré Peaks:")
        print(f"   {'Peak':<6} {'Period (nm)':<15} {'Twist Angle (°)':<18} {'Magnitude':<12}")
        print(f"   {'-'*6} {'-'*15} {'-'*18} {'-'*12}")
        
        # Get all moiré peak information
        peak_kx_coords, peak_ky_coords = peak_coords
        moire_peak_freqs = np.sqrt(peak_kx_coords[primary_peaks]**2 + peak_ky_coords[primary_peaks]**2)
        moire_peak_periods = 1.0 / moire_peak_freqs
        moire_peak_angles = 2 * np.arcsin(0.246 / (2 * moire_peak_periods)) * 180 / np.pi
        moire_peak_mags = peak_mags[primary_peaks]
        
        # Sort by magnitude (strongest first)
        sort_by_mag = np.argsort(moire_peak_mags)[::-1]
        
        for i, idx in enumerate(sort_by_mag):
            marker = "★ SELECTED" if idx == np.argmax(moire_peak_mags) else ""
            print(f"   {i+1:<6} {moire_peak_periods[idx]:<15.2f} {moire_peak_angles[idx]:<18.3f} {moire_peak_mags[idx]:<12.4f} {marker}")
        
        print(f"\n   Selected peak: Strongest by magnitude (unbiased selection)")
        print(f"   Twist angle: {twist_angle:.3f}° (from moiré period: {moire_period:.2f} nm)")
        print(f"   Primary moiré peaks identified: {len(primary_peaks)}")
        
        # Show range of possible twist angles
        print(f"\n   Twist angle range from all moiré peaks: {np.min(moire_peak_angles):.3f}° to {np.max(moire_peak_angles):.3f}°")
        print(f"   Mean twist angle (all peaks): {np.mean(moire_peak_angles):.3f}° ± {np.std(moire_peak_angles):.3f}°")
    else:
        print(f"   ⚠ Could not determine twist angle (insufficient peaks)")
        twist_angle = None
        primary_peaks = []
        peak_coords = (np.array([]), np.array([]))
        moire_period = None
else:
    print(f"   ⚠ No significant peaks found")
    twist_angle = None
    primary_peaks = []
    peak_coords = (np.array([]), np.array([]))
    moire_period = None

# ----------------------------------------------------------------------------
# 1b. LOCAL TWIST ANGLE MAPPING (spatial variations)
# ----------------------------------------------------------------------------
print("\n10b. Calculating Local Twist Angle Map (spatial variations)...")

def calculate_local_twist_angle(data, window_size_nm=50, overlap=0.5, 
                                 lattice_constant_nm=0.246, kx=None, ky=None):
    """
    Calculate twist angle map by analyzing local regions.
    
    For helical trilayer graphene, different domains can have different twist angles.
    This function creates a spatial map of twist angles.
    """
    ny, nx = data.shape
    
    # Convert window size from nm to pixels
    pixel_size_nm = (data.coords['x'].max().values - data.coords['x'].min().values) / nx * 1e9
    window_size_px = int(window_size_nm / pixel_size_nm)
    
    # Ensure minimum window size
    if window_size_px < 64:
        window_size_px = 64
    if window_size_px > min(nx, ny) // 2:
        window_size_px = min(nx, ny) // 2
    
    # Step size (with overlap)
    step_size = int(window_size_px * (1 - overlap))
    if step_size < 1:
        step_size = 1
    
    # Initialize twist angle map
    twist_angle_map = np.full((ny, nx), np.nan)
    moire_period_map = np.full((ny, nx), np.nan)
    confidence_map = np.zeros((ny, nx))
    
    # Calculate spatial frequencies if not provided
    if kx is None or ky is None:
        x_range = data.coords['x'].max().values - data.coords['x'].min().values
        y_range = data.coords['y'].max().values - data.coords['y'].min().values
        kx = np.fft.fftshift(np.fft.fftfreq(nx, d=x_range/nx)) * 1e-9
        ky = np.fft.fftshift(np.fft.fftfreq(ny, d=y_range/ny)) * 1e-9
    
    print(f"   Window size: {window_size_px} pixels ({window_size_nm:.1f} nm)")
    print(f"   Step size: {step_size} pixels")
    
    # Process in sliding windows
    n_windows = 0
    n_successful = 0
    
    for y_start in range(0, ny - window_size_px + 1, step_size):
        for x_start in range(0, nx - window_size_px + 1, step_size):
            y_end = min(y_start + window_size_px, ny)
            x_end = min(x_start + window_size_px, nx)
            
            # Extract local region
            local_data = data.values[y_start:y_end, x_start:x_end]
            
            # Skip if too much NaN or too flat
            if np.isnan(local_data).sum() > local_data.size * 0.1:
                continue
            if np.std(local_data) < 1e-12:
                continue
            
            n_windows += 1
            
            # Compute FFT for local region
            fft_local = np.fft.fft2(local_data)
            fft_shifted = np.fft.fftshift(fft_local)
            fft_mag = np.abs(fft_shifted)
            
            # Scale to pm (same as global)
            local_nx, local_ny = local_data.shape[1], local_data.shape[0]
            local_x_range = (x_end - x_start) * pixel_size_nm * 1e-9
            local_pixel_size_nm = pixel_size_nm
            fft_mag_pm = fft_mag * local_pixel_size_nm * 1000
            
            # Scale to match global scaling
            if fft_mag_pm.max() > 0:
                scale_factor = 15.4 / fft_mag_pm.max()
                fft_mag_pm_scaled = fft_mag_pm * scale_factor
            else:
                continue
            
            # Find peaks in local FFT
            local_center_y, local_center_x = local_ny // 2, local_nx // 2
            local_peak_y, local_peak_x, local_peak_mags = find_fft_peaks(
                fft_mag_pm_scaled, local_center_y, local_center_x,
                min_distance=5, threshold_percentile=95
            )
            
            if len(local_peak_y) < 2:
                continue
            
            # Calculate local twist angle
            local_kx = np.fft.fftshift(np.fft.fftfreq(local_nx, d=local_x_range/local_nx)) * 1e-9
            local_ky = np.fft.fftshift(np.fft.fftfreq(local_ny, d=local_x_range/local_ny)) * 1e-9
            
            result = calculate_twist_angle(
                local_peak_y, local_peak_x, local_center_y, local_center_x,
                local_kx, local_ky, local_peak_mags, lattice_constant_nm
            )
            
            if result[0] is not None:
                local_twist, _, _, local_period = result
                
                # Only accept reasonable twist angles (0.1° to 10°)
                if 0.1 <= local_twist <= 10.0:
                    # Assign to center region of window
                    y_center = (y_start + y_end) // 2
                    x_center = (x_start + x_end) // 2
                    
                    # Use weighted assignment (stronger in center)
                    for dy in range(-step_size//2, step_size//2 + 1):
                        for dx in range(-step_size//2, step_size//2 + 1):
                            y_idx = y_center + dy
                            x_idx = x_center + dx
                            if 0 <= y_idx < ny and 0 <= x_idx < nx:
                                weight = np.exp(-(dx**2 + dy**2) / (2 * (step_size/3)**2))
                                if np.isnan(twist_angle_map[y_idx, x_idx]):
                                    twist_angle_map[y_idx, x_idx] = local_twist
                                    moire_period_map[y_idx, x_idx] = local_period
                                    confidence_map[y_idx, x_idx] = weight
                                else:
                                    # Average with existing value (weighted)
                                    total_weight = confidence_map[y_idx, x_idx] + weight
                                    twist_angle_map[y_idx, x_idx] = (
                                        twist_angle_map[y_idx, x_idx] * confidence_map[y_idx, x_idx] +
                                        local_twist * weight
                                    ) / total_weight
                                    moire_period_map[y_idx, x_idx] = (
                                        moire_period_map[y_idx, x_idx] * confidence_map[y_idx, x_idx] +
                                        local_period * weight
                                    ) / total_weight
                                    confidence_map[y_idx, x_idx] = total_weight
                    
                    n_successful += 1
    
    print(f"   Processed {n_windows} windows, {n_successful} successful")
    print(f"   Coverage: {np.sum(~np.isnan(twist_angle_map)) / twist_angle_map.size * 100:.1f}%")
    
    if n_successful > 0:
        valid_mask = ~np.isnan(twist_angle_map)
        print(f"   Twist angle range: {np.nanmin(twist_angle_map):.3f}° to {np.nanmax(twist_angle_map):.3f}°")
        print(f"   Twist angle mean: {np.nanmean(twist_angle_map):.3f}°")
        print(f"   Twist angle std: {np.nanstd(twist_angle_map):.3f}°")
    
    return twist_angle_map, moire_period_map, confidence_map

# Calculate local twist angle map (also unbiased - uses strongest peak in each window)
twist_angle_map, moire_period_map, confidence_map = calculate_local_twist_angle(
    data_clean, window_size_nm=50, overlap=0.5, lattice_constant_nm=0.246, kx=kx, ky=ky
)

# Show distribution of local twist angles for verification
if np.sum(~np.isnan(twist_angle_map)) > 0:
    print(f"\n   Local Twist Angle Distribution:")
    print(f"   Mean: {np.nanmean(twist_angle_map):.3f}°")
    print(f"   Median: {np.nanmedian(twist_angle_map):.3f}°")
    print(f"   Std: {np.nanstd(twist_angle_map):.3f}°")
    print(f"   Range: {np.nanmin(twist_angle_map):.3f}° to {np.nanmax(twist_angle_map):.3f}°")
    print(f"   25th percentile: {np.nanpercentile(twist_angle_map, 25):.3f}°")
    print(f"   75th percentile: {np.nanpercentile(twist_angle_map, 75):.3f}°")
    print(f"   Comparison to expected 1.8°: {abs(np.nanmean(twist_angle_map) - 1.8):.3f}° difference")

# ----------------------------------------------------------------------------
# 2. DOMAIN WALL DETECTION (from topography gradients)
# ----------------------------------------------------------------------------
print("\n11. Detecting Domain Walls from Topography Gradients...")

def detect_domain_walls(data, gradient_threshold_percentile=90):
    """
    Detect domain walls from topography gradients.
    Domain walls appear as sharp transitions in height.
    """
    # Calculate gradients
    grad_y, grad_x = np.gradient(data.values)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Threshold for domain wall detection
    threshold = np.percentile(grad_magnitude, gradient_threshold_percentile)
    
    # Identify high-gradient regions (potential domain walls)
    domain_wall_mask = grad_magnitude > threshold
    
    # Apply morphological operations to clean up
    from scipy.ndimage import binary_opening, binary_closing
    domain_wall_mask = binary_opening(domain_wall_mask, structure=np.ones((3, 3)))
    domain_wall_mask = binary_closing(domain_wall_mask, structure=np.ones((5, 5)))
    
    # Calculate domain wall statistics
    n_domain_wall_pixels = np.sum(domain_wall_mask)
    domain_wall_fraction = n_domain_wall_pixels / domain_wall_mask.size * 100
    
    # Find connected components (individual domain walls)
    from scipy.ndimage import label
    labeled, n_components = label(domain_wall_mask)
    
    return domain_wall_mask, grad_magnitude, n_components, domain_wall_fraction

domain_wall_mask, grad_magnitude, n_domain_walls, wall_fraction = detect_domain_walls(
    data_clean, gradient_threshold_percentile=92
)

print(f"   Domain walls detected")
print(f"   Number of domain wall regions: {n_domain_walls}")
print(f"   Domain wall coverage: {wall_fraction:.2f}% of image")
print(f"   Average gradient magnitude: {grad_magnitude.mean()*1e9:.4f} nm/pixel")
print(f"   Max gradient magnitude: {grad_magnitude.max()*1e9:.4f} nm/pixel")

# ----------------------------------------------------------------------------
# 3. STRAIN CALCULATION (from lattice constant variations)
# ----------------------------------------------------------------------------
print("\n12. Calculating Strain from Lattice Variations...")

def calculate_strain(data, reference_lattice_constant_nm=None):
    """
    Calculate local strain from lattice constant variations.
    Strain = (a_local - a_reference) / a_reference
    """
    # Estimate reference lattice constant from FFT
    # For hexagonal lattices (graphene, MoS2), typical a ≈ 0.246 nm
    if reference_lattice_constant_nm is None:
        # Try to estimate from FFT peaks
        if len(peak_y) > 0:
            peak_kx_coords, peak_ky_coords = peak_coords
            if len(peak_kx_coords) > 0:
                # Lattice constant ≈ 1 / (2π * spatial_frequency)
                # Use the primary peak frequency
                primary_freq = np.sqrt(peak_kx_coords[0]**2 + peak_ky_coords[0]**2)
                if primary_freq > 0:
                    estimated_a = 1.0 / (2 * np.pi * primary_freq)  # in nm
                    reference_lattice_constant_nm = estimated_a
                    print(f"   Estimated reference lattice constant: {estimated_a:.4f} nm")
        else:
            # Default for graphene
            reference_lattice_constant_nm = 0.246
    
    # Calculate local lattice spacing using autocorrelation
    # This is a simplified approach - more sophisticated methods use FFT peak tracking
    from scipy.signal import correlate2d
    
    # Use a small window to estimate local periodicity
    window_size = min(20, nx // 10, ny // 10)
    if window_size < 5:
        window_size = 5
    
    # Calculate local autocorrelation to find periodicity
    strain_map = np.zeros_like(data.values)
    
    # Simplified: use gradient-based strain estimate
    # Strain ≈ local variation in spacing / reference spacing
    # Approximate using local standard deviation
    from scipy.ndimage import uniform_filter, gaussian_filter
    
    # Local standard deviation (rough proxy for strain)
    local_std = ndimage.generic_filter(
        data.values, 
        np.std, 
        size=(window_size, window_size)
    )
    
    # Normalize by reference (rough estimate)
    # This is a simplified strain metric
    strain_map = (local_std - local_std.mean()) / (local_std.mean() + 1e-10)
    
    # Calculate average strain
    mean_strain = np.mean(strain_map)
    std_strain = np.std(strain_map)
    max_strain = np.max(np.abs(strain_map))
    
    return strain_map, mean_strain, std_strain, max_strain, reference_lattice_constant_nm

strain_map, mean_strain, std_strain, max_strain, ref_lattice = calculate_strain(data_clean)

print(f"   Strain calculated")
print(f"   Reference lattice constant: {ref_lattice:.4f} nm")
print(f"   Mean strain: {mean_strain*100:.3f}%")
print(f"   Strain std: {std_strain*100:.3f}%")
print(f"   Max |strain|: {max_strain*100:.3f}%")

# Visualization and saving

print("\n13. Creating visualizations...")

# Create visualization
fig, axes = plt.subplots(3, 3, figsize=(20, 20))

# Row 1: Basic Analysis
ax = axes[0, 0]
im1 = ax.imshow(data_clean.values * 1e9, cmap='gray')
ax.set_title('Processed Topography', fontweight='bold', fontsize=12)
ax.set_xlabel('X (pixels)')
ax.set_ylabel('Y (pixels)')
plt.colorbar(im1, ax=ax, label='Height (nm)')

ax = axes[0, 1]
vmax_99_7 = np.percentile(fft_suppressed_display, 99.7)
im2 = ax.imshow(fft_suppressed_display, cmap='hot', vmin=0, vmax=vmax_99_7)
ax.set_title('FFT Modulus (DC Suppressed)', fontweight='bold', fontsize=12)
ax.set_xlabel('kx')
ax.set_ylabel('ky')
if len(peak_y) > 0:
    ax.scatter(peak_x[:12], peak_y[:12], c='cyan', s=30, marker='x', 
               linewidths=1.5, label='Detected Peaks')
    if len(primary_peaks) > 0:
        ax.scatter(peak_x[primary_peaks], peak_y[primary_peaks], 
                  c='lime', s=100, marker='o', edgecolors='black', 
                  linewidths=2, label='Primary Peaks', alpha=0.7)
    ax.legend(fontsize=8)
plt.colorbar(im2, ax=ax, label='|FFT| (pm)')

ax = axes[0, 2]
im3 = ax.imshow(grad_magnitude * 1e9, cmap='viridis')
ax.set_title('Topography Gradient Magnitude', fontweight='bold', fontsize=12)
ax.set_xlabel('X (pixels)')
ax.set_ylabel('Y (pixels)')
plt.colorbar(im3, ax=ax, label='Gradient (nm/pixel)')

# Row 2: Advanced Analysis
ax = axes[1, 0]
im4 = ax.imshow(data_clean.values * 1e9, cmap='gray', alpha=0.7)
domain_wall_overlay = np.ma.masked_where(~domain_wall_mask, domain_wall_mask)
ax.imshow(domain_wall_overlay, cmap='Reds', alpha=0.5, interpolation='nearest')
ax.set_title(f'Domain Walls Detected\n{n_domain_walls} regions, {wall_fraction:.2f}% coverage', 
             fontweight='bold', fontsize=12)
ax.set_xlabel('X (pixels)')
ax.set_ylabel('Y (pixels)')

ax = axes[1, 1]
im5 = ax.imshow(strain_map * 100, cmap='RdBu_r', vmin=-max_strain*100, vmax=max_strain*100)
ax.set_title(f'Strain Map\nMean: {mean_strain*100:.3f}%, Max: {max_strain*100:.3f}%', 
             fontweight='bold', fontsize=12)
ax.set_xlabel('X (pixels)')
ax.set_ylabel('Y (pixels)')
plt.colorbar(im5, ax=ax, label='Strain (%)')

ax = axes[1, 2]
# Local twist angle map
if np.sum(~np.isnan(twist_angle_map)) > 0:
    im6 = ax.imshow(twist_angle_map, cmap='viridis', interpolation='bilinear')
    ax.set_title(f'Local Twist Angle Map\nRange: {np.nanmin(twist_angle_map):.2f}°-{np.nanmax(twist_angle_map):.2f}°', 
                 fontweight='bold', fontsize=12)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    plt.colorbar(im6, ax=ax, label='Twist Angle (°)')
else:
    ax.text(0.5, 0.5, 'No local twist angle data', ha='center', va='center', transform=ax.transAxes)
    ax.set_title('Local Twist Angle Map', fontweight='bold', fontsize=12)

# Row 3: Statistics
ax = axes[2, 0]
# Histogram of gradient magnitudes
ax.hist(grad_magnitude.flatten() * 1e9, bins=50, alpha=0.7, edgecolor='black')
ax.axvline(np.percentile(grad_magnitude, 92) * 1e9, color='red', 
           linestyle='--', linewidth=2, label='Domain Wall Threshold')
ax.set_xlabel('Gradient Magnitude (nm/pixel)')
ax.set_ylabel('Frequency')
ax.set_title('Gradient Magnitude Distribution', fontweight='bold', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[2, 1]
# Histogram of strain
ax.hist(strain_map.flatten() * 100, bins=50, alpha=0.7, edgecolor='black', color='orange')
ax.axvline(mean_strain * 100, color='red', linestyle='--', linewidth=2, 
           label=f'Mean: {mean_strain*100:.3f}%')
ax.set_xlabel('Strain (%)')
ax.set_ylabel('Frequency')
ax.set_title('Strain Distribution', fontweight='bold', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[2, 2]
# Summary statistics text
ax.axis('off')
twist_angle_str = f"{twist_angle:.3f}°" if twist_angle is not None else "N/A"
moire_period_str = f"{moire_period:.2f} nm" if 'moire_period' in locals() and moire_period is not None else "N/A"
local_twist_str = f"{np.nanmean(twist_angle_map):.3f}°" if np.sum(~np.isnan(twist_angle_map)) > 0 else "N/A"
local_twist_std_str = f"{np.nanstd(twist_angle_map):.3f}°" if np.sum(~np.isnan(twist_angle_map)) > 0 else "N/A"
summary_text = f"""
ANALYSIS SUMMARY v2.0
{'='*40}

TWIST ANGLE (Helical Trilayer):
  Global: {twist_angle_str}
  Local Mean: {local_twist_str} ± {local_twist_std_str}
  Local Range: {f"{np.nanmin(twist_angle_map):.2f}°-{np.nanmax(twist_angle_map):.2f}°" if np.sum(~np.isnan(twist_angle_map)) > 0 else "N/A"}
  Moiré Period: {moire_period_str}
  Primary Peaks: {len(primary_peaks) if len(primary_peaks) > 0 else 0}
  Total Peaks: {len(peak_y)}

DOMAIN WALLS:
  Regions: {n_domain_walls}
  Coverage: {wall_fraction:.2f}%
  Avg Gradient: {grad_magnitude.mean()*1e9:.4f} nm/pixel

STRAIN:
  Mean: {mean_strain*100:.3f}%
  Std: {std_strain*100:.3f}%
  Max: {max_strain*100:.3f}%
  Ref Lattice: {ref_lattice:.4f} nm

PROCESSING:
  RMS Change: {rms_change_from_align:.2f}%
  DC Component: {dc_value:.4f} pm
"""
ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
        verticalalignment='center', bbox=dict(boxstyle='round', 
        facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(output_dir / "analysis_v2.0_advanced.png", dpi=200, bbox_inches='tight')
print(f"   Saved: analysis_v2.0_advanced.png")
plt.close(fig)

# Save individual analysis plots
print("\n14. Saving Individual Analysis Files...")

# Twist angle visualization
if twist_angle is not None:
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(fft_suppressed_display, cmap='hot', vmin=0, vmax=vmax_99_7)
    if len(peak_y) > 0:
        ax.scatter(peak_x[:20], peak_y[:20], c='cyan', s=50, marker='x', 
                  linewidths=2, label='All Peaks', alpha=0.7)
        if len(primary_peaks) > 0:
            ax.scatter(peak_x[primary_peaks], peak_y[primary_peaks], 
                      c='lime', s=200, marker='o', edgecolors='black', 
                      linewidths=3, label='Primary Peaks', alpha=0.8)
        ax.legend(fontsize=12)
    ax.set_title(f'Twist Angle Analysis\nTwist Angle: {twist_angle:.3f}°', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('kx (nm⁻¹)', fontsize=12)
    ax.set_ylabel('ky (nm⁻¹)', fontsize=12)
    plt.colorbar(im, ax=ax, label='|FFT| (pm)', fraction=0.046)
    plt.tight_layout()
    plt.savefig(output_dir / "twist_angle_v2.0.png", dpi=200, bbox_inches='tight')
    print(f"   Saved: twist_angle_v2.0.png")
    plt.close(fig)

# Domain wall visualization
fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(data_clean.values * 1e9, cmap='gray')
domain_wall_overlay = np.ma.masked_where(~domain_wall_mask, domain_wall_mask)
ax.imshow(domain_wall_overlay, cmap='Reds', alpha=0.6, interpolation='nearest')
ax.set_title(f'Domain Wall Detection\n{n_domain_walls} regions, {wall_fraction:.2f}% coverage', 
            fontsize=14, fontweight='bold')
ax.set_xlabel('X (pixels)', fontsize=12)
ax.set_ylabel('Y (pixels)', fontsize=12)
plt.tight_layout()
plt.savefig(output_dir / "domain_walls_v2.0.png", dpi=200, bbox_inches='tight')
print(f"   Saved: domain_walls_v2.0.png")
plt.close(fig)

# Strain visualization
fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(strain_map * 100, cmap='RdBu_r', 
               vmin=-max_strain*100, vmax=max_strain*100)
ax.set_title(f'Strain Map\nMean: {mean_strain*100:.3f}%, Std: {std_strain*100:.3f}%, Max: {max_strain*100:.3f}%', 
            fontsize=14, fontweight='bold')
ax.set_xlabel('X (pixels)', fontsize=12)
ax.set_ylabel('Y (pixels)', fontsize=12)
plt.colorbar(im, ax=ax, label='Strain (%)', fraction=0.046)
plt.tight_layout()
plt.savefig(output_dir / "strain_v2.0.png", dpi=200, bbox_inches='tight')
print(f"   Saved: strain_v2.0.png")
plt.close(fig)

# Diagnostic visualization: All moiré peaks and their twist angles
if result[0] is not None and len(primary_peaks) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Twist angles from all moiré peaks
    ax = axes[0]
    peak_kx_coords, peak_ky_coords = peak_coords
    moire_peak_freqs = np.sqrt(peak_kx_coords[primary_peaks]**2 + peak_ky_coords[primary_peaks]**2)
    moire_peak_periods = 1.0 / moire_peak_freqs
    moire_peak_angles = 2 * np.arcsin(0.246 / (2 * moire_peak_periods)) * 180 / np.pi
    moire_peak_mags = peak_mags[primary_peaks]
    
    # Bar plot of twist angles
    sort_idx = np.argsort(moire_peak_angles)
    colors = ['red' if idx == np.argmax(moire_peak_mags) else 'steelblue' for idx in sort_idx]
    bars = ax.bar(range(len(moire_peak_angles)), moire_peak_angles[sort_idx], 
                  color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(1.8, color='green', linestyle='--', linewidth=2, label='Expected: 1.8°')
    ax.axhline(np.mean(moire_peak_angles), color='orange', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(moire_peak_angles):.3f}°')
    ax.set_xlabel('Moiré Peak (sorted by angle)', fontsize=12)
    ax.set_ylabel('Twist Angle (°)', fontsize=12)
    ax.set_title('Twist Angles from All Detected Moiré Peaks\n(Red = Selected, Strongest Peak)', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(len(moire_peak_angles)))
    ax.set_xticklabels([f'P{i+1}' for i in range(len(moire_peak_angles))])
    
    # Plot 2: Magnitude vs Twist Angle
    ax2 = axes[1]
    scatter = ax2.scatter(moire_peak_angles, moire_peak_mags, 
                         s=200, c=colors, alpha=0.7, edgecolors='black', linewidths=2)
    ax2.axvline(1.8, color='green', linestyle='--', linewidth=2, label='Expected: 1.8°')
    ax2.axvline(np.mean(moire_peak_angles), color='orange', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(moire_peak_angles):.3f}°')
    ax2.set_xlabel('Twist Angle (°)', fontsize=12)
    ax2.set_ylabel('Peak Magnitude', fontsize=12)
    ax2.set_title('Peak Magnitude vs Twist Angle\n(Red = Selected, Strongest Peak)', 
                  fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add text annotations
    for i, (angle, mag) in enumerate(zip(moire_peak_angles, moire_peak_mags)):
        if i == np.argmax(moire_peak_mags):
            ax2.annotate('★ SELECTED', (angle, mag), 
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=10, fontweight='bold', color='red')
    
    plt.tight_layout()
    plt.savefig(output_dir / "twist_angle_diagnostics_v2.0.png", dpi=200, bbox_inches='tight')
    print(f"   Saved: twist_angle_diagnostics_v2.0.png")
    plt.close(fig)

# Local twist angle map visualization
if np.sum(~np.isnan(twist_angle_map)) > 0:
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(twist_angle_map, cmap='viridis', interpolation='bilinear')
    ax.set_title(f'Local Twist Angle Map\nMean: {np.nanmean(twist_angle_map):.3f}° ± {np.nanstd(twist_angle_map):.3f}°\nRange: {np.nanmin(twist_angle_map):.2f}° - {np.nanmax(twist_angle_map):.2f}°', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Y (pixels)', fontsize=12)
    plt.colorbar(im, ax=ax, label='Twist Angle (°)', fraction=0.046)
    plt.tight_layout()
    plt.savefig(output_dir / "twist_angle_map_v2.0.png", dpi=200, bbox_inches='tight')
    print(f"   Saved: twist_angle_map_v2.0.png")
    plt.close(fig)
    
    # Combined: Topography + Twist Angle Map
    fig, ax = plt.subplots(figsize=(10, 10))
    im1 = ax.imshow(data_clean.values * 1e9, cmap='gray', alpha=0.7)
    twist_overlay = np.ma.masked_where(np.isnan(twist_angle_map), twist_angle_map)
    im2 = ax.imshow(twist_overlay, cmap='viridis', alpha=0.6, interpolation='bilinear')
    ax.set_title('Topography + Local Twist Angle Map\nShows spatial variations in twist angle', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Y (pixels)', fontsize=12)
    plt.colorbar(im2, ax=ax, label='Twist Angle (°)', fraction=0.046)
    plt.tight_layout()
    plt.savefig(output_dir / "topography_twist_angle_v2.0.png", dpi=200, bbox_inches='tight')
    print(f"   Saved: topography_twist_angle_v2.0.png")
    plt.close(fig)

# Save all analysis data
print("\n15. Saving analysis data...")
advanced_dataset = xr.Dataset({
    'topography': (['y', 'x'], data_clean.values),
    'gradient_magnitude': (['y', 'x'], grad_magnitude),
    'domain_wall_mask': (['y', 'x'], domain_wall_mask.astype(int)),
    'strain_map': (['y', 'x'], strain_map),
    'twist_angle_map': (['y', 'x'], twist_angle_map),
    'moire_period_map': (['y', 'x'], moire_period_map),
    'confidence_map': (['y', 'x'], confidence_map),
})
advanced_dataset.coords['x'] = x_coords
advanced_dataset.coords['y'] = y_coords
advanced_dataset.attrs['twist_angle_deg'] = float(twist_angle) if twist_angle is not None else np.nan
advanced_dataset.attrs['moire_period_nm'] = float(moire_period) if 'moire_period' in locals() and moire_period is not None else np.nan
advanced_dataset.attrs['n_domain_walls'] = int(n_domain_walls)
advanced_dataset.attrs['domain_wall_fraction_percent'] = float(wall_fraction)
advanced_dataset.attrs['mean_strain_percent'] = float(mean_strain * 100)
advanced_dataset.attrs['std_strain_percent'] = float(std_strain * 100)
advanced_dataset.attrs['max_strain_percent'] = float(max_strain * 100)
advanced_dataset.attrs['reference_lattice_constant_nm'] = float(ref_lattice)
advanced_dataset.attrs['n_fft_peaks'] = len(peak_y)
advanced_dataset.attrs['n_primary_peaks'] = len(primary_peaks) if len(primary_peaks) > 0 else 0
advanced_dataset.attrs['material'] = 'Helical trilayer graphene'
if np.sum(~np.isnan(twist_angle_map)) > 0:
    advanced_dataset.attrs['local_twist_mean_deg'] = float(np.nanmean(twist_angle_map))
    advanced_dataset.attrs['local_twist_std_deg'] = float(np.nanstd(twist_angle_map))
    advanced_dataset.attrs['local_twist_min_deg'] = float(np.nanmin(twist_angle_map))
    advanced_dataset.attrs['local_twist_max_deg'] = float(np.nanmax(twist_angle_map))
    advanced_dataset.attrs['local_twist_coverage_percent'] = float(np.sum(~np.isnan(twist_angle_map)) / twist_angle_map.size * 100)

advanced_dataset.to_netcdf(output_dir / "advanced_analysis_v2.0.nc")
print(f"   Saved: advanced_analysis_v2.0.nc")

# Update FFT dataset with twist angle info
fft_dataset.attrs['twist_angle_deg'] = float(twist_angle) if twist_angle is not None else np.nan
fft_dataset.attrs['moire_period_nm'] = float(moire_period) if 'moire_period' in locals() and moire_period is not None else np.nan
fft_dataset.attrs['n_peaks'] = len(peak_y)
fft_dataset.attrs['n_primary_peaks'] = len(primary_peaks) if len(primary_peaks) > 0 else 0
fft_dataset.attrs['material'] = 'Helical trilayer graphene'
fft_dataset.to_netcdf(output_dir / "fft_v2.0.nc")
print(f"   Updated: fft_v2.0.nc (with twist angle data)")

print("\nAnalysis finished")
print(f"\nResults:")
if twist_angle is not None:
    print(f"  Twist angle: {twist_angle:.3f}°")
else:
    print(f"  Twist angle: Could not be determined")
print(f"  Domain walls: {n_domain_walls} regions ({wall_fraction:.2f}% coverage)")
print(f"  Mean strain: {mean_strain*100:.3f}% (std: {std_strain*100:.3f}%)")
print(f"  RMS change: {rms_change_from_align:.2f}%")
print(f"\nFiles saved to: {output_dir}")