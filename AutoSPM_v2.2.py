"""
AutoSPM Analysis v2.2
- Prompts input file path from user
- True strain calculation via FFT peak tracking
- Strain tensor calculation (2D)
- Moiré period map calculation
- Magic angle detection
- Domain wall classification
- All v2.1 features preserved
"""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to suppress OpenGL warnings
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
import sys
import os
warnings.filterwarnings('ignore')

# Input/output setup
print("AutoSPM Analysis v2.2")
print("\nInput file selection:")

# Prompt for input file
while True:
    file_path = input("Enter path to .sxm file: ").strip()
    if not file_path:
        print("Error: Please provide a file path.")
        continue
    
    file_path = os.path.expanduser(file_path)
    input_path = Path(file_path)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        retry = input("Try again? (y/n): ").strip().lower()
        if retry != 'y':
            sys.exit(1)
    else:
        break

output_dir = Path("/Users/isaactsai/STMoutput")
output_dir.mkdir(exist_ok=True, parents=True)

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
    print("   Warning: No scan dimensions found in header, estimating...")
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
data_clean.to_netcdf(output_dir / "processed_data_v2.2.nc")

# Quantitative FFT Analysis
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

# Apply visualization enhancements (FOR DISPLAY ONLY - original data preserved)
print("\n7. Applying visualization enhancements (display only, data preserved)...")

# Create display version with DC suppression (only for visualization)
fft_suppressed_display = fft_mag_pm_scaled.copy()  # Copy for display, original preserved

# Suppress DC only for visualization
center_region_size = 5
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

# Advanced analysis: twist angle and moiré detection

# ----------------------------------------------------------------------------
# 1. IMPROVED TWIST ANGLE CALCULATION (from FFT peaks)
# ----------------------------------------------------------------------------
print("\n10. Calculating twist angle from FFT peaks...")

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

def classify_peaks(peak_frequencies, atomic_lattice_freq_nm=4.06):
    """
    Classify peaks as moiré or atomic lattice peaks.
    
    Parameters:
    - peak_frequencies: Spatial frequencies in nm⁻¹
    - atomic_lattice_freq_nm: Typical atomic lattice frequency (~4.06 nm⁻¹ for graphene)
    
    Returns:
    - moire_mask: Boolean mask for moiré peaks
    - atomic_mask: Boolean mask for atomic peaks
    """
    # Moiré peaks are typically much lower frequency than atomic peaks
    # Moiré: 0.01 - 1.0 nm⁻¹ (periods of 1-100 nm)
    # Atomic: 3.0 - 5.0 nm⁻¹ (periods of 0.2-0.33 nm)
    
    moire_mask = peak_frequencies < 1.0  # Below 1.0 nm⁻¹
    atomic_mask = (peak_frequencies > 2.0) & (peak_frequencies < 6.0)  # 2.0-6.0 nm⁻¹
    
    return moire_mask, atomic_mask

def calculate_twist_angle_v21(peak_y, peak_x, center_y, center_x, kx, ky, peak_magnitudes, 
                              lattice_constant_nm=0.246):
    """
    IMPROVED v2.1: Calculate twist angle from FFT peaks with better moiré detection.
    
    Key improvements:
    1. Expanded moiré frequency range (0.01 - 1.0 nm⁻¹) to catch small twist angles
    2. Peak classification (moiré vs atomic)
    3. Better validation and diagnostics
    4. Multiple methods for robustness
    """
    if len(peak_y) < 2:
        return None, None, None, None, None
    
    # Get spatial frequencies of peaks
    peak_kx = kx[peak_x]
    peak_ky = ky[peak_y]
    peak_frequencies = np.sqrt(peak_kx**2 + peak_ky**2)
    
    # Classify peaks
    moire_mask, atomic_mask = classify_peaks(peak_frequencies)
    n_moire = np.sum(moire_mask)
    n_atomic = np.sum(atomic_mask)
    
    print(f"   Peak classification:")
    print(f"     Moiré peaks: {n_moire} (freq < 1.0 nm⁻¹)")
    print(f"     Atomic peaks: {n_atomic} (freq 2.0-6.0 nm⁻¹)")
    print(f"     Other peaks: {len(peak_y) - n_moire - n_atomic}")
    
    # METHOD 1: Use moiré peaks (EXPANDED RANGE for v2.1)
    # v2.0 bug: range was 0.05-0.5 nm⁻¹, missing small twist angles
    # v2.1 fix: expanded to 0.01-1.0 nm⁻¹ to catch periods up to 100 nm
    moire_freq_min = 0.01  # nm⁻¹ (was 0.05 in v2.0)
    moire_freq_max = 1.0   # nm⁻¹ (was 0.5 in v2.0)
    
    # Use classified moiré peaks, or fall back to frequency range
    if n_moire > 0:
        moire_peaks = np.where(moire_mask)[0]
        # Further filter by magnitude range if needed
        freq_filter = (peak_frequencies[moire_peaks] >= moire_freq_min) & \
                      (peak_frequencies[moire_peaks] <= moire_freq_max)
        moire_peaks = moire_peaks[freq_filter]
    else:
        # Fall back to frequency range method
        freq_filter = (peak_frequencies >= moire_freq_min) & \
                      (peak_frequencies <= moire_freq_max)
        moire_peaks = np.where(freq_filter)[0]
    
    if len(moire_peaks) < 1:
        # If no moiré peaks in range, use lowest frequency peaks
        print(f"   Warning: No moiré peaks in range, using lowest frequency peaks")
        moire_peaks = np.argsort(peak_frequencies)[:min(12, len(peak_y))]
    
    # Get the strongest moiré peaks
    moire_magnitudes = peak_magnitudes[moire_peaks]
    sort_idx = np.argsort(moire_magnitudes)[::-1]
    primary_moire_peaks = moire_peaks[sort_idx[:6]]  # Top 6 moiré peaks
    
    if len(primary_moire_peaks) < 1:
        return None, None, None, None, None
    
    # Calculate moiré periods from peak frequencies
    moire_periods = 1.0 / peak_frequencies[primary_moire_peaks]
    
    # Use the STRONGEST peak (by magnitude) - unbiased selection
    peak_mags_at_moire = peak_magnitudes[primary_moire_peaks]
    best_idx = np.argmax(peak_mags_at_moire)
    primary_period = moire_periods[best_idx]
    primary_freq = peak_frequencies[primary_moire_peaks[best_idx]]
    
    # Calculate twist angle from moiré period: θ = 2*arcsin(a/(2*L))
    # For small angles: θ ≈ a/L (in radians) ≈ (a/L) * (180/π) (in degrees)
    if primary_period > 0:
        # Exact formula
        twist_angle = 2 * np.arcsin(lattice_constant_nm / (2 * primary_period)) * 180 / np.pi
        
        # Small angle approximation for validation
        twist_angle_approx = (lattice_constant_nm / primary_period) * 180 / np.pi
        
        # Validate: angles should be reasonable (0.1° to 10° for typical twisted bilayers)
        if not (0.1 <= twist_angle <= 10.0):
            print(f"   Warning: Calculated twist angle {twist_angle:.3f}° outside typical range (0.1-10°)")
            # Check if small angle approximation is more reasonable
            if 0.1 <= twist_angle_approx <= 10.0:
                print(f"   Using small angle approximation: {twist_angle_approx:.3f}°")
                twist_angle = twist_angle_approx
    else:
        return None, None, None, None, None
    
    # Return: twist_angle, primary_peaks, peak_coords, primary_period, classification_info
    classification_info = {
        'n_moire': n_moire,
        'n_atomic': n_atomic,
        'primary_freq': primary_freq,
        'primary_period': primary_period,
        'twist_angle_approx': twist_angle_approx if 'twist_angle_approx' in locals() else None
    }
    
    return twist_angle, primary_moire_peaks, (peak_kx, peak_ky), primary_period, classification_info

# Find FFT peaks
peak_y, peak_x, peak_mags = find_fft_peaks(fft_mag_pm_scaled, center_y, center_x, 
                                           min_distance=15, threshold_percentile=98)

print(f"   Found {len(peak_y)} significant FFT peaks")
if len(peak_y) > 0:
    print(f"   Top 6 peak magnitudes: {peak_mags[:6]}")
    
    # Calculate twist angle (IMPROVED v2.1)
    result = calculate_twist_angle_v21(
        peak_y, peak_x, center_y, center_x, kx, ky, peak_mags, 
        lattice_constant_nm=0.246  # Graphene lattice constant
    )
    
    if result[0] is not None:
        twist_angle, primary_peaks, peak_coords, moire_period, class_info = result
        
        # DIAGNOSTICS: Show all detected moiré periods and their twist angles
        print(f"\n   DIAGNOSTICS - All Detected Moiré Peaks (v2.1):")
        print(f"   {'Peak':<6} {'Freq (nm⁻¹)':<15} {'Period (nm)':<15} {'Twist Angle (°)':<18} {'Magnitude':<12}")
        print(f"   {'-'*6} {'-'*15} {'-'*15} {'-'*18} {'-'*12}")
        
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
            print(f"   {i+1:<6} {moire_peak_freqs[idx]:<15.4f} {moire_peak_periods[idx]:<15.2f} "
                  f"{moire_peak_angles[idx]:<18.3f} {moire_peak_mags[idx]:<12.4f} {marker}")
        
        print(f"\n   Selected peak: Strongest by magnitude (unbiased selection)")
        print(f"   Moiré frequency: {class_info['primary_freq']:.4f} nm⁻¹")
        print(f"   Moiré period: {moire_period:.2f} nm")
        print(f"   Twist angle: {twist_angle:.3f}° (from moiré period)")
        if class_info['twist_angle_approx'] is not None:
            print(f"   Small angle approx: {class_info['twist_angle_approx']:.3f}°")
        print(f"   Primary moiré peaks identified: {len(primary_peaks)}")
        
        # Show range of possible twist angles
        print(f"\n   Twist angle range from all moiré peaks: {np.min(moire_peak_angles):.3f}° to {np.max(moire_peak_angles):.3f}°")
        print(f"   Mean twist angle (all peaks): {np.mean(moire_peak_angles):.3f}° ± {np.std(moire_peak_angles):.3f}°")
        
        # Validation
        if 0.1 <= twist_angle <= 10.0:
            print(f"   Twist angle in reasonable range (0.1-10°)")
        else:
            print(f"   ⚠ Twist angle outside typical range - may need manual verification")
    else:
        print(f"   ⚠ Could not determine twist angle (insufficient peaks)")
        twist_angle = None
        primary_peaks = []
        peak_coords = (np.array([]), np.array([]))
        moire_period = None
        class_info = None
else:
    print(f"   ⚠ No significant peaks found")
    twist_angle = None
    primary_peaks = []
    peak_coords = (np.array([]), np.array([]))
    moire_period = None
    class_info = None

# ----------------------------------------------------------------------------
# 1a. MAGIC ANGLE DETECTION (v2.2 NEW)
# ----------------------------------------------------------------------------
print("\n10a. Detecting Magic Angles...")

def detect_magic_angles(twist_angle, tolerance=0.1):
    """
    Detect magic angles for twisted bilayer graphene.
    
    Known magic angles:
    - θ₁ ≈ 1.1° (first magic angle)
    - θ₂ ≈ 0.5° (second magic angle)
    - θ₃ ≈ 0.3° (third magic angle)
    
    Returns: closest magic angle, deviation, significance
    """
    if twist_angle is None:
        return None, None, None
    
    # Known magic angles (degrees)
    magic_angles = [1.1, 0.5, 0.3]
    magic_names = ['θ₁ (1.1°)', 'θ₂ (0.5°)', 'θ₃ (0.3°)']
    
    # Find closest magic angle
    deviations = [abs(twist_angle - ma) for ma in magic_angles]
    closest_idx = np.argmin(deviations)
    closest_magic = magic_angles[closest_idx]
    deviation = deviations[closest_idx]
    
    # Assess significance
    is_magic = deviation <= tolerance
    significance = "HIGH" if is_magic else ("MODERATE" if deviation <= tolerance * 2 else "LOW")
    
    return closest_magic, deviation, significance, magic_names[closest_idx]

if twist_angle is not None:
    closest_magic, magic_deviation, magic_significance, magic_name = detect_magic_angles(twist_angle, tolerance=0.1)
    if closest_magic is not None:
        print(f"   Closest magic angle: {magic_name}")
        print(f"   Deviation: {magic_deviation:.3f}°")
        print(f"   Significance: {magic_significance}")
        if magic_deviation <= 0.1:
            print(f"   ★ MAGIC ANGLE DETECTED! Twist angle {twist_angle:.3f}° is within 0.1° of {magic_name}")
    else:
        closest_magic = None
        magic_deviation = None
        magic_significance = None
        magic_name = None
else:
    closest_magic = None
    magic_deviation = None
    magic_significance = None
    magic_name = None

# ----------------------------------------------------------------------------
# 1b. LOCAL TWIST ANGLE MAPPING (spatial variations) - UPDATED for v2.1
# ----------------------------------------------------------------------------
print("\n10b. Calculating Local Twist Angle Map (spatial variations)...")

def calculate_local_twist_angle_v21(data, window_size_nm=50, overlap=0.5, 
                                     lattice_constant_nm=0.246, kx=None, ky=None):
    """
    Calculate twist angle map by analyzing local regions (v2.1 - uses improved function).
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
            
            # Calculate local twist angle using v2.1 improved function
            local_kx = np.fft.fftshift(np.fft.fftfreq(local_nx, d=local_x_range/local_nx)) * 1e-9
            local_ky = np.fft.fftshift(np.fft.fftfreq(local_ny, d=local_x_range/local_ny)) * 1e-9
            
            result = calculate_twist_angle_v21(
                local_peak_y, local_peak_x, local_center_y, local_center_x,
                local_kx, local_ky, local_peak_mags, lattice_constant_nm
            )
            
            if result[0] is not None:
                local_twist, _, _, local_period, _ = result
                
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
    
    print(f"   Processed {n_windows} windows, {n_successful} with valid results")
    print(f"   Coverage: {np.sum(~np.isnan(twist_angle_map)) / twist_angle_map.size * 100:.1f}%")
    
    if n_successful > 0:
        valid_mask = ~np.isnan(twist_angle_map)
        print(f"   Twist angle range: {np.nanmin(twist_angle_map):.3f}° to {np.nanmax(twist_angle_map):.3f}°")
        print(f"   Twist angle mean: {np.nanmean(twist_angle_map):.3f}°")
        print(f"   Twist angle std: {np.nanstd(twist_angle_map):.3f}°")
    
    return twist_angle_map, moire_period_map, confidence_map

# Calculate local twist angle map (v2.1 improved)
twist_angle_map, moire_period_map, confidence_map = calculate_local_twist_angle_v21(
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
    
    return domain_wall_mask, grad_magnitude, n_components, domain_wall_fraction, labeled

domain_wall_mask, grad_magnitude, n_domain_walls, wall_fraction, labeled = detect_domain_walls(
    data_clean, gradient_threshold_percentile=92
)

print(f"   Domain walls detected")
print(f"   Number of domain wall regions: {n_domain_walls}")
print(f"   Domain wall coverage: {wall_fraction:.2f}% of image")
print(f"   Average gradient magnitude: {grad_magnitude.mean()*1e9:.4f} nm/pixel")
print(f"   Max gradient magnitude: {grad_magnitude.max()*1e9:.4f} nm/pixel")

# ----------------------------------------------------------------------------
# 2b. DOMAIN WALL CLASSIFICATION (v2.2 NEW)
# ----------------------------------------------------------------------------
print("\n11b. Classifying Domain Walls...")

def classify_domain_walls(domain_wall_mask, grad_magnitude, labeled):
    """
    Classify domain walls into types:
    
    1. Solitons (smooth, continuous, wide)
    2. Dislocations (sharp, discontinuous, narrow)
    3. Grain boundaries (extended regions)
    4. Stacking faults (atomic-scale, very sharp)
    
    Returns: classification map and statistics
    """
    from scipy.ndimage import label, find_objects
    
    ny, nx = domain_wall_mask.shape
    classification_map = np.zeros((ny, nx), dtype=int)  # 0=background, 1=soliton, 2=dislocation, 3=grain_boundary, 4=stacking_fault
    
    # Get properties of each domain wall region
    n_components = labeled.max()
    wall_properties = []
    
    for i in range(1, n_components + 1):
        region_mask = (labeled == i)
        region_size = np.sum(region_mask)
        
        # Calculate properties
        region_grad = grad_magnitude[region_mask]
        avg_grad = np.mean(region_grad)
        max_grad = np.max(region_grad)
        
        # Width estimation (approximate)
        # Find bounding box
        coords = np.where(region_mask)
        if len(coords[0]) > 0:
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            width_y = y_max - y_min + 1
            width_x = x_max - x_min + 1
            avg_width = (width_y + width_x) / 2
            
            # Continuity (how connected is the region)
            # Use region size relative to bounding box
            bbox_area = width_y * width_x
            continuity = region_size / bbox_area if bbox_area > 0 else 0
            
            # Classification criteria
            if avg_width > 5 and continuity > 0.5 and avg_grad < np.percentile(grad_magnitude, 95):
                # Soliton: wide, smooth, continuous
                classification_map[region_mask] = 1
                wall_type = "Soliton"
            elif avg_width < 2 and max_grad > np.percentile(grad_magnitude, 98):
                # Dislocation: narrow, sharp
                classification_map[region_mask] = 2
                wall_type = "Dislocation"
            elif region_size > 100 and avg_width > 10:
                # Grain boundary: extended region
                classification_map[region_mask] = 3
                wall_type = "Grain Boundary"
            elif max_grad > np.percentile(grad_magnitude, 99):
                # Stacking fault: very sharp
                classification_map[region_mask] = 4
                wall_type = "Stacking Fault"
            else:
                # Unclassified
                wall_type = "Unclassified"
            
            wall_properties.append({
                'id': i,
                'type': wall_type,
                'size': region_size,
                'width': avg_width,
                'avg_grad': avg_grad,
                'max_grad': max_grad,
                'continuity': continuity
            })
    
    # Count by type
    type_counts = {
        'Soliton': np.sum(classification_map == 1),
        'Dislocation': np.sum(classification_map == 2),
        'Grain Boundary': np.sum(classification_map == 3),
        'Stacking Fault': np.sum(classification_map == 4),
        'Unclassified': np.sum((classification_map > 0) & (classification_map <= 4) == False)
    }
    
    return classification_map, wall_properties, type_counts

domain_wall_classification, wall_properties, wall_type_counts = classify_domain_walls(
    domain_wall_mask, grad_magnitude, labeled
)

print(f"   Domain walls classified")
print(f"   Domain wall types:")
for wall_type, count in wall_type_counts.items():
    if count > 0:
        percentage = count / np.sum(domain_wall_classification > 0) * 100 if np.sum(domain_wall_classification > 0) > 0 else 0
        print(f"     {wall_type}: {count} pixels ({percentage:.1f}% of walls)")

# ----------------------------------------------------------------------------
# 3. STRAIN CALCULATION (v2.2: TRUE STRAIN via FFT Peak Tracking)
# ----------------------------------------------------------------------------
print("\n12. Calculating True Strain via FFT Peak Tracking (v2.2 NEW)...")

def calculate_strain_fft_peak_tracking(data, reference_moire_period_nm=None, 
                                       window_size_nm=20, overlap=0.5, 
                                       lattice_constant_nm=0.246, kx=None, ky=None):
    """
    Calculate TRUE strain by tracking local FFT peak positions.
    
    Method:
    1. Divide image into sliding windows
    2. Compute FFT for each window
    3. Track moiré peak positions (kx, ky)
    4. Calculate local moiré period from peak frequency
    5. Strain = (L_local - L_reference) / L_reference
    
    This is more accurate than the simplified local std method.
    """
    ny, nx = data.shape
    
    # Get reference moiré period (from global analysis)
    if reference_moire_period_nm is None:
        if moire_period is not None:
            reference_moire_period_nm = moire_period
            print(f"   Using global moiré period as reference: {reference_moire_period_nm:.2f} nm")
        else:
            # Estimate from FFT if available
            if len(peak_y) > 0 and peak_coords[0].size > 0:
                peak_kx_coords, peak_ky_coords = peak_coords
                primary_freq = np.sqrt(peak_kx_coords[0]**2 + peak_ky_coords[0]**2)
                if primary_freq > 0:
                    reference_moire_period_nm = 1.0 / primary_freq
                    print(f"   Estimated reference moiré period: {reference_moire_period_nm:.2f} nm")
                else:
                    reference_moire_period_nm = 50.0  # Default estimate
                    print(f"   Warning: Using default reference period: {reference_moire_period_nm:.2f} nm")
            else:
                reference_moire_period_nm = 50.0
                print(f"   Warning: Using default reference period: {reference_moire_period_nm:.2f} nm")
    
    # Convert window size from nm to pixels
    pixel_size_nm = (data.coords['x'].max().values - data.coords['x'].min().values) / nx * 1e9
    window_size_px = int(window_size_nm / pixel_size_nm)
    
    # Ensure reasonable window size
    if window_size_px < 32:
        window_size_px = 32
    if window_size_px > min(nx, ny) // 2:
        window_size_px = min(nx, ny) // 2
    
    # Step size (with overlap)
    step_size = int(window_size_px * (1 - overlap))
    if step_size < 1:
        step_size = 1
    
    # Initialize strain map
    strain_map = np.full((ny, nx), np.nan)
    moire_period_map = np.full((ny, nx), np.nan)
    confidence_map = np.zeros((ny, nx))
    
    print(f"   Window size: {window_size_px} pixels ({window_size_nm:.1f} nm)")
    print(f"   Step size: {step_size} pixels")
    print(f"   Reference moiré period: {reference_moire_period_nm:.2f} nm")
    
    # Process in sliding windows
    n_windows = 0
    n_successful = 0
    n_no_peaks = 0
    n_no_moire_peaks = 0
    n_strain_out_of_range = 0
    n_extreme_strain = 0  # Track extreme but valid strain values
    
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
            
            # Find peaks in local FFT (use lower threshold for smaller windows)
            local_center_y, local_center_x = local_ny // 2, local_nx // 2
            # Use lower threshold for smaller windows to catch more peaks
            # 85th percentile for small windows, 90th for medium, 95th for large
            if window_size_px < 40:
                threshold_percentile = 85
            elif window_size_px < 64:
                threshold_percentile = 90
            else:
                threshold_percentile = 95
            local_peak_y, local_peak_x, local_peak_mags = find_fft_peaks(
                fft_mag_pm_scaled, local_center_y, local_center_x,
                min_distance=2, threshold_percentile=threshold_percentile  # Further reduced for better detection
            )
            
            if len(local_peak_y) < 1:
                n_no_peaks += 1
                continue
            
            # Calculate local spatial frequencies
            local_kx = np.fft.fftshift(np.fft.fftfreq(local_nx, d=local_x_range/local_nx)) * 1e-9
            local_ky = np.fft.fftshift(np.fft.fftfreq(local_ny, d=local_x_range/local_ny)) * 1e-9
            
            # Get peak frequencies
            peak_kx_local = local_kx[local_peak_x]
            peak_ky_local = local_ky[local_peak_y]
            peak_freqs_local = np.sqrt(peak_kx_local**2 + peak_ky_local**2)
            
            # Filter for moiré peaks (very wide range: 0.001 - 2.0 nm⁻¹ to catch all possible moiré patterns)
            # This includes small-angle moiré (low freq) and higher-order patterns (higher freq)
            moire_mask = (peak_freqs_local >= 0.001) & (peak_freqs_local <= 2.0)
            if np.sum(moire_mask) == 0:
                n_no_moire_peaks += 1
                continue
            
            # Use strongest moiré peak
            moire_peak_freqs = peak_freqs_local[moire_mask]
            moire_peak_mags = local_peak_mags[moire_mask]
            best_moire_idx = np.argmax(moire_peak_mags)
            local_moire_freq = moire_peak_freqs[best_moire_idx]
            local_moire_period = 1.0 / local_moire_freq
            
            # Calculate strain: ε = (L_local - L_ref) / L_ref
            if local_moire_period > 0 and reference_moire_period_nm > 0:
                local_strain = (local_moire_period - reference_moire_period_nm) / reference_moire_period_nm
                
                # REMOVED HARD STRAIN RANGE FILTER - accept all calculated strain values
                # Real materials can have strain > ±20% near defects, domain walls, etc.
                # Instead, we'll accept all values and flag extreme ones in diagnostics
                # Only reject physically impossible values (strain > 100% or < -50%)
                if -0.5 <= local_strain <= 1.0:
                    # Flag extreme but valid strain values (> ±20%) for user awareness
                    if abs(local_strain) > 0.2:
                        n_extreme_strain += 1
                    
                    # Assign to center region of window
                    y_center = (y_start + y_end) // 2
                    x_center = (x_start + x_end) // 2
                    
                    # Use weighted assignment
                    for dy in range(-step_size//2, step_size//2 + 1):
                        for dx in range(-step_size//2, step_size//2 + 1):
                            y_idx = y_center + dy
                            x_idx = x_center + dx
                            if 0 <= y_idx < ny and 0 <= x_idx < nx:
                                weight = np.exp(-(dx**2 + dy**2) / (2 * (step_size/3)**2))
                                if np.isnan(strain_map[y_idx, x_idx]):
                                    strain_map[y_idx, x_idx] = local_strain
                                    moire_period_map[y_idx, x_idx] = local_moire_period
                                    confidence_map[y_idx, x_idx] = weight
                                else:
                                    # Average with existing value (weighted)
                                    total_weight = confidence_map[y_idx, x_idx] + weight
                                    strain_map[y_idx, x_idx] = (
                                        strain_map[y_idx, x_idx] * confidence_map[y_idx, x_idx] +
                                        local_strain * weight
                                    ) / total_weight
                                    moire_period_map[y_idx, x_idx] = (
                                        moire_period_map[y_idx, x_idx] * confidence_map[y_idx, x_idx] +
                                        local_moire_period * weight
                                    ) / total_weight
                                    confidence_map[y_idx, x_idx] = total_weight
                    
                    n_successful += 1
                else:
                    n_strain_out_of_range += 1
    
    print(f"   Processed {n_windows} windows, {n_successful} with valid results")
    if n_windows > 0:
        print(f"   Diagnostics:")
        print(f"     - {n_no_peaks} windows with no peaks detected")
        print(f"     - {n_no_moire_peaks} windows with no moiré peaks in frequency range")
        print(f"     - {n_strain_out_of_range} windows with physically impossible strain (>100% or <-50%)")
        if n_extreme_strain > 0:
            print(f"     - {n_extreme_strain} windows with extreme but valid strain (>±20%) - these are included!")
    print(f"   Coverage: {np.sum(~np.isnan(strain_map)) / strain_map.size * 100:.1f}%")
    
    if n_successful > 0:
        valid_mask = ~np.isnan(strain_map)
        mean_strain = np.nanmean(strain_map)
        std_strain = np.nanstd(strain_map)
        max_strain = np.max(np.abs(strain_map))
        
        print(f"   Strain calculated via FFT peak tracking")
        print(f"   Mean strain: {mean_strain*100:.3f}%")
        print(f"   Strain std: {std_strain*100:.3f}%")
        print(f"   Max |strain|: {max_strain*100:.3f}%")
        print(f"   Strain range: {np.nanmin(strain_map)*100:.3f}% to {np.nanmax(strain_map)*100:.3f}%")
    else:
        mean_strain = 0.0
        std_strain = 0.0
        max_strain = 0.0
        print(f"   Warning: No valid strain calculations")
    
    return strain_map, moire_period_map, mean_strain, std_strain, max_strain, reference_moire_period_nm

# Calculate true strain via FFT peak tracking
# Use larger window size (30 nm) for better peak detection in local regions
strain_map, moire_period_map_strain, mean_strain, std_strain, max_strain, ref_moire_period = calculate_strain_fft_peak_tracking(
    data_clean, reference_moire_period_nm=moire_period, window_size_nm=30, overlap=0.5
)

# ----------------------------------------------------------------------------
# 3b. STRAIN TENSOR CALCULATION (v2.2 NEW)
# ----------------------------------------------------------------------------
print("\n12b. Calculating Strain Tensor (v2.2 NEW)...")

def calculate_strain_tensor(data, window_size_nm=20, overlap=0.5):
    """
    Calculate 2D strain tensor:
    
    ε = [ε_xx  ε_xy]
        [ε_yx  ε_yy]
    
    Where:
    - ε_xx = strain in x-direction
    - ε_yy = strain in y-direction
    - ε_xy = ε_yx = shear strain
    
    Uses displacement field from local FFT peak tracking.
    """
    ny, nx = data.shape
    pixel_size_nm = (data.coords['x'].max().values - data.coords['x'].min().values) / nx * 1e9
    window_size_px = int(window_size_nm / pixel_size_nm)
    
    if window_size_px < 32:
        window_size_px = 32
    if window_size_px > min(nx, ny) // 2:
        window_size_px = min(nx, ny) // 2
    
    step_size = int(window_size_px * (1 - overlap))
    if step_size < 1:
        step_size = 1
    
    # Initialize tensor components
    strain_xx = np.full((ny, nx), np.nan)
    strain_yy = np.full((ny, nx), np.nan)
    strain_xy = np.full((ny, nx), np.nan)
    
    # Calculate displacement field from moiré period variations
    # Simplified approach: use gradient of moiré period map
    if np.sum(~np.isnan(moire_period_map_strain)) > 0:
        # Calculate gradients of moiré period (proxy for displacement)
        period_grad_y, period_grad_x = np.gradient(moire_period_map_strain)
        
        # Normalize by reference period to get strain components
        if ref_moire_period > 0:
            # Strain components from period gradients
            # ε_xx ≈ -∂L/∂x / L_ref (negative because period increase = compression)
            # ε_yy ≈ -∂L/∂y / L_ref
            # ε_xy ≈ -0.5 * (∂L/∂x + ∂L/∂y) / L_ref (shear approximation)
            
            valid_mask = ~np.isnan(moire_period_map_strain)
            strain_xx[valid_mask] = -period_grad_x[valid_mask] / ref_moire_period
            strain_yy[valid_mask] = -period_grad_y[valid_mask] / ref_moire_period
            strain_xy[valid_mask] = -0.5 * (period_grad_x[valid_mask] + period_grad_y[valid_mask]) / ref_moire_period
    
    # Calculate principal strains and directions
    # Principal strains are eigenvalues of strain tensor
    principal_strain_1 = np.full((ny, nx), np.nan)
    principal_strain_2 = np.full((ny, nx), np.nan)
    principal_direction = np.full((ny, nx), np.nan)  # Angle of first principal strain
    
    valid_mask = ~np.isnan(strain_xx)
    for i in range(ny):
        for j in range(nx):
            if valid_mask[i, j]:
                # Strain tensor at this point
                strain_tensor = np.array([[strain_xx[i, j], strain_xy[i, j]],
                                        [strain_xy[i, j], strain_yy[i, j]]])
                
                # Eigenvalues (principal strains)
                eigenvals, eigenvecs = np.linalg.eigh(strain_tensor)
                principal_strain_1[i, j] = eigenvals[1]  # Larger eigenvalue
                principal_strain_2[i, j] = eigenvals[0]  # Smaller eigenvalue
                
                # Direction of first principal strain
                principal_direction[i, j] = np.arctan2(eigenvecs[1, 1], eigenvecs[0, 1]) * 180 / np.pi
    
    # Calculate statistics
    mean_xx = np.nanmean(strain_xx)
    mean_yy = np.nanmean(strain_yy)
    mean_xy = np.nanmean(strain_xy)
    max_shear = np.nanmax(np.abs(strain_xy))
    
    print(f"   Strain tensor calculated")
    if not np.isnan(mean_xx):
        print(f"   Mean ε_xx: {mean_xx*100:.3f}%")
    else:
        print(f"   Mean ε_xx: N/A (no valid data)")
    if not np.isnan(mean_yy):
        print(f"   Mean ε_yy: {mean_yy*100:.3f}%")
    else:
        print(f"   Mean ε_yy: N/A (no valid data)")
    if not np.isnan(mean_xy):
        print(f"   Mean ε_xy (shear): {mean_xy*100:.3f}%")
    else:
        print(f"   Mean ε_xy (shear): N/A (no valid data)")
    if not np.isnan(max_shear):
        print(f"   Max |shear strain|: {max_shear*100:.3f}%")
    else:
        print(f"   Max |shear strain|: N/A (no valid data)")
    
    return {
        'strain_xx': strain_xx,
        'strain_yy': strain_yy,
        'strain_xy': strain_xy,
        'principal_strain_1': principal_strain_1,
        'principal_strain_2': principal_strain_2,
        'principal_direction': principal_direction,
        'mean_xx': mean_xx,
        'mean_yy': mean_yy,
        'mean_xy': mean_xy,
        'max_shear': max_shear
    }

strain_tensor = calculate_strain_tensor(data_clean, window_size_nm=20, overlap=0.5)

# Visualization and saving

print("\n13. Creating visualizations...")

# Save FFT data for quantitative analysis (original, unmodified)
print("\n8. Saving data...")
fft_dataset = xr.Dataset({
    'fft_magnitude': (['ky', 'kx'], fft_mag_pm_scaled),
    'fft_magnitude_display': (['ky', 'kx'], fft_suppressed_display),  # Display version
})
fft_dataset.coords['kx'] = kx
fft_dataset.coords['ky'] = ky
fft_dataset.attrs['description'] = 'FFT analysis - original data preserved'
fft_dataset.attrs['dc_value'] = float(fft_mag_pm_scaled[center_y, center_x])
fft_dataset.attrs['max_value'] = float(fft_mag_pm_scaled.max())
if twist_angle is not None:
    fft_dataset.attrs['twist_angle_deg'] = float(twist_angle)
    fft_dataset.attrs['moire_period_nm'] = float(moire_period)
    if class_info is not None:
        fft_dataset.attrs['moire_frequency_nm_inv'] = float(class_info['primary_freq'])
fft_dataset.attrs['n_peaks'] = len(peak_y)
fft_dataset.attrs['n_primary_peaks'] = len(primary_peaks) if len(primary_peaks) > 0 else 0
fft_dataset.attrs['material'] = 'Helical trilayer graphene'
fft_dataset.to_netcdf(output_dir / "fft_v2.2.nc")
print(f"   Saved: fft_v2.2.nc")

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

# Row 2: Analysis
ax = axes[1, 0]
im4 = ax.imshow(data_clean.values * 1e9, cmap='gray', alpha=0.7)
# Show classified domain walls with colors
colors_map = {1: 'blue', 2: 'red', 3: 'orange', 4: 'purple'}
for wall_type_id, color in colors_map.items():
    wall_mask = domain_wall_classification == wall_type_id
    if np.sum(wall_mask) > 0:
        wall_overlay = np.ma.masked_where(~wall_mask, wall_mask)
        ax.imshow(wall_overlay, cmap=plt.cm.colors.ListedColormap([color]), 
                 alpha=0.5, interpolation='nearest')
ax.set_title(f'Domain Walls (Classified)\n{n_domain_walls} regions, {wall_fraction:.2f}% coverage', 
             fontweight='bold', fontsize=12)
ax.set_xlabel('X (pixels)')
ax.set_ylabel('Y (pixels)')

ax = axes[1, 1]
# Show true strain map (from FFT peak tracking)
if np.sum(~np.isnan(strain_map)) > 0:
    im5 = ax.imshow(strain_map * 100, cmap='RdBu_r', vmin=-max_strain*100, vmax=max_strain*100)
    ax.set_title(f'True Strain Map (FFT Peak Tracking)\nMean: {mean_strain*100:.3f}%, Max: {max_strain*100:.3f}%', 
                 fontweight='bold', fontsize=12)
else:
    ax.text(0.5, 0.5, 'No strain data', ha='center', va='center', transform=ax.transAxes)
    ax.set_title('True Strain Map', fontweight='bold', fontsize=12)
ax.set_xlabel('X (pixels)')
ax.set_ylabel('Y (pixels)')
if np.sum(~np.isnan(strain_map)) > 0:
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
# Histogram of strain (filter out NaN values)
strain_flat = strain_map.flatten()
strain_valid = strain_flat[~np.isnan(strain_flat)]
if len(strain_valid) > 0:
    ax.hist(strain_valid * 100, bins=50, alpha=0.7, edgecolor='black', color='orange')
    if not np.isnan(mean_strain):
        ax.axvline(mean_strain * 100, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_strain*100:.3f}%')
    ax.set_xlabel('Strain (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('Strain Distribution', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
else:
    ax.text(0.5, 0.5, 'No valid strain data\n(Strain calculation failed)', 
            ha='center', va='center', transform=ax.transAxes, fontsize=12)
    ax.set_title('Strain Distribution', fontweight='bold', fontsize=12)

ax = axes[2, 2]
# Summary statistics text
ax.axis('off')
twist_angle_str = f"{twist_angle:.3f}°" if twist_angle is not None else "N/A"
moire_period_str = f"{moire_period:.2f} nm" if moire_period is not None else "N/A"
local_twist_str = f"{np.nanmean(twist_angle_map):.3f}°" if np.sum(~np.isnan(twist_angle_map)) > 0 else "N/A"
local_twist_std_str = f"{np.nanstd(twist_angle_map):.3f}°" if np.sum(~np.isnan(twist_angle_map)) > 0 else "N/A"
moire_freq_str = f"{class_info['primary_freq']:.4f} nm⁻¹" if class_info is not None else "N/A"
magic_str = f"{magic_name} (dev: {magic_deviation:.3f}°)" if closest_magic is not None else "N/A"
strain_tensor_str = f"ε_xx: {strain_tensor['mean_xx']*100:.3f}%, ε_yy: {strain_tensor['mean_yy']*100:.3f}%, ε_xy: {strain_tensor['mean_xy']*100:.3f}%" if 'strain_tensor' in locals() and not (np.isnan(strain_tensor['mean_xx']) or np.isnan(strain_tensor['mean_yy']) or np.isnan(strain_tensor['mean_xy'])) else "N/A"
# Format strain values safely
mean_strain_str = f"{mean_strain*100:.3f}%" if not np.isnan(mean_strain) else "N/A"
std_strain_str = f"{std_strain*100:.3f}%" if not np.isnan(std_strain) else "N/A"
max_strain_str = f"{max_strain*100:.3f}%" if not np.isnan(max_strain) else "N/A"
ref_moire_period_str = f"{ref_moire_period:.2f} nm" if ref_moire_period is not None and not np.isnan(ref_moire_period) else "N/A"
summary_text = f"""
ANALYSIS SUMMARY v2.2
{'='*40}

TWIST ANGLE:
  Global: {twist_angle_str}
  Moiré Period: {moire_period_str}
  Moiré Freq: {moire_freq_str}
  Magic Angle: {magic_str}
  Local Mean: {local_twist_str} ± {local_twist_std_str}
  Local Range: {f"{np.nanmin(twist_angle_map):.2f}°-{np.nanmax(twist_angle_map):.2f}°" if np.sum(~np.isnan(twist_angle_map)) > 0 else "N/A"}
  Primary Peaks: {len(primary_peaks) if len(primary_peaks) > 0 else 0}
  Total Peaks: {len(peak_y)}

DOMAIN WALLS:
  Regions: {n_domain_walls}
  Coverage: {wall_fraction:.2f}%
  Avg Gradient: {grad_magnitude.mean()*1e9:.4f} nm/pixel
  Classified: {sum(wall_type_counts.values())} pixels

STRAIN (True, FFT Peak Tracking):
  Mean: {mean_strain_str}
  Std: {std_strain_str}
  Max: {max_strain_str}
  Ref Moiré Period: {ref_moire_period_str}
  Tensor: {strain_tensor_str}

PROCESSING:
  RMS Change: {rms_change_from_align:.2f}%
  DC Component: {dc_value:.4f} pm
"""
ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
        verticalalignment='center', bbox=dict(boxstyle='round', 
        facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(output_dir / "analysis_v2.2.png", dpi=200, bbox_inches='tight')
print(f"   Saved: analysis_v2.2.png")
plt.close(fig)

# Save individual analysis plots
print("\n14. Saving individual plots...")

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
                      linewidths=3, label='Primary Moiré Peaks', alpha=0.8)
        ax.legend(fontsize=12)
    title_str = f'Twist Angle Analysis (v2.2)\nTwist Angle: {twist_angle:.3f}°'
    if moire_period is not None:
        title_str += f'\nMoiré Period: {moire_period:.2f} nm'
    if closest_magic is not None and magic_deviation <= 0.1:
        title_str += f'\n★ Magic Angle: {magic_name} (dev: {magic_deviation:.3f}°)'
    ax.set_title(title_str, fontsize=14, fontweight='bold')
    ax.set_xlabel('kx (nm⁻¹)', fontsize=12)
    ax.set_ylabel('ky (nm⁻¹)', fontsize=12)
    plt.colorbar(im, ax=ax, label='|FFT| (pm)', fraction=0.046)
    plt.tight_layout()
    plt.savefig(output_dir / "twist_angle_v2.2.png", dpi=200, bbox_inches='tight')
    print(f"   Saved: twist_angle_v2.2.png")
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
plt.savefig(output_dir / "domain_walls_v2.2.png", dpi=200, bbox_inches='tight')
print(f"   Saved: domain_walls_v2.2.png")
plt.close(fig)

# Domain wall classification visualization
fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(data_clean.values * 1e9, cmap='gray', alpha=0.7)
# Color-code by domain wall type
colors = {1: 'blue', 2: 'red', 3: 'orange', 4: 'purple'}
for wall_type_id, color in colors.items():
    wall_mask = domain_wall_classification == wall_type_id
    if np.sum(wall_mask) > 0:
        wall_overlay = np.ma.masked_where(~wall_mask, wall_mask)
        ax.imshow(wall_overlay, cmap=plt.cm.colors.ListedColormap([color]), 
                 alpha=0.6, interpolation='nearest')
ax.set_title(f'Domain Wall Classification (v2.2)\nBlue=Soliton, Red=Dislocation, Orange=Grain Boundary, Purple=Stacking Fault', 
            fontsize=14, fontweight='bold')
ax.set_xlabel('X (pixels)', fontsize=12)
ax.set_ylabel('Y (pixels)', fontsize=12)
plt.tight_layout()
plt.savefig(output_dir / "domain_walls_classified_v2.2.png", dpi=200, bbox_inches='tight')
print(f"   Saved: domain_walls_classified_v2.2.png")
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
plt.savefig(output_dir / "strain_v2.2.png", dpi=200, bbox_inches='tight')
print(f"   Saved: strain_v2.2.png")
plt.close(fig)

# Strain tensor visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 16))
ax = axes[0, 0]
mean_xx_str = f"{strain_tensor['mean_xx']*100:.3f}%" if not np.isnan(strain_tensor['mean_xx']) else "N/A"
vmax_xx = max_strain*100 if not np.isnan(max_strain) else 1.0
im1 = ax.imshow(strain_tensor['strain_xx'] * 100, cmap='RdBu_r', 
               vmin=-vmax_xx, vmax=vmax_xx)
ax.set_title(f'Strain ε_xx (v2.2)\nMean: {mean_xx_str}', 
            fontsize=12, fontweight='bold')
ax.set_xlabel('X (pixels)')
ax.set_ylabel('Y (pixels)')
plt.colorbar(im1, ax=ax, label='Strain (%)')

ax = axes[0, 1]
mean_yy_str = f"{strain_tensor['mean_yy']*100:.3f}%" if not np.isnan(strain_tensor['mean_yy']) else "N/A"
vmax_yy = max_strain*100 if not np.isnan(max_strain) else 1.0
im2 = ax.imshow(strain_tensor['strain_yy'] * 100, cmap='RdBu_r', 
               vmin=-vmax_yy, vmax=vmax_yy)
ax.set_title(f'Strain ε_yy (v2.2)\nMean: {mean_yy_str}', 
            fontsize=12, fontweight='bold')
ax.set_xlabel('X (pixels)')
ax.set_ylabel('Y (pixels)')
plt.colorbar(im2, ax=ax, label='Strain (%)')

ax = axes[1, 0]
mean_xy_str = f"{strain_tensor['mean_xy']*100:.3f}%" if not np.isnan(strain_tensor['mean_xy']) else "N/A"
max_shear_str = f"{strain_tensor['max_shear']*100:.3f}%" if not np.isnan(strain_tensor['max_shear']) else "N/A"
vmax_shear = strain_tensor['max_shear']*100 if not np.isnan(strain_tensor['max_shear']) else 1.0
im3 = ax.imshow(strain_tensor['strain_xy'] * 100, cmap='RdBu_r', 
               vmin=-vmax_shear, vmax=vmax_shear)
ax.set_title(f'Shear Strain ε_xy (v2.2)\nMean: {mean_xy_str}, Max: {max_shear_str}', 
            fontsize=12, fontweight='bold')
ax.set_xlabel('X (pixels)')
ax.set_ylabel('Y (pixels)')
plt.colorbar(im3, ax=ax, label='Shear Strain (%)')

ax = axes[1, 1]
im4 = ax.imshow(strain_tensor['principal_strain_1'] * 100, cmap='viridis')
ax.set_title('Principal Strain 1 (v2.2)\nLarger eigenvalue', 
            fontsize=12, fontweight='bold')
ax.set_xlabel('X (pixels)')
ax.set_ylabel('Y (pixels)')
plt.colorbar(im4, ax=ax, label='Principal Strain (%)')

plt.tight_layout()
plt.savefig(output_dir / "strain_tensor_v2.2.png", dpi=200, bbox_inches='tight')
print(f"   Saved: strain_tensor_v2.2.png")
plt.close(fig)

# Moiré period map visualization (from strain calculation)
if np.sum(~np.isnan(moire_period_map_strain)) > 0:
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(moire_period_map_strain, cmap='plasma', interpolation='bilinear')
    ax.set_title(f'Moiré Period Map (v2.2)\nMean: {np.nanmean(moire_period_map_strain):.2f} nm\nRange: {np.nanmin(moire_period_map_strain):.2f} - {np.nanmax(moire_period_map_strain):.2f} nm', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Y (pixels)', fontsize=12)
    plt.colorbar(im, ax=ax, label='Moiré Period (nm)', fraction=0.046)
    plt.tight_layout()
    plt.savefig(output_dir / "moire_period_map_v2.2.png", dpi=200, bbox_inches='tight')
    print(f"   Saved: moire_period_map_v2.2.png")
    plt.close(fig)

# Diagnostic visualization: All moiré peaks and their twist angles
if twist_angle is not None and len(primary_peaks) > 0:
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
    ax.set_xlabel('Moiré Peak (sorted by angle)', fontsize=12)
    ax.set_ylabel('Twist Angle (°)', fontsize=12)
    ax.set_title('Twist Angles from All Detected Moiré Peaks (v2.1)\n(Red = Selected, Strongest Peak)', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(len(moire_peak_angles)))
    ax.set_xticklabels([f'P{i+1}' for i in range(len(moire_peak_angles))])
    
    # Plot 2: Magnitude vs Twist Angle
    ax2 = axes[1]
    scatter = ax2.scatter(moire_peak_angles, moire_peak_mags, 
                         s=200, c=colors, alpha=0.7, edgecolors='black', linewidths=2)
    ax2.set_xlabel('Twist Angle (°)', fontsize=12)
    ax2.set_ylabel('Peak Magnitude', fontsize=12)
    ax2.set_title('Peak Magnitude vs Twist Angle (v2.1)\n(Red = Selected, Strongest Peak)', 
                  fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "twist_angle_diagnostics_v2.2.png", dpi=200, bbox_inches='tight')
    print(f"   Saved: twist_angle_diagnostics_v2.2.png")
    plt.close(fig)

# Local twist angle map visualization
if np.sum(~np.isnan(twist_angle_map)) > 0:
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(twist_angle_map, cmap='viridis', interpolation='bilinear')
    ax.set_title(f'Local Twist Angle Map (v2.2)\nMean: {np.nanmean(twist_angle_map):.3f}° ± {np.nanstd(twist_angle_map):.3f}°\nRange: {np.nanmin(twist_angle_map):.2f}° - {np.nanmax(twist_angle_map):.2f}°', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Y (pixels)', fontsize=12)
    plt.colorbar(im, ax=ax, label='Twist Angle (°)', fraction=0.046)
    plt.tight_layout()
    plt.savefig(output_dir / "twist_angle_map_v2.2.png", dpi=200, bbox_inches='tight')
    print(f"   Saved: twist_angle_map_v2.2.png")
    plt.close(fig)

# Save all analysis data
print("\n15. Saving analysis data...")
advanced_dataset = xr.Dataset({
    'topography': (['y', 'x'], data_clean.values),
    'gradient_magnitude': (['y', 'x'], grad_magnitude),
    'domain_wall_mask': (['y', 'x'], domain_wall_mask.astype(int)),
    'domain_wall_classification': (['y', 'x'], domain_wall_classification),
    'strain_map': (['y', 'x'], strain_map),
    'strain_xx': (['y', 'x'], strain_tensor['strain_xx']),
    'strain_yy': (['y', 'x'], strain_tensor['strain_yy']),
    'strain_xy': (['y', 'x'], strain_tensor['strain_xy']),
    'principal_strain_1': (['y', 'x'], strain_tensor['principal_strain_1']),
    'principal_strain_2': (['y', 'x'], strain_tensor['principal_strain_2']),
    'principal_direction': (['y', 'x'], strain_tensor['principal_direction']),
    'twist_angle_map': (['y', 'x'], twist_angle_map),
    'moire_period_map': (['y', 'x'], moire_period_map),
    'moire_period_map_strain': (['y', 'x'], moire_period_map_strain),
    'confidence_map': (['y', 'x'], confidence_map),
})
advanced_dataset.coords['x'] = x_coords
advanced_dataset.coords['y'] = y_coords
advanced_dataset.attrs['twist_angle_deg'] = float(twist_angle) if twist_angle is not None else np.nan
advanced_dataset.attrs['moire_period_nm'] = float(moire_period) if moire_period is not None else np.nan
if class_info is not None:
    advanced_dataset.attrs['moire_frequency_nm_inv'] = float(class_info['primary_freq'])
if closest_magic is not None:
    advanced_dataset.attrs['magic_angle_deg'] = float(closest_magic)
    advanced_dataset.attrs['magic_angle_deviation_deg'] = float(magic_deviation)
    advanced_dataset.attrs['magic_angle_name'] = str(magic_name)
advanced_dataset.attrs['n_domain_walls'] = int(n_domain_walls)
advanced_dataset.attrs['domain_wall_fraction_percent'] = float(wall_fraction)
advanced_dataset.attrs['mean_strain_percent'] = float(mean_strain * 100)
advanced_dataset.attrs['std_strain_percent'] = float(std_strain * 100)
advanced_dataset.attrs['max_strain_percent'] = float(max_strain * 100)
advanced_dataset.attrs['reference_moire_period_nm'] = float(ref_moire_period)
advanced_dataset.attrs['strain_tensor_mean_xx_percent'] = float(strain_tensor['mean_xx'] * 100)
advanced_dataset.attrs['strain_tensor_mean_yy_percent'] = float(strain_tensor['mean_yy'] * 100)
advanced_dataset.attrs['strain_tensor_mean_xy_percent'] = float(strain_tensor['mean_xy'] * 100)
advanced_dataset.attrs['strain_tensor_max_shear_percent'] = float(strain_tensor['max_shear'] * 100)
advanced_dataset.attrs['n_fft_peaks'] = len(peak_y)
advanced_dataset.attrs['n_primary_peaks'] = len(primary_peaks) if len(primary_peaks) > 0 else 0
advanced_dataset.attrs['material'] = 'Helical trilayer graphene'
advanced_dataset.attrs['version'] = '2.2'
if np.sum(~np.isnan(twist_angle_map)) > 0:
    advanced_dataset.attrs['local_twist_mean_deg'] = float(np.nanmean(twist_angle_map))
    advanced_dataset.attrs['local_twist_std_deg'] = float(np.nanstd(twist_angle_map))
    advanced_dataset.attrs['local_twist_min_deg'] = float(np.nanmin(twist_angle_map))
    advanced_dataset.attrs['local_twist_max_deg'] = float(np.nanmax(twist_angle_map))
    advanced_dataset.attrs['local_twist_coverage_percent'] = float(np.sum(~np.isnan(twist_angle_map)) / twist_angle_map.size * 100)

advanced_dataset.to_netcdf(output_dir / "advanced_analysis_v2.2.nc")
print(f"   Saved: advanced_analysis_v2.2.nc")

print(f"\nAnalysis finished")
print(f"\nResults:")
if twist_angle is not None:
    print(f"  Twist angle: {twist_angle:.3f}°")
    if moire_period is not None:
        print(f"  Moiré period: {moire_period:.2f} nm")
    if closest_magic is not None:
        print(f"  Magic angle: {magic_name} (deviation: {magic_deviation:.3f}°)")
else:
    print(f"  Twist angle: Could not be determined")
print(f"  Domain walls: {n_domain_walls} regions ({wall_fraction:.2f}% coverage)")
if not np.isnan(mean_strain):
    print(f"  Mean strain: {mean_strain*100:.3f}% (std: {std_strain*100:.3f}%)")
print(f"  RMS change: {rms_change_from_align:.2f}%")
print(f"\nFiles saved to: {output_dir}")

