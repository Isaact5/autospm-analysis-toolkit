# Automated SPM Analysis Toolkit

Automated analysis toolkit for Scanning Probe Microscopy (SPM) data, specifically designed for helical trilayer graphene characterization. This toolkit provides comprehensive material analysis including twist angle calculation, true strain measurement, strain tensor analysis, domain wall detection and classification, and magic angle detection.

## Features

### Core Analysis Capabilities

- **True Strain Measurement**: FFT peak tracking for accurate strain calculation from moiré period variations
- **2D Strain Tensor**: Full tensor analysis (ε_xx, ε_yy, ε_xy) with principal strain calculation
- **Twist Angle Calculation**: Moiré pattern analysis with expanded frequency detection (0.01-1.0 nm⁻¹)
- **Domain Wall Detection & Classification**: 4-type classification (Solitons, Dislocations, Grain Boundaries, Stacking Faults)
- **Moiré Period Mapping**: Spatial mapping of moiré periods across the sample
- **Local Spatial Analysis**: Twist angle and strain mapping with 82%+ coverage

### Technical Highlights

- **1,600+ lines** of Python code (v2.2)
- **40-50x time savings** compared to manual Gwyddion analysis (15-20 min → 15-30 sec per image)
- **Quantitative validation** framework with RMS change tracking
- **Data preservation** architecture (original data + display versions)
- **Unbiased algorithms** with comprehensive diagnostics

## Installation

### Requirements

```bash
pip install spym nanonispy numpy matplotlib xarray scipy
```

### Dependencies

- Python 3.7+
- spym
- nanonispy
- numpy
- matplotlib
- xarray
- scipy

## Usage

### Basic Usage

```python
# Run analysis (modify input_path in script)
python spym_v2.2.py
```
```bash
Input file selection
Enter path to .sxm file: 
```

### Input

- **Format**: Nanonis .sxm files
- **Channel**: Z (topography) by default
- **Size**: Tested on 608×608 pixel images

### Output

- **NetCDF files**: Quantitative data (processed_data_v2.2.nc, fft_v2.2.nc, advanced_analysis_v2.2.nc)
- **Visualizations**: 12+ PNG images including:
  - Processed topography
  - FFT modulus with peak markers
  - Twist angle analysis
  - Domain wall detection and classification
  - True strain map
  - Strain tensor components
  - Moiré period map
  - Local twist angle map
  - Comprehensive analysis summary

## Version History

### v2.2 (Current)
- True strain calculation via FFT peak tracking
- 2D strain tensor calculation
- Moiré period map
- Magic angle detection
- Domain wall classification

### v2.1
- Expanded moiré frequency range (0.01-1.0 nm⁻¹)
- Peak classification (moiré vs atomic)
- Enhanced validation and diagnostics

### v2.0
- Twist angle calculation
- Domain wall detection
- Strain calculation (simplified)

### v1.2
- Analytical accuracy focus
- Data preservation
- Quantitative validation

### v1.1
- Accurate spatial frequency calculations
- Proper FFT scaling
- Meaningful quantitative analysis
- Orange colormap

### v1.0
- Basic data processing operations & FFT
- Visualizations
- Proof of concept

## Key Algorithms

### True Strain Calculation
Uses FFT peak tracking in sliding windows to measure local moiré periods:
```
ε = (L_local - L_reference) / L_reference
```

### Strain Tensor
Calculates full 2D tensor from moiré period gradients:
```
ε = [ε_xx  ε_xy]
    [ε_yx  ε_yy]
```

### Twist Angle
From moiré period using moiré pattern physics:
```
θ = 2arcsin(a/(2L))
```

## Performance

- **Processing time**: 15-30 sec per image (608×608 pixels)
- **Spatial coverage**: 82%+ for local analysis maps
- **Accuracy**: Validated processing quality (RMS change < 5%)

## Author

Isaac Tsai