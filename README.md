# McLaren 2023 F1 Front Wing FSI Analysis

**Comparative aerodynamic analysis of rigid vs. deformed McLaren 2023 F1 front wing under high-speed conditions**

![29d5454f-4901-4fce-af53-e09a96cfac65](https://github.com/user-attachments/assets/a81f9e8e-b27e-42c9-a3e3-6e118005ea8e)

[![OpenFOAM](https://img.shields.io/badge/OpenFOAM-11-blue)](https://openfoam.org/)
[![CalculiX](https://img.shields.io/badge/CalculiX-2.20-green)](http://www.calculix.de/)
[![Python](https://img.shields.io/badge/Python-3.8+-red)](https://python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

![1d79c84a-0738-45a3-ad3b-6d05f5fbcbcd](https://github.com/user-attachments/assets/86af9ca0-b3c1-4cf7-a151-4107a83289d7)
![09f08b5a-27cd-4d27-9a8a-51fb6479a40e](https://github.com/user-attachments/assets/43734ddc-c5bb-43d7-a988-724474dfd303)
![f33767bc-c6be-42e8-8cf7-165dc8ad98bb](https://github.com/user-attachments/assets/5e3dce29-2f11-4fd3-8528-173ceeb2971a)

## Project Objective

This project demonstrates a critical engineering challenge in Formula 1: **How does McLaren's 2023 front wing design perform under structural deformation at 350 km/h?**

We provide a complete automated pipeline that:
1. **Analyzes wing deformation** under aerodynamic loads (CalculiX FEA)
2. **Compares aerodynamic performance** between rigid and deformed wings (OpenFOAM CFD)
3. **Quantifies performance loss** due to structural flexibility
4. **Visualizes flow differences** through advanced streamline analysis

## Key Results

| Metric | Rigid Wing | Deformed Wing | Performance Loss |
|--------|------------|---------------|------------------|
| **Downforce** | 2,450 N | 2,263 N | **-7.6%** |
| **Drag** | 485 N | 523 N | **+7.8%** |
| **L/D Ratio** | 5.05 | 4.33 | **-14.3%** |
| **Max Deflection** | 0 mm | 18.5 mm | Wing tip displacement |

## Visuals & Results

### Streamline Analysis
<img width="1167" height="798" alt="Screenshot 2025-08-18 225742" src="https://github.com/user-attachments/assets/c829cd1f-a368-443f-8ca1-5ac684b88a8e" />


### Structural Deformation

<img width="1407" height="791" alt="Screenshot 2025-08-18 224643" src="https://github.com/user-attachments/assets/b4550ad7-0201-4a0c-9433-f7ca2d1dc887" />

## Technology Stack

- **CFD Solver**: OpenFOAM 11 (rhoCentralFoam for compressible flow)
- **FEA Solver**: CalculiX 2.20 (static structural analysis)
- **Meshing**: snappyHexMesh (adaptive refinement)
- **Automation**: Python 3.8+ (complete pipeline)
- **Visualization**: ParaView, matplotlib, custom VTK processing, Blender

## Project Structure

```
McLaren-2023-Front-Wing-FSI/
├── geometry/
│   ├── mclaren_2023_original.stl    # Rigid McLaren wing geometry
│   └── mclaren_2023_deformed.stl    # Wing deformed at 350 km/h
├── fea-analysis/
│   ├── calculix/                    # CalculiX input files
│   └── results/                     # Stress, deformation results
├── cfd-analysis/
│   ├── rigid-wing/                  # Baseline McLaren CFD case
│   ├── deformed-wing/               # Deformed McLaren wing CFD case
│   └── comparison/                  # Comparative analysis
├── scripts/
│   ├── run_analysis.py              # Master automation script
│   ├── mesh_generation.py           # Automated meshing
│   ├── post_process.py              # Results extraction
│   └── visualization.py             # Plot generation
└── results/
    ├── animations/                  # Flow animations
    ├── plots/                       # Publication-quality plots
    └── reports/                     # Technical documentation
```

## Quick Start

### Prerequisites
```bash
# OpenFOAM 11
source /opt/openfoam11/etc/bashrc

# CalculiX
sudo apt install calculix-ccx calculix-cgx

# Python dependencies
pip install numpy matplotlib vtk pandas pyyaml
```

### Run Complete Analysis
```bash
# Clone repository
git clone https://github.com/yourusername/McLaren-2023-Front-Wing-FSI.git
cd McLaren-2023-Front-Wing-FSI

# Execute full pipeline
python scripts/run_analysis.py

# Generate comparison plots
python scripts/visualization.py --comparison
```

### Individual Analysis Steps
```bash
# FEA analysis only
python scripts/fea_analysis.py --input geometry/mclaren_2023_original.stl

# CFD analysis only
python scripts/cfd_analysis.py --case rigid-wing

# Post-processing only
python scripts/post_process.py --compare rigid-wing deformed-wing
```

## Methodology

### 1. Structural Analysis (FEA)
- **Material**: Carbon fiber composite (E = 70 GPa, ν = 0.3)
- **Loading**: Realistic aerodynamic pressure distribution
- **Boundary Conditions**: Wing root mounting constraints
- **Output**: 3D deformation field, stress distribution

### 2. CFD Analysis
- **Flow Conditions**: 350 km/h (97 m/s) compressible flow
- **Turbulence Model**: k-ω SST for automotive aerodynamics
- **Mesh**: 420,000 cells with boundary layer refinement
- **Solver**: rhoCentralFoam (density-based compressible)

### 3. Comparison Analysis
- **Force Integration**: Automated force coefficient extraction
- **Flow Visualization**: 100+ streamlines comparison
- **Performance Metrics**: Downforce, drag, efficiency analysis

## Advanced Features

### Automated Streamline Generation
Based on proven OpenFOAM methodology:
- **100+ streamlines** for complete flow visualization
- **Time-accurate tracking** through full simulation
- **VTK output** for professional visualization
- **Automated seed point placement** for optimal coverage

### Mesh Independence Study
- **Convergence verification** across 3 mesh densities
- **Y+ validation** for boundary layer accuracy
