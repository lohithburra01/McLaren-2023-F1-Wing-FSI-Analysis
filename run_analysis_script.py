#!/usr/bin/env python3
"""
McLaren 2023 F1 Front Wing FSI Analysis Pipeline
Complete automation script for rigid vs deformed wing comparison

Author: Your Name
Date: August 2025
"""

import os
import sys
import argparse
import yaml
import subprocess
import logging
from pathlib import Path
import time

# Add scripts directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mesh_generation import generate_mesh
from fea_analysis import run_calculix
from cfd_analysis import run_openfoam
from post_process import extract_results
from visualization import create_comparison_plots

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class McLarenWingAnalysis:
    """Main analysis pipeline for McLaren 2023 front wing FSI study"""
    
    def __init__(self, config_file='config/default.yaml'):
        """Initialize analysis with configuration"""
        self.config = self.load_config(config_file)
        self.project_root = Path(__file__).parent.parent
        self.setup_directories()
        
    def load_config(self, config_file):
        """Load configuration from YAML file"""
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_file}")
            return config
        except FileNotFoundError:
            logger.error(f"Config file {config_file} not found")
            sys.exit(1)
            
    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            'fea-analysis/results',
            'cfd-analysis/rigid-wing/postProcessing',
            'cfd-analysis/deformed-wing/postProcessing',
            'cfd-analysis/comparison',
            'results/plots',
            'results/animations',
            'results/reports'
        ]
        
        for directory in directories:
            Path(self.project_root / directory).mkdir(parents=True, exist_ok=True)
            
    def run_fea_analysis(self):
        """Execute structural analysis with CalculiX"""
        logger.info("=== Starting FEA Analysis ===")
        
        fea_config = self.config['fea']
        geometry_file = self.project_root / "geometry/mclaren_2023_original.stl"
        
        if not geometry_file.exists():
            logger.error(f"Geometry file not found: {geometry_file}")
            return False
            
        try:
            # Run CalculiX analysis
            deformed_geometry = run_calculix(
                input_geometry=str(geometry_file),
                material_props=fea_config['materials']['carbon_fiber'],
                loads=fea_config['aerodynamic_loads'],
                output_dir=str(self.project_root / "fea-analysis/results")
            )
            
            # Copy deformed geometry to geometry folder
            deformed_path = self.project_root / "geometry/mclaren_2023_deformed.stl"
            subprocess.run(['cp', deformed_geometry, str(deformed_path)], check=True)
            
            logger.info("FEA analysis completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"FEA analysis failed: {e}")
            return False
            
    def run_cfd_analysis(self, case_name):
        """Execute CFD analysis for specified case"""
        logger.info(f"=== Starting CFD Analysis: {case_name} ===")
        
        case_dir = self.project_root / f"cfd-analysis/{case_name}"
        cfd_config = self.config['cfd']
        
        # Select geometry based on case
        if case_name == "rigid-wing":
            geometry_file = "geometry/mclaren_2023_original.stl"
        elif case_name == "deformed-wing":
            geometry_file = "geometry/mclaren_2023_deformed.stl"
        else:
            logger.error(f"Unknown case: {case_name}")
            return False
            
        try:
            # Generate mesh
            logger.info(f"Generating mesh for {case_name}")
            generate_mesh(
                geometry_file=str(self.project_root / geometry_file),
                case_directory=str(case_dir),
                mesh_config=cfd_config['mesh']
            )
            
            # Run OpenFOAM simulation
            logger.info(f"Running OpenFOAM simulation for {case_name}")
            run_openfoam(
                case_directory=str(case_dir),
                solver_config=cfd_config['solver'],
                flow_conditions=cfd_config['flow_conditions']
            )
            
            # Generate streamlines
            logger.info(f"Generating streamlines for {case_name}")
            self.generate_streamlines(case_dir)
            
            logger.info(f"CFD analysis completed: {case_name}")
            return True
            
        except Exception as e:
            logger.error(f"CFD analysis failed for {case_name}: {e}")
            return False
            
    def generate_streamlines(self, case_dir):
        """Generate streamlines using proven methodology from PDF"""
        
        # Update controlDict with streamlines function
        controldict_path = case_dir / "system/controlDict"
        
        streamlines_config = '''
functions
{
    streams100
    {
        type streamlines;
        libs ("libfieldFunctionObjects.so");
        
        seedSampleSet
        {
            type points;
            ordered true;
            points
            (
                (-2.0 -2.0 -0.01) (-2.0 -1.96 -0.01) (-2.0 -1.92 -0.01) (-2.0 -1.88 -0.01) (-2.0 -1.84 -0.01)
                (-2.0 -1.8 -0.01) (-2.0 -1.76 -0.01) (-2.0 -1.72 -0.01) (-2.0 -1.68 -0.01) (-2.0 -1.64 -0.01)
                (-2.0 -1.6 -0.01) (-2.0 -1.56 -0.01) (-2.0 -1.52 -0.01) (-2.0 -1.48 -0.01) (-2.0 -1.44 -0.01)
                (-2.0 -1.4 -0.01) (-2.0 -1.36 -0.01) (-2.0 -1.32 -0.01) (-2.0 -1.28 -0.01) (-2.0 -1.24 -0.01)
                (-2.0 -1.2 -0.01) (-2.0 -1.16 -0.01) (-2.0 -1.12 -0.01) (-2.0 -1.08 -0.01) (-2.0 -1.04 -0.01)
                (-2.0 -1.0 -0.01) (-2.0 -0.96 -0.01) (-2.0 -0.92 -0.01) (-2.0 -0.88 -0.01) (-2.0 -0.84 -0.01)
                (-2.0 -0.8 -0.01) (-2.0 -0.76 -0.01) (-2.0 -0.72 -0.01) (-2.0 -0.68 -0.01) (-2.0 -0.64 -0.01)
                (-2.0 -0.6 -0.01) (-2.0 -0.56 -0.01) (-2.0 -0.52 -0.01) (-2.0 -0.48 -0.01) (-2.0 -0.44 -0.01)
                (-2.0 -0.4 -0.01) (-2.0 -0.36 -0.01) (-2.0 -0.32 -0.01) (-2.0 -0.28 -0.01) (-2.0 -0.24 -0.01)
                (-2.0 -0.2 -0.01) (-2.0 -0.16 -0.01) (-2.0 -0.12 -0.01) (-2.0 -0.08 -0.01) (-2.0 -0.04 -0.01)
                (-2.0 0.0 -0.01) (-2.0 0.04 -0.01) (-2.0 0.08 -0.01) (-2.0 0.12 -0.01) (-2.0 0.16 -0.01)
                (-2.0 0.2 -0.01) (-2.0 0.24 -0.01) (-2.0 0.28 -0.01) (-2.0 0.32 -0.01) (-2.0 0.36 -0.01)
                (-2.0 0.4 -0.01) (-2.0 0.44 -0.01) (-2.0 0.48 -0.01) (-2.0 0.52 -0.01) (-2.0 0.56 -0.01)
                (-2.0 0.6 -0.01) (-2.0 0.64 -0.01) (-2.0 0.68 -0.01) (-2.0 0.72 -0.01) (-2.0 0.76 -0.01)
                (-2.0 0.8 -0.01) (-2.0 0.84 -0.01) (-2.0 0.88 -0.01) (-2.0 0.92 -0.01) (-2.0 0.96 -0.01)
                (-2.0 1.0 -0.01) (-2.0 1.04 -0.01) (-2.0 1.08 -0.01) (-2.0 1.12 -0.01) (-2.0 1.16 -0.01)
                (-2.0 1.2 -0.01) (-2.0 1.24 -0.01) (-2.0 1.28 -0.01) (-2.0 1.32 -0.01) (-2.0 1.36 -0.01)
                (-2.0 1.4 -0.01) (-2.0 1.44 -0.01) (-2.0 1.48 -0.01) (-2.0 1.52 -0.01) (-2.0 1.56 -0.01)
                (-2.0 1.6 -0.01) (-2.0 1.64 -0.01) (-2.0 1.68 -0.01) (-2.0 1.72 -0.01) (-2.0 1.76 -0.01)
                (-2.0 1.8 -0.01) (-2.0 1.84 -0.01) (-2.0 1.88 -0.01) (-2.0 1.92 -0.01) (-2.0 1.96 -0.01)
                (-2.0 2.0 -0.01)
            );
        }
        
        direction forward;
        maxSteps 10000;
        stepSize 0.001;
        lifeTime 1000;
        trackLength 5.0;
        interpolationScheme cellPoint;
        fields (U);
        setFormat vtk;
    }
}
'''
        
        # Append streamlines configuration to controlDict
        with open(controldict_path, 'a') as f:
            f.write(streamlines_config)
            
        # Run post-processing to generate streamlines
        os.chdir(case_dir)
        subprocess.run(['foamPostProcess'], check=True, cwd=case_dir)
        
    def run_comparison_analysis(self):
        """Compare results between rigid and deformed wings"""
        logger.info("=== Starting Comparison Analysis ===")
        
        try:
            # Extract results from both cases
            rigid_results = extract_results("cfd-analysis/rigid-wing")
            deformed_results = extract_results("cfd-analysis/deformed-wing")
            
            # Calculate performance differences
            comparison_data = {
                'rigid': rigid_results,
                'deformed': deformed_results,
                'performance_loss': {
                    'downforce_loss_percent': ((rigid_results['downforce'] - deformed_results['downforce']) / rigid_results['downforce']) * 100,
                    'drag_increase_percent': ((deformed_results['drag'] - rigid_results['drag']) / rigid_results['drag']) * 100,
                    'efficiency_loss_percent': ((rigid_results['efficiency'] - deformed_results['efficiency']) / rigid_results['efficiency']) * 100
                }
            }
            
            # Generate comparison plots
            create_comparison_plots(comparison_data, "results/plots")
            
            # Save comparison data
            import json
            with open("results/comparison_summary.json", 'w') as f:
                json.dump(comparison_data, f, indent=2)
                
            logger.info("Comparison analysis completed")
            return comparison_data
            
        except Exception as e:
            logger.error(f"Comparison analysis failed: {e}")
            return None
            
    def run_full_pipeline(self):
        """Execute complete analysis pipeline"""
        logger.info("=== Starting Complete McLaren 2023 Wing Analysis ===")
        start_time = time.time()
        
        # Step 1: FEA Analysis
        if not self.run_fea_analysis():
            logger.error("Pipeline failed at FEA stage")
            return False
            
        # Step 2: CFD Analysis - Rigid Wing
        if not self.run_cfd_analysis("rigid-wing"):
            logger.error("Pipeline failed at rigid wing CFD stage")
            return False
            
        # Step 3: CFD Analysis - Deformed Wing
        if not self.run_cfd_analysis("deformed-wing"):
            logger.error("Pipeline failed at deformed wing CFD stage")
            return False
            
        # Step 4: Comparison Analysis
        comparison_results = self.run_comparison_analysis()
        if comparison_results is None:
            logger.error("Pipeline failed at comparison stage")
            return False
            
        # Pipeline completed successfully
        end_time = time.time()
        total_time = end_time - start_time
        
        logger.info(f"=== Pipeline Completed Successfully ===")
        logger.info(f"Total execution time: {total_time:.2f} seconds")
        
        # Print key results
        perf_loss = comparison_results['performance_loss']
        logger.info(f"Performance Impact Summary:")
        logger.info(f"  Downforce Loss: {perf_loss['downforce_loss_percent']:.1f}%")
        logger.info(f"  Drag Increase: {perf_loss['drag_increase_percent']:.1f}%")
        logger.info(f"  Efficiency Loss: {perf_loss['efficiency_loss_percent']:.1f}%")
        
        return True

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='McLaren 2023 F1 Front Wing FSI Analysis')
    parser.add_argument('--config', default='config/default.yaml', help='Configuration file path')
    parser.add_argument('--step', choices=['fea', 'cfd-rigid', 'cfd-deformed', 'comparison', 'full'], 
                       default='full', help='Analysis step to run')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Initialize analysis
    analysis = McLarenWingAnalysis(args.config)
    
    # Execute requested step
    if args.step == 'fea':
        success = analysis.run_fea_analysis()
    elif args.step == 'cfd-rigid':
        success = analysis.run_cfd_analysis('rigid-wing')
    elif args.step == 'cfd-deformed':
        success = analysis.run_cfd_analysis('deformed-wing')
    elif args.step == 'comparison':
        success = analysis.run_comparison_analysis() is not None
    elif args.step == 'full':
        success = analysis.run_full_pipeline()
    else:
        logger.error(f"Unknown step: {args.step}")
        success = False
        
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()