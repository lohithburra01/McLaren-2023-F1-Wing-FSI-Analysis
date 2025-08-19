#!/usr/bin/env python3
"""
McLaren 2023 F1 Front Wing Post-Processing
Streamlines generation and results extraction based on proven OpenFOAM methodology

Implements the exact methodology from the provided PDF for reliable streamline generation
"""

import os
import sys
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import json
import logging

logger = logging.getLogger(__name__)

class McLarenWingPostProcessor:
    """Post-processing class for McLaren wing analysis"""
    
    def __init__(self, case_directory):
        self.case_dir = Path(case_directory)
        self.post_dir = self.case_dir / "postProcessing"
        self.results = {}
        
    def setup_streamlines_function(self):
        """
        Setup streamlines function in controlDict
        Based on proven methodology from PDF documentation
        """
        logger.info("Setting up streamlines function in controlDict")
        
        controldict_path = self.case_dir / "system/controlDict"
        
        # Streamlines configuration using SHORT name (critical for path length)
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
        
        # Check if functions section already exists
        with open(controldict_path, 'r') as f:
            content = f.read()
            
        if "functions" in content:
            logger.warning("Functions section already exists in controlDict")
            return
            
        # Remove any existing functions section and add new one
        lines = content.split('\n')
        
        # Find the end of file (before final comment)
        insert_index = len(lines) - 1
        for i, line in enumerate(lines):
            if line.strip().startswith('//') and '***' in line:
                insert_index = i
                break
                
        # Insert streamlines configuration
        streamlines_lines = streamlines_config.strip().split('\n')
        lines[insert_index:insert_index] = streamlines_lines
        
        # Write back to file
        with open(controldict_path, 'w') as f:
            f.write('\n'.join(lines))
            
        logger.info("Streamlines function added to controlDict")
        
    def generate_streamlines(self):
        """
        Generate streamlines using proven OpenFOAM methodology
        Follows exact procedure from PDF documentation
        """
        logger.info("Generating streamlines for McLaren wing")
        
        # Ensure we're in the case directory
        original_dir = os.getcwd()
        os.chdir(self.case_dir)
        
        try:
            # Setup streamlines function
            self.setup_streamlines_function()
            
            # Run foamPostProcess (NO -func flag as per PDF instructions)
            logger.info("Running foamPostProcess...")
            result = subprocess.run(
                ['foamPostProcess'],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Check for success indicators
            if "Seeded 101 particles" in result.stdout:
                logger.info("✅ Successfully seeded 101 streamlines")
            else:
                logger.warning("Streamline seeding output not as expected")
                
            # Verify output directory exists
            streamlines_dir = self.post_dir / "streams100"
            if streamlines_dir.exists():
                logger.info(f"✅ Streamlines data generated in {streamlines_dir}")
                
                # Count VTK files
                vtk_files = list(streamlines_dir.glob("**/tracks.vtk"))
                logger.info(f"Generated {len(vtk_files)} timestep streamline files")
            else:
                raise FileNotFoundError("Streamlines directory not created")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"foamPostProcess failed: {e}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            raise
        finally:
            os.chdir(original_dir)
            
    def extract_force_coefficients(self):
        """Extract force coefficients from simulation"""
        logger.info("Extracting force coefficients")
        
        # Look for forces directory
        forces_dir = self.post_dir / "forces"
        if not forces_dir.exists():
            logger.warning("Forces directory not found - running forces function")
            self.setup_forces_function()
            return
            
        # Read coefficient data
        coeff_file = forces_dir / "0" / "coefficient.dat"
        if coeff_file.exists():
            df = pd.read_csv(coeff_file, delim_whitespace=True, comment='#')
            
            # Extract final converged values
            final_values = df.iloc[-1]
            
            self.results['force_coefficients'] = {
                'Cd': float(final_values.get('Cd', 0)),
                'Cl': float(final_values.get('Cl', 0)),
                'CmPitch': float(final_values.get('CmPitch', 0))
            }
            
            # Calculate dimensional forces (assuming reference area = 0.8 m²)
            rho = 1.225  # kg/m³
            U = 97.22    # m/s (350 km/h)
            A_ref = 0.8  # m²
            q_inf = 0.5 * rho * U**2 * A_ref
            
            self.results['forces'] = {
                'downforce': -self.results['force_coefficients']['Cl'] * q_inf,  # Negative Cl = downforce
                'drag': self.results['force_coefficients']['Cd'] * q_inf,
                'efficiency': abs(self.results['force_coefficients']['Cl'] / self.results['force_coefficients']['Cd'])
            }
            
            logger.info(f"Downforce: {self.results['forces']['downforce']:.1f} N")
            logger.info(f"Drag: {self.results['forces']['drag']:.1f} N")
            logger.info(f"L/D Ratio: {self.results['forces']['efficiency']:.2f}")
            
    def setup_forces_function(self):
        """Setup forces calculation function"""
        logger.info("Setting up forces function")
        
        forces_config = '''
    forces
    {
        type forces;
        libs ("libforces.so");
        
        patches (mclaren_wing);
        rho rhoInf;
        rhoInf 1.225;
        
        CofR (0 0 0);
        pitchAxis (0 1 0);
        
        writeControl timeStep;
        writeInterval 1;
    }
    
    forceCoeffs
    {
        type forceCoeffs;
        libs ("libforces.so");
        
        patches (mclaren_wing);
        rho rhoInf;
        rhoInf 1.225;
        
        CofR (0 0 0);
        liftDir (0 0 1);
        dragDir (1 0 0);
        pitchAxis (0 1 0);
        
        magUInf 97.22;
        lRef 1.8;        // Wing span
        Aref 0.8;        // Wing area
        
        writeControl timeStep;
        writeInterval 1;
    }
'''
        
        # Add to controlDict functions section
        controldict_path = self.case_dir / "system/controlDict"
        with open(controldict_path, 'r') as f:
            content = f.read()
            
        # Insert forces functions before closing brace of functions section
        if "functions" in content:
            content = content.replace(
                "    }\n}", 
                f"{forces_config}    }}\n}}"
            )
        else:
            # Add functions section if it doesn't exist
            content = content.replace(
                "// ************************************************************************* //",
                f"functions\n{{\n{forces_config}}}\n\n// ************************************************************************* //"
            )
            
        with open(controldict_path, 'w') as f:
            f.write(content)
            
    def analyze_streamlines(self, timestep="latest"):
        """Analyze generated streamlines"""
        logger.info("Analyzing streamlines data")
        
        streamlines_dir = self.post_dir / "streams100"
        if not streamlines_dir.exists():
            logger.error("Streamlines directory not found")
            return
            
        # Find latest timestep if not specified
        if timestep == "latest":
            timestep_dirs = [d for d in streamlines_dir.iterdir() if d.is_dir() and d.name.replace('.', '').isdigit()]
            if timestep_dirs:
                timestep = sorted(timestep_dirs, key=lambda x: float(x.name))[-1].name
            else:
                logger.error("No timestep directories found")
                return
                
        vtk_file = streamlines_dir / timestep / "tracks.vtk"
        if not vtk_file.exists():
            logger.error(f"VTK file not found: {vtk_file}")
            return
            
        # Read VTK file
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(str(vtk_file))
        reader.Update()
        
        polydata = reader.GetOutput()
        
        # Extract streamline statistics
        num_points = polydata.GetNumberOfPoints()
        num_lines = polydata.GetNumberOfLines()
        
        # Get velocity data
        velocity_array = polydata.GetPointData().GetArray("U")
        if velocity_array:
            velocities = vtk_to_numpy(velocity_array)
            
            self.results['streamlines'] = {
                'num_streamlines': num_lines,
                'num_points': num_points,
                'avg_velocity': float(np.mean(np.linalg.norm(velocities, axis=1))),
                'max_velocity': float(np.max(np.linalg.norm(velocities, axis=1))),
                'min_velocity': float(np.min(np.linalg.norm(velocities, axis=1)))
            }
            
            logger.info(f"Streamlines analysis complete:")
            logger.info(f"  Number of streamlines: {num_lines}")
            logger.info(f"  Total points: {num_points}")
            logger.info(f"  Velocity range: {self.results['streamlines']['min_velocity']:.1f} - {self.results['streamlines']['max_velocity']:.1f} m/s")
            
    def extract_pressure_data(self):
        """Extract pressure distribution on wing surface"""
        logger.info("Extracting pressure data")
        
        # Look for surface sampling data
        surfaces_dir = self.post_dir / "surfaces"
        if surfaces_dir.exists():
            # Find latest timestep pressure data
            pressure_files = list(surfaces_dir.glob("**/wing_p.raw"))
            if pressure_files:
                latest_file = sorted(pressure_files)[-1]
                
                # Read pressure data
                pressure_data = np.loadtxt(latest_file)
                
                self.results['pressure'] = {
                    'min_pressure': float(np.min(pressure_data)),
                    'max_pressure': float(np.max(pressure_data)),
                    'mean_pressure': float(np.mean(pressure_data)),
                    'pressure_range': float(np.max(pressure_data) - np.min(pressure_data))
                }
                
                logger.info(f"Pressure analysis complete:")
                logger.info(f"  Pressure range: {self.results['pressure']['min_pressure']:.0f} - {self.results['pressure']['max_pressure']:.0f} Pa")
                
    def save_results(self, output_file=None):
        """Save extracted results to JSON file"""
        if output_file is None:
            output_file = self.case_dir / "analysis_results.json"
            
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        logger.info(f"Results saved to {output_file}")
        
    def run_complete_postprocessing(self):
        """Run complete post-processing pipeline"""
        logger.info("=== Starting Complete Post-Processing ===")
        
        try:
            # Generate streamlines
            self.generate_streamlines()
            
            # Extract force coefficients
            self.extract_force_coefficients()
            
            # Analyze streamlines
            self.analyze_streamlines()
            
            # Extract pressure data
            self.extract_pressure_data()
            
            # Save results
            self.save_results()
            
            logger.info("=== Post-Processing Complete ===")
            return self.results
            
        except Exception as e:
            logger.error(f"Post-processing failed: {e}")
            raise

def extract_results(case_directory):
    """
    Main function to extract results from OpenFOAM case
    Returns dictionary with all analysis results
    """
    processor = McLarenWingPostProcessor(case_directory)
    return processor.run_complete_postprocessing()

def compare_cases(rigid_case, deformed_case, output_dir="comparison"):
    """Compare results between rigid and deformed wing cases"""
    logger.info("Comparing rigid vs deformed wing results")
    
    # Extract results from both cases
    rigid_results = extract_results(rigid_case)
    deformed_results = extract_results(deformed_case)
    
    # Calculate performance differences
    comparison = {
        'rigid_wing': rigid_results,
        'deformed_wing': deformed_results,
        'performance_delta': {}
    }
    
    if 'forces' in rigid_results and 'forces' in deformed_results:
        rigid_forces = rigid_results['forces']
        deformed_forces = deformed_results['forces']
        
        comparison['performance_delta'] = {
            'downforce_loss_N': rigid_forces['downforce'] - deformed_forces['downforce'],
            'downforce_loss_percent': ((rigid_forces['downforce'] - deformed_forces['downforce']) / rigid_forces['downforce']) * 100,
            'drag_increase_N': deformed_forces['drag'] - rigid_forces['drag'],
            'drag_increase_percent': ((deformed_forces['drag'] - rigid_forces['drag']) / rigid_forces['drag']) * 100,
            'efficiency_loss': rigid_forces['efficiency'] - deformed_forces['efficiency'],
            'efficiency_loss_percent': ((rigid_forces['efficiency'] - deformed_forces['efficiency']) / rigid_forces['efficiency']) * 100
        }
        
    # Save comparison results
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/comparison_results.json", 'w') as f:
        json.dump(comparison, f, indent=2)
        
    logger.info("Comparison analysis complete")
    return comparison

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='McLaren 2023 Wing Post-Processing')
    parser.add_argument('case_dir', help='OpenFOAM case directory')
    parser.add_argument('--compare', help='Compare with another case directory')
    parser.add_argument('--streamlines-only', action='store_true', help='Generate streamlines only')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    if args.streamlines_only:
        processor = McLarenWingPostProcessor(args.case_dir)
        processor.generate_streamlines()
    elif args.compare:
        compare_cases(args.case_dir, args.compare)
    else:
        extract_results(args.case_dir)