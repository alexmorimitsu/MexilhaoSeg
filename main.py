#!/usr/bin/env python3
"""
Main pipeline script that runs segment.py on all images in input_folder,
then runs inference.py on the outputs of segment.py.

This script orchestrates the complete pipeline:
1. Segment images using SAM (segment.py)
2. Run YOLO inference on segmented outputs (inference.py)
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import glob
import numpy as np


def get_image_files(folder_path):
    """
    Get all image files from the folder
    """
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
    
    return sorted(image_files)


def run_segmentation(input_folder, sam_params=None, model_type="vit_h", checkpoint_path=None):
    """
    Run segment.py on all images in the input folder
    
    Args:
        input_folder: Path to input folder containing images
        sam_params: Dictionary of SAM parameters (optional)
        model_type: SAM model type (default: vit_h)
        checkpoint_path: Path to SAM checkpoint (optional)
    
    Returns:
        Path to segmentation output folder
    """
    print("🔍 Starting segmentation phase...")
    
    # Get current script directory
    script_dir = Path(__file__).parent
    segment_script = script_dir / "segment.py"
    
    # Create output folder path
    output_folder = "./outputs/Mexilhao_SAM_Output"
    
    # Build command for segment.py
    cmd = [sys.executable, str(segment_script), "--input_folder", input_folder, "--output_folder", output_folder]
    
    # Add SAM parameters if provided
    if sam_params:
        for key, value in sam_params.items():
            cmd.extend([f"--{key}", str(value)])
    
    # Add model parameters
    cmd.extend(["--model_type", model_type])
    if checkpoint_path:
        cmd.extend(["--checkpoint_path", checkpoint_path])
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run segment.py
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Segmentation completed successfully!")
        print("Segmentation output:")
        print(result.stdout)
        
        if result.stderr:
            print("Segmentation warnings/errors:")
            print(result.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Segmentation failed with error code {e.returncode}")
        print("Error output:")
        print(e.stderr)
        return None
    
    return output_folder


def run_inference(input_folder, model_path, inference_params=None, segmentation_output_folder=None):
    """
    Run inference.py on the input folder
    
    Args:
        input_folder: Path to input folder containing images
        model_path: Path to YOLO model weights
        inference_params: Dictionary of inference parameters (optional)
        segmentation_output_folder: Path to segmentation output folder (for mask output location)
    
    Returns:
        Path to inference output folder
    """
    print("\n🎯 Starting inference phase...")
    
    # Get current script directory
    script_dir = Path(__file__).parent
    inference_script = script_dir / "inference.py"
    
    # Create inference output folder
    if segmentation_output_folder:
        inference_output_folder = "./outputs/inference_output"
        mask_output_folder = "./outputs/binary_masks"
        input_images_folder = "./outputs/input_images"
    else:
        # If no segmentation output folder, create outputs in ./outputs
        inference_output_folder = "./outputs/inference_output"
        mask_output_folder = "./outputs/binary_masks"
        input_images_folder = "./outputs/input_images"
    
    # Build command for inference.py
    cmd = [
        sys.executable, str(inference_script),
        "--model", model_path,
        "--input_folder", input_folder,
        "--output_folder", inference_output_folder,
        "--mask_output_folder", mask_output_folder,
        "--input_images_folder", input_images_folder
    ]
    
    # Add inference parameters if provided
    if inference_params:
        for key, value in inference_params.items():
            cmd.extend([f"--{key}", str(value)])
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run inference.py
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Inference completed successfully!")
        print("Inference output:")
        print(result.stdout)
        
        if result.stderr:
            print("Inference warnings/errors:")
            print(result.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Inference failed with error code {e.returncode}")
        print("Error output:")
        print(e.stderr)
        return None
    
    return inference_output_folder


def run_merge_masks(binary_fills_folder, sam_outputs_folder, merge_params=None):
    """
    Run merge_masks.py to merge binary fills with SAM outputs
    
    Args:
        binary_fills_folder: Path to folder containing binary fills
        sam_outputs_folder: Path to folder containing SAM outputs
        merge_params: Dictionary of merge parameters (optional)
    
    Returns:
        Path to merge masks output folder
    """
    print("\n🔗 Starting mask merging phase...")
    
    # Get current script directory
    script_dir = Path(__file__).parent
    merge_script = script_dir / "merge_masks.py"
    
    # Create merge output folder
    merge_output_folder = "./outputs/merged_masks"
    
    # Build command for merge_masks.py
    cmd = [
        sys.executable, str(merge_script),
        "--binary_fills", binary_fills_folder,
        "--sam_outputs", sam_outputs_folder,
        "--output_folder", merge_output_folder
    ]
    
    # Add merge parameters if provided
    if merge_params:
        for key, value in merge_params.items():
            cmd.extend([f"--{key}", str(value)])
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run merge_masks.py
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Mask merging completed successfully!")
        print("Merge masks output:")
        print(result.stdout)
        
        if result.stderr:
            print("Merge masks warnings/errors:")
            print(result.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Mask merging failed with error code {e.returncode}")
        print("Error output:")
        print(e.stderr)
        return None
    
    return merge_output_folder


def run_statistics(merge_output_folder):
    """
    Run statistics computation on merged masks
    
    Args:
        merge_output_folder: Path to folder containing merged masks (.npy files)
    
    Returns:
        Path to statistics output file
    """
    print("\n📊 Starting statistics computation phase...")
    
    # Import stats functions
    from stats import compute_stats, save_stats_to_csv
    
    # Find all .npy files in the merge output folder
    npy_files = []
    if os.path.exists(merge_output_folder):
        for file in os.listdir(merge_output_folder):
            if file.endswith('.npy'):
                npy_files.append(os.path.join(merge_output_folder, file))
    
    if not npy_files:
        print(f"⚠️  No .npy files found in {merge_output_folder}")
        return None
    
    print(f"Found {len(npy_files)} .npy files to process")
    
    # Compute statistics
    try:
        results = compute_stats(npy_files)
        
        if not results:
            print("❌ No statistics computed")
            return None
        
        # Save statistics to CSV
        stats_output_file = "./outputs/statistics.csv"
        save_stats_to_csv(results, stats_output_file)
        
        print("✅ Statistics computation completed successfully!")
        print(f"📁 Statistics saved to: {stats_output_file}")
        
        # Print summary statistics
        if results:
            total_files = len(results)
            avg_coverage = np.mean([r['coverage_percentage'] for r in results])
            avg_components = np.mean([r['num_connected_components'] for r in results])
            
            print(f"\n📈 Summary Statistics:")
            print(f"  - Total files processed: {total_files}")
            print(f"  - Average coverage: {avg_coverage:.2f}%")
            print(f"  - Average connected components: {avg_components:.2f}")
        
        return stats_output_file
        
    except Exception as e:
        print(f"❌ Statistics computation failed: {str(e)}")
        return None


def run_regression(statistics_csv_path, model_path):
    """
    Run regression analysis on statistics CSV
    
    Args:
        statistics_csv_path: Path to statistics CSV file
        model_path: Path to trained regression model
    
    Returns:
        Path to regression output CSV file
    """
    print("\n🤖 Starting regression analysis phase...")
    
    # Get current script directory
    script_dir = Path(__file__).parent
    regression_script = script_dir / "regression.py"
    
    # Create regression output path
    regression_output_csv = "./outputs/resultados.csv"
    
    # Build command for regression.py
    cmd = [
        sys.executable, str(regression_script),
        "--model_path", model_path,
        "--statistics_csv", statistics_csv_path,
        "--output_csv", regression_output_csv
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run regression.py
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Regression analysis completed successfully!")
        print("Regression output:")
        print(result.stdout)
        
        if result.stderr:
            print("Regression warnings/errors:")
            print(result.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Regression analysis failed with error code {e.returncode}")
        print("Error output:")
        print(e.stderr)
        return None
    
    return regression_output_csv


def main():
    """
    Main function to orchestrate the complete pipeline
    """
    parser = argparse.ArgumentParser(description='Run complete segmentation and inference pipeline')
    
    # Required arguments
    parser.add_argument('--input_folder', type=str, required=True,
                       help='Path to input folder containing images')
    parser.add_argument('--yolo_model', type=str, required=True,
                       help='Path to trained YOLO model weights (.pt file)')
    parser.add_argument('--binary_fills_folder', type=str, default=None,
                       help='Path to folder containing binary fills (optional - will auto-detect from YOLO binary masks if not provided)')
    
    # SAM model arguments
    parser.add_argument('--sam_model_type', type=str, default='vit_h',
                       choices=['vit_h', 'vit_l', 'vit_b'],
                       help='SAM model type (default: vit_h)')
    parser.add_argument('--sam_checkpoint', type=str, 
                       default='/home/alexandre/IO/Data/Models/sam_vit_h_4b8939.pth',
                       help='Path to SAM model checkpoint')
    
    # SAM parameters
    parser.add_argument('--points_per_side', type=int, default=64,
                       help='Number of points to sample along each side of the image')
    parser.add_argument('--pred_iou_thresh', type=float, default=0.88,
                       help='Predicted IoU threshold for mask filtering')
    parser.add_argument('--stability_score_thresh', type=float, default=0.95,
                       help='Stability score threshold for mask filtering')
    parser.add_argument('--crop_n_layers', type=int, default=0,
                       help='Number of layers to crop the image')
    parser.add_argument('--crop_n_points_downscale_factor', type=int, default=1,
                       help='Downscale factor for points in crop layers')
    parser.add_argument('--min_mask_region_area', type=int, default=0,
                       help='Minimum area for mask regions')
    parser.add_argument('--output_mode', type=str, default='binary_mask',
                       choices=['binary_mask', 'uncompressed_rle', 'coco_rle'],
                       help='Output mode for masks')
    
    # YOLO inference parameters
    parser.add_argument('--conf', type=float, default=0.25,
                       help='YOLO confidence threshold (default: 0.25)')
    parser.add_argument('--alpha', type=float, default=0.3,
                       help='Transparency factor for masks (0.0-1.0, default: 0.3)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size for YOLO inference (default: 640)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda, 0, 1, etc.)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible colors')
    
    # Merge masks parameters
    parser.add_argument('--merge_threshold', type=float, default=0.5,
                       help='Intersection threshold for merge masks (default: 0.5 = 50 percent)')
    parser.add_argument('--min_cc_area', type=int, default=100,
                       help='Minimum area for connected components in merge masks (default: 100)')
    
    # Pipeline control
    parser.add_argument('--skip_segmentation', action='store_true',
                       help='Skip segmentation phase (use existing segmentation outputs)')
    parser.add_argument('--skip_inference', action='store_true',
                       help='Skip inference phase (only run segmentation)')
    parser.add_argument('--skip_merge_masks', action='store_true',
                       help='Skip merge masks phase')
    parser.add_argument('--skip_statistics', action='store_true',
                       help='Skip statistics computation phase')
    parser.add_argument('--skip_regression', action='store_true',
                       help='Skip regression analysis phase')
    parser.add_argument('--regression_model', type=str, default='./Modelos/huber_neural_network_model.pth',
                       help='Path to regression model file (default: ./Modelos/huber_neural_network_model.pth)')
    parser.add_argument('--segmentation_output_folder', type=str, default=None,
                       help='Path to existing segmentation output folder (used with --skip_segmentation)')
    
    args = parser.parse_args()
    
    # Validate input folder
    if not os.path.exists(args.input_folder):
        print(f"❌ Error: Input folder {args.input_folder} does not exist")
        return 1
    
    # Check for images in input folder
    image_files = get_image_files(args.input_folder)
    if not image_files:
        print(f"❌ Error: No image files found in {args.input_folder}")
        return 1
    
    print(f"📁 Found {len(image_files)} images in input folder")
    
    # Prepare SAM parameters
    sam_params = {
        'points_per_side': args.points_per_side,
        'pred_iou_thresh': args.pred_iou_thresh,
        'stability_score_thresh': args.stability_score_thresh,
        'crop_n_layers': args.crop_n_layers,
        'crop_n_points_downscale_factor': args.crop_n_points_downscale_factor,
        'min_mask_region_area': args.min_mask_region_area,
        'output_mode': args.output_mode
    }
    
    # Prepare inference parameters
    inference_params = {
        'conf': args.conf,
        'alpha': args.alpha,
        'imgsz': args.imgsz,
        'device': args.device,
        'seed': args.seed
    }
    
    # Prepare merge masks parameters
    merge_params = {
        'threshold': args.merge_threshold,
        'min_cc_area': args.min_cc_area
    }
    
    # Phase 1: Segmentation
    segmentation_output_folder = None
    if not args.skip_segmentation:
        segmentation_output_folder = run_segmentation(
            args.input_folder, 
            sam_params, 
            args.sam_model_type, 
            args.sam_checkpoint
        )
        if segmentation_output_folder is None:
            print("❌ Segmentation phase failed")
            return 1
    else:
        if args.segmentation_output_folder:
            segmentation_output_folder = args.segmentation_output_folder
        else:
            # Try to find the default output folder
            segmentation_output_folder = "./outputs/Mexilhao_SAM_Output"
        
        if not os.path.exists(segmentation_output_folder):
            print(f"❌ Error: Segmentation output folder {segmentation_output_folder} does not exist")
            return 1
        
        print(f"✅ Using existing segmentation output folder: {segmentation_output_folder}")
    
    # Phase 2: Inference
    if not args.skip_inference:
        # Validate YOLO model
        if not os.path.exists(args.yolo_model):
            print(f"❌ Error: YOLO model file {args.yolo_model} does not exist")
            return 1
        
        inference_output_folder = run_inference(
            args.input_folder, 
            args.yolo_model, 
            inference_params,
            segmentation_output_folder
        )
        if inference_output_folder is None:
            print("❌ Inference phase failed")
            return 1
        
        print(f"\n🎉 Complete pipeline finished successfully!")
        print(f"📁 Segmentation outputs: {segmentation_output_folder}")
        print(f"📁 Inference outputs: {inference_output_folder}")
        print(f"📁 Binary masks: ./outputs/binary_masks")
        print(f"📁 Input images: ./outputs/input_images")
    else:
        print(f"\n🎉 Segmentation phase completed successfully!")
        print(f"📁 Segmentation outputs: {segmentation_output_folder}")
    
    # Phase 3: Merge Masks
    if not args.skip_merge_masks:
        # Determine binary fills folder
        binary_fills_folder = None
        
        if args.binary_fills_folder:
            # Use explicitly provided binary fills folder
            binary_fills_folder = args.binary_fills_folder
            print(f"\n🔗 Using provided binary fills folder: {binary_fills_folder}")
        else:
            # Auto-detect binary fills from YOLO binary masks
            if segmentation_output_folder:
                yolo_binary_masks_folder = "./outputs/binary_masks"
                if os.path.exists(yolo_binary_masks_folder):
                    binary_fills_folder = yolo_binary_masks_folder
                    print(f"\n🔗 Auto-detected binary fills from YOLO masks: {binary_fills_folder}")
                else:
                    print(f"\n⚠️  Skipping merge masks phase: YOLO binary masks folder not found at {yolo_binary_masks_folder}")
                    binary_fills_folder = None
            else:
                print(f"\n⚠️  Skipping merge masks phase: No segmentation output folder available for auto-detection")
                binary_fills_folder = None
        
        if binary_fills_folder:
            # Validate binary fills folder
            if not os.path.exists(binary_fills_folder):
                print(f"❌ Error: Binary fills folder {binary_fills_folder} does not exist")
                return 1
            
            merge_output_folder = run_merge_masks(
                binary_fills_folder,
                segmentation_output_folder,
                merge_params
            )
            if merge_output_folder is None:
                print("❌ Merge masks phase failed")
                return 1
            
            print(f"\n🔗 Merge masks phase completed successfully!")
            print(f"📁 Merged masks outputs: {merge_output_folder}")
            
            # Phase 4: Statistics (if merge masks completed and not skipped)
            stats_output_file = None
            if not args.skip_statistics:
                stats_output_file = run_statistics(merge_output_folder)
                if stats_output_file:
                    print(f"\n📊 Statistics phase completed successfully!")
                    print(f"📁 Statistics saved to: {stats_output_file}")
                else:
                    print("❌ Statistics phase failed")
                    return 1
            else:
                print("\n⏭️  Skipping statistics phase: --skip_statistics specified")
                # Try to find existing statistics file
                stats_output_file = "./outputs/statistics.csv"
                if not os.path.exists(stats_output_file):
                    print(f"⚠️  No existing statistics file found at {stats_output_file}")
                    stats_output_file = None
            
            # Phase 5: Regression Analysis (if statistics completed and not skipped)
            if not args.skip_regression and stats_output_file:
                # Validate regression model
                if not os.path.exists(args.regression_model):
                    print(f"❌ Error: Regression model file {args.regression_model} does not exist")
                    return 1
                
                regression_output_file = run_regression(stats_output_file, args.regression_model)
                if regression_output_file:
                    print(f"\n🤖 Regression analysis phase completed successfully!")
                    print(f"📁 Regression results saved to: {regression_output_file}")
                else:
                    print("❌ Regression analysis phase failed")
                    return 1
            elif args.skip_regression:
                print("\n⏭️  Skipping regression analysis phase: --skip_regression specified")
            elif not stats_output_file:
                print("\n⏭️  Skipping regression analysis phase: No statistics file available")
    else:
        print("\n⏭️  Skipping merge masks phase: --skip_merge_masks specified")
        # Try to find existing statistics file for regression
        stats_output_file = "./outputs/statistics.csv"
        if not os.path.exists(stats_output_file):
            print(f"⚠️  No existing statistics file found at {stats_output_file}")
            stats_output_file = None
    
    # Phase 5: Regression Analysis (independent of merge masks, if statistics available and not skipped)
    if not args.skip_regression and stats_output_file:
        # Validate regression model
        if not os.path.exists(args.regression_model):
            print(f"❌ Error: Regression model file {args.regression_model} does not exist")
            return 1
        
        regression_output_file = run_regression(stats_output_file, args.regression_model)
        if regression_output_file:
            print(f"\n🤖 Regression analysis phase completed successfully!")
            print(f"📁 Regression results saved to: {regression_output_file}")
        else:
            print("❌ Regression analysis phase failed")
            return 1
    elif args.skip_regression:
        print("\n⏭️  Skipping regression analysis phase: --skip_regression specified")
    elif not stats_output_file:
        print("\n⏭️  Skipping regression analysis phase: No statistics file available")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
