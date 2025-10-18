#!/usr/bin/env python3
"""
Merge masks script for processing binary fills and SAM outputs.

This script processes two folders:
1. binary_fills: Contains binary images with filled objects
2. sam_outputs: Contains images, some of which are binary masks of borders

For every matching filename, it computes intersections between connected components
of SAM masks and filled masks, keeping only CCs that overlap above a threshold.
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path
import glob
from scipy import ndimage
from skimage.measure import label, regionprops


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Merge binary fills with SAM outputs based on connected component intersections'
    )
    
    parser.add_argument('--binary_fills', type=str, required=True,
                       help='Path to folder containing binary images with filled objects')
    parser.add_argument('--sam_outputs', type=str, required=True,
                       help='Path to folder containing SAM output images (some are binary masks)')
    parser.add_argument('--output_folder', type=str, default='./outputs',
                       help='Path to output folder for labels images (default: ./outputs)')
    parser.add_argument('--threshold', type=float, default=0.001,
                       help='Intersection threshold as percentage (default: 0.001 = 0.1%)')
    parser.add_argument('--min_cc_area', type=int, default=1,
                       help='Minimum area for connected components to consider (default: 1)')
    
    return parser.parse_args()


def get_image_files(folder_path):
    """Get all image files from the folder."""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
    
    return sorted(image_files)


def get_base_filename(filepath):
    """Extract base filename without extension."""
    return Path(filepath).stem


def find_matching_files(binary_fills_folder, sam_outputs_folder):
    """Find matching filenames between the two folders."""
    binary_files = get_image_files(binary_fills_folder)
    sam_files = get_image_files(sam_outputs_folder)
    
    # Create dictionaries mapping base filenames to full paths
    binary_dict = {get_base_filename(f): f for f in binary_files}
    sam_dict = {get_base_filename(f): f for f in sam_files}
    
    # Find common base filenames
    common_names = set(binary_dict.keys()) & set(sam_dict.keys())
    
    matches = []
    for name in common_names:
        matches.append({
            'name': name,
            'binary_path': binary_dict[name],
            'sam_path': sam_dict[name]
        })
    
    return matches


def load_binary_image(image_path):
    """Load and convert image to binary format."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert to binary (0 and 1)
    binary_img = (img > 0).astype(np.uint8)
    return binary_img


def get_connected_components(mask, invert=False):
    """Get connected components from a binary mask."""
    # Optionally invert the mask to analyze black regions instead of white regions
    if invert:
        mask = 1 - mask
    
    # Use scipy's label function for connected component analysis
    labeled_mask = label(mask, connectivity=2)
    
    # Get properties of each connected component
    regions = regionprops(labeled_mask)
    
    return labeled_mask, regions


def compute_intersection_ratio(cc_mask, filled_mask):
    """Compute the intersection ratio between a connected component and filled mask."""
    # Intersection: pixels that are 1 in both masks
    intersection = np.logical_and(cc_mask, filled_mask)
    
    # Area of the connected component
    cc_area = np.sum(cc_mask)
    
    if cc_area == 0:
        return 0.0
    
    # Intersection area
    intersection_area = np.sum(intersection)
    
    # Ratio of intersection to CC area
    ratio = intersection_area / cc_area
    
    return ratio


def create_labels_image(sam_mask, filled_mask, threshold=0.5, min_cc_area=100):
    """
    Create a labels image by filtering connected components based on intersection with filled mask.
    
    Args:
        sam_mask: Binary mask from SAM output
        filled_mask: Binary mask with filled objects
        threshold: Minimum intersection ratio to keep a CC
        min_cc_area: Minimum area for CCs to consider
    
    Returns:
        Tuple of (labels_image, colored_labels_image) where:
        - labels_image: Numeric labels where each kept CC has a unique label value
        - colored_labels_image: RGB image with random colors for each label
    """
    # Get connected components from SAM mask (inverted to analyze black regions)
    labeled_mask, regions = get_connected_components(sam_mask, invert=True)
    
    # Create output labels image
    labels_image = np.zeros_like(sam_mask, dtype=np.uint16)
    current_label = 1
    
    # Store valid regions for color assignment
    valid_regions = []
    
    # Process each connected component
    for region in regions:
        # Skip small components
        if region.area < min_cc_area:
            continue
        
        # Create mask for this connected component
        cc_mask = (labeled_mask == region.label).astype(np.uint8)
        
        # Compute intersection ratio with filled mask
        intersection_ratio = compute_intersection_ratio(cc_mask, filled_mask)
        
        # Keep CC if intersection ratio is above threshold
        if intersection_ratio >= threshold:
            labels_image[cc_mask == 1] = current_label
            valid_regions.append((current_label, cc_mask))
            current_label += 1
    
    # Create colored labels image
    colored_labels_image = np.zeros((sam_mask.shape[0], sam_mask.shape[1], 3), dtype=np.uint8)
    
    # Generate random colors for each valid label
    np.random.seed(42)  # For reproducible colors
    for label_id, cc_mask in valid_regions:
        # Generate random RGB color
        color = np.random.randint(50, 255, 3)
        
        # Apply color to the connected component
        colored_labels_image[cc_mask == 1] = color
    
    # Copy white pixels from original SAM mask (which is already borders) to overlay them
    border_mask = sam_mask > 0
    colored_labels_image[border_mask] = [255, 255, 255]  # White borders
    
    return labels_image, colored_labels_image


def process_matching_files(matches, output_folder, threshold=0.5, min_cc_area=100):
    """Process all matching files and create labels images."""
    os.makedirs(output_folder, exist_ok=True)
    
    processed_count = 0
    
    for match in matches:
        try:
            print(f"Processing: {match['name']}")
            
            # Load images
            binary_fill = load_binary_image(match['binary_path'])
            sam_output = load_binary_image(match['sam_path'])
            
            # Ensure images have the same dimensions
            if binary_fill.shape != sam_output.shape:
                print(f"  Warning: Image dimensions don't match for {match['name']}")
                print(f"    Binary fill: {binary_fill.shape}, SAM output: {sam_output.shape}")
                # Resize sam_output to match binary_fill
                sam_output = cv2.resize(sam_output, (binary_fill.shape[1], binary_fill.shape[0]))
            
            # Get connected components from SAM output for counting (inverted to analyze black regions)
            labeled_mask, regions = get_connected_components(sam_output, invert=True)
            total_ccs = len(regions)
            print(f"  SAM output has {total_ccs} connected components (black regions)")
            
            # Create filtered labels image
            labels_image, colored_labels_image = create_labels_image(sam_output, binary_fill, threshold, min_cc_area)
            
            # Save filtered labels as .npy file
            npy_output_path = os.path.join(output_folder, f"{match['name']}_labels.npy")
            np.save(npy_output_path, labels_image)
            
            # Save filtered colored labels as .png file
            colored_output_path = os.path.join(output_folder, f"{match['name']}.png")
            cv2.imwrite(colored_output_path, cv2.cvtColor(colored_labels_image, cv2.COLOR_RGB2BGR))
            
            # Count number of labels (excluding background)
            num_labels = len(np.unique(labels_image)) - 1
            print(f"  Kept {num_labels} components after intersection filtering (threshold: {threshold*100:.1f}%)")
            print(f"  Saved filtered labels (.npy) to: {npy_output_path}")
            print(f"  Saved colored mask (.png) to: {colored_output_path}")
            
            processed_count += 1
            
        except Exception as e:
            print(f"  Error processing {match['name']}: {str(e)}")
    
    return processed_count


def main():
    """Main function."""
    args = parse_arguments()
    
    # Check if input folders exist
    if not os.path.exists(args.binary_fills):
        print(f"Error: Binary fills folder not found: {args.binary_fills}")
        return
    
    if not os.path.exists(args.sam_outputs):
        print(f"Error: SAM outputs folder not found: {args.sam_outputs}")
        return
    
    print(f"Binary fills folder: {args.binary_fills}")
    print(f"SAM outputs folder: {args.sam_outputs}")
    print(f"Output folder: {args.output_folder}")
    print(f"Intersection threshold: {args.threshold * 100:.1f}%")
    print(f"Minimum CC area: {args.min_cc_area}")
    print()
    
    # Find matching files
    print("Finding matching files...")
    matches = find_matching_files(args.binary_fills, args.sam_outputs)
    
    if not matches:
        print("No matching files found between the two folders.")
        return
    
    print(f"Found {len(matches)} matching files:")
    for match in matches:
        print(f"  - {match['name']}")
    print()
    
    # Process matching files
    print("Processing files...")
    processed_count = process_matching_files(
        matches, 
        args.output_folder, 
        args.threshold, 
        args.min_cc_area
    )
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed {processed_count}/{len(matches)} files")
    print(f"Output files saved to: {args.output_folder}")
    print(f"  - Labels: *_labels.npy")
    print(f"  - Colored mask: *.png (with white borders)")


if __name__ == "__main__":
    main()
