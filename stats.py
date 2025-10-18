import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_dilation, binary_erosion
import os
import csv


def compute_stats(npy_files):
    """
    Compute statistics for a list of .npy files containing 2D label images.
    
    Args:
        npy_files (list): List of paths to .npy files
        
    Returns:
        list: List of dictionaries containing statistics for each file
    """
    results = []
    
    for npy_file in npy_files:
        try:
            # Load the .npy file
            labels = np.load(npy_file)
            
            # Ensure it's 2D
            if labels.ndim != 2:
                print(f"Warning: {npy_file} is not 2D, skipping...")
                continue
            
            # Get image dimensions
            height, width = labels.shape
            total_pixels = height * width
            
            # Create binary mask for non-zero labels
            binary_mask = (labels > 0).astype(int)
            
            # Create morphological opening mask (dilate then erode by 5 pixels)
            structure = ndimage.generate_binary_structure(2, 1)  # 4-connectivity
            
            # Dilate by 5 pixels
            dilated_mask = binary_mask.copy()
            for _ in range(5):
                dilated_mask = binary_dilation(dilated_mask, structure=structure)
            
            # Erode by 5 pixels
            morphed_mask = dilated_mask.copy()
            for _ in range(5):
                morphed_mask = binary_erosion(morphed_mask, structure=structure)
            
            # Count non-zero pixels (coverage area using morphed mask)
            non_zero_pixels = np.count_nonzero(morphed_mask)
            coverage_percentage = (non_zero_pixels / total_pixels) * 100
            
            # Find connected components (excluding label 0)
            # Find connected components
            labeled_array, num_components = ndimage.label(binary_mask)
            
            # Calculate areas of each connected component
            areas = []
            for i in range(1, num_components + 1):
                component_area = np.sum(labeled_array == i)
                areas.append(component_area)
            
            # Find connected components in dilated mask
            dilated_labeled_array, dilated_num_components = ndimage.label(dilated_mask)
            
            # Calculate areas of each connected component in dilated mask
            dilated_areas = []
            for i in range(1, dilated_num_components + 1):
                component_area = np.sum(dilated_labeled_array == i)
                dilated_areas.append(component_area)
            
            # Calculate statistics for original mask
            if areas:
                avg_area = np.mean(areas)
                std_area = np.std(areas)
            else:
                avg_area = 0
                std_area = 0
            
            # Calculate statistics for dilated mask
            if dilated_areas:
                dilated_avg_area = np.mean(dilated_areas)
                dilated_std_area = np.std(dilated_areas)
            else:
                dilated_avg_area = 0
                dilated_std_area = 0
            
            # Store results
            result = {
                'filename': os.path.basename(npy_file),
                'filepath': npy_file,
                'image_height': height,
                'image_width': width,
                'total_pixels': total_pixels,
                'non_zero_pixels': non_zero_pixels,
                'coverage_percentage': round(coverage_percentage, 2),
                'num_connected_components': num_components,
                'avg_area': round(avg_area, 2),
                'std_area': round(std_area, 2),
                'min_area': min(areas) if areas else 0,
                'max_area': max(areas) if areas else 0,
                'num_dilated_connected_components': dilated_num_components,
                'dilated_avg_area': round(dilated_avg_area, 2),
                'dilated_std_area': round(dilated_std_area, 2),
                'dilated_min_area': min(dilated_areas) if dilated_areas else 0,
                'dilated_max_area': max(dilated_areas) if dilated_areas else 0
            }
            
            results.append(result)
            print(f"Processed {os.path.basename(npy_file)}: {num_components} components, {dilated_num_components} dilated components, {coverage_percentage:.2f}% coverage")
            
        except Exception as e:
            print(f"Error processing {npy_file}: {str(e)}")
            continue
    
    return results


def save_stats_to_csv(results, output_file):
    """
    Save statistics results to a CSV file.
    
    Args:
        results (list): List of dictionaries containing statistics
        output_file (str): Path to output CSV file
    """
    if not results:
        print("No results to save.")
        return
    
    fieldnames = [
        'filename', 'filepath', 'image_height', 'image_width', 'total_pixels',
        'non_zero_pixels', 'coverage_percentage', 'num_connected_components',
        'avg_area', 'std_area', 'min_area', 'max_area',
        'num_dilated_connected_components', 'dilated_avg_area', 'dilated_std_area',
        'dilated_min_area', 'dilated_max_area'
    ]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Statistics saved to {output_file}")
