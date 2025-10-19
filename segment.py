import os
import cv2
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import torch
from pathlib import Path
import glob
import argparse

# Input folder path (default - can be overridden via command line)
input_folder = "./inputs"

# Default SAM generator parameters
DEFAULT_SAM_PARAMS = {
    'points_per_side': 64,
    'pred_iou_thresh': 0.88,
    'stability_score_thresh': 0.95,
    'crop_n_layers': 0,
    'crop_n_points_downscale_factor': 1,
    'min_mask_region_area': 0,
    'output_mode': 'binary_mask'
}

# Conservative approach (moderate increase)
# DEFAULT_SAM_PARAMS = {
#     'points_per_side': 128,           # Double the points
#     'pred_iou_thresh': 0.7,           # Lower threshold
#     'stability_score_thresh': 0.8,    # Lower threshold
#     'crop_n_layers': 1,               # Add one crop layer
#     'crop_n_points_downscale_factor': 2,
#     'min_mask_region_area': 100,      # Small minimum area
#     'output_mode': 'binary_mask'
# }

# # Aggressive approach (many more segments)
# DEFAULT_SAM_PARAMS = {
#     'points_per_side': 256,           # Many more points
#     'pred_iou_thresh': 0.5,           # Much lower threshold
#     'stability_score_thresh': 0.6,    # Much lower threshold
#     'crop_n_layers': 2,               # Multiple crop layers
#     'crop_n_points_downscale_factor': 2,
#     'min_mask_region_area': 50,       # Very small minimum area
#     'output_mode': 'binary_mask'
# }

def parse_arguments():
    """
    Parse command line arguments for SAM parameters
    """
    parser = argparse.ArgumentParser(description='Segment images using SAM with configurable parameters')
    
    # SAM generator parameters
    parser.add_argument('--points_per_side', type=int, default=DEFAULT_SAM_PARAMS['points_per_side'],
                       help='Number of points to sample along each side of the image (default: 32)')
    parser.add_argument('--pred_iou_thresh', type=float, default=DEFAULT_SAM_PARAMS['pred_iou_thresh'],
                       help='Predicted IoU threshold for mask filtering (default: 0.88)')
    parser.add_argument('--stability_score_thresh', type=float, default=DEFAULT_SAM_PARAMS['stability_score_thresh'],
                       help='Stability score threshold for mask filtering (default: 0.95)')
    parser.add_argument('--crop_n_layers', type=int, default=DEFAULT_SAM_PARAMS['crop_n_layers'],
                       help='Number of layers to crop the image (default: 0)')
    parser.add_argument('--crop_n_points_downscale_factor', type=int, default=DEFAULT_SAM_PARAMS['crop_n_points_downscale_factor'],
                       help='Downscale factor for points in crop layers (default: 1)')
    parser.add_argument('--min_mask_region_area', type=int, default=DEFAULT_SAM_PARAMS['min_mask_region_area'],
                       help='Minimum area for mask regions (default: 0)')
    parser.add_argument('--output_mode', type=str, default=DEFAULT_SAM_PARAMS['output_mode'],
                       choices=['binary_mask', 'uncompressed_rle', 'coco_rle'],
                       help='Output mode for masks (default: binary_mask)')
    
    # Other options
    parser.add_argument('--input_folder', type=str, default=input_folder,
                       help=f'Input folder path (default: {input_folder})')
    parser.add_argument('--output_folder', type=str, default='/app/outputs/Mexilhao_SAM_Output',
                       help='Output folder path (default: /app/outputs/Mexilhao_SAM_Output)')
    parser.add_argument('--model_type', type=str, default='vit_h',
                       choices=['vit_h', 'vit_l', 'vit_b'],
                       help='SAM model type (default: vit_h)')
    parser.add_argument('--checkpoint_path', type=str, 
                       default='./Modelos/sam_vit_h_4b8939.pth',
                       help='Path to SAM model checkpoint')
    
    return parser.parse_args()

def load_sam_model(model_type="vit_h", checkpoint_path="./Modelos/sam_vit_h_4b8939.pth"):
    """
    Load the SAM model. You'll need to download the model checkpoint first.
    """
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Error: SAM checkpoint not found at {checkpoint_path}")
        print("Please download the SAM model checkpoint from:")
        print("https://github.com/facebookresearch/segment-anything")
        return None
    
    # Load the model
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    
    return sam

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

def extract_borders_from_labels_image(labels_image):
    """
    Extract borders from a colored labels image by finding pixels where neighbors have different colors
    Also considers pixels at image boundaries as borders when they have neighbors outside the image
    """
    # Create border image
    h, w = labels_image.shape[:2]
    border_image = np.zeros((h, w), dtype=np.uint8)
    
    # Check each pixel for border condition
    for y in range(h):
        for x in range(w):
            current_pixel = labels_image[y, x]
            
            # Check 8-neighborhood, handling boundary conditions
            neighbors = []
            
            # Top-left
            if y > 0 and x > 0:
                neighbors.append(labels_image[y-1, x-1])
            else:
                neighbors.append(None)  # Outside image boundary
                
            # Top
            if y > 0:
                neighbors.append(labels_image[y-1, x])
            else:
                neighbors.append(None)  # Outside image boundary
                
            # Top-right
            if y > 0 and x < w-1:
                neighbors.append(labels_image[y-1, x+1])
            else:
                neighbors.append(None)  # Outside image boundary
                
            # Left
            if x > 0:
                neighbors.append(labels_image[y, x-1])
            else:
                neighbors.append(None)  # Outside image boundary
                
            # Right
            if x < w-1:
                neighbors.append(labels_image[y, x+1])
            else:
                neighbors.append(None)  # Outside image boundary
                
            # Bottom-left
            if y < h-1 and x > 0:
                neighbors.append(labels_image[y+1, x-1])
            else:
                neighbors.append(None)  # Outside image boundary
                
            # Bottom
            if y < h-1:
                neighbors.append(labels_image[y+1, x])
            else:
                neighbors.append(None)  # Outside image boundary
                
            # Bottom-right
            if y < h-1 and x < w-1:
                neighbors.append(labels_image[y+1, x+1])
            else:
                neighbors.append(None)  # Outside image boundary
            
            # If current pixel is different from any neighbor, it's a border
            # Compare RGB values, treating None (outside boundary) as different
            for neighbor in neighbors:
                if neighbor is None or not np.array_equal(current_pixel, neighbor):
                    border_image[y, x] = 255
                    break
    
    return border_image

def extract_borders_from_mask(mask):
    """
    Extract borders from a binary mask by finding pixels where neighbors have different labels
    """
    # Convert mask to binary (0 and 1)
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8)
    
    # Convert to binary (0 and 1)
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Create border image
    border_image = np.zeros_like(binary_mask)
    
    # Get image dimensions
    h, w = binary_mask.shape
    
    # Check each pixel for border condition
    for y in range(1, h-1):
        for x in range(1, w-1):
            current_pixel = binary_mask[y, x]
            
            # Check 8-neighborhood
            neighbors = [
                binary_mask[y-1, x-1], binary_mask[y-1, x], binary_mask[y-1, x+1],
                binary_mask[y, x-1],                           binary_mask[y, x+1],
                binary_mask[y+1, x-1], binary_mask[y+1, x], binary_mask[y+1, x+1]
            ]
            
            # If current pixel is different from any neighbor, it's a border
            if any(neighbor != current_pixel for neighbor in neighbors):
                border_image[y, x] = 1
    
    # Convert to 0-255 range
    border_image = border_image * 255
    
    return border_image

def create_border_overlay(original_image, border_image, overlay_color=(0, 0, 0)):
    """
    Create an overlay of border pixels on the original image
    
    Args:
        original_image: Original BGR image
        border_image: Binary border image where non-zero pixels are borders
        overlay_color: Color for the border pixels (BGR format) - default is black
    
    Returns:
        Overlay image with border pixels drawn in specified color
    """
    # Create a copy of the original image
    overlay = original_image.copy()
    
    # Create a mask where non-zero pixels in border_image become True
    mask = border_image > 0
    
    # Apply the color to the border pixels
    overlay[mask] = overlay_color
    
    return overlay

def process_image_with_sam(image_path, sam_model, output_folder, sam_params=None):
    """
    Process a single image with SAM and extract borders
    """
    if sam_params is None:
        sam_params = DEFAULT_SAM_PARAMS
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create automatic mask generator with custom parameters
    mask_generator = SamAutomaticMaskGenerator(
        model=sam_model,
        points_per_side=sam_params['points_per_side'],
        pred_iou_thresh=sam_params['pred_iou_thresh'],
        stability_score_thresh=sam_params['stability_score_thresh'],
        crop_n_layers=sam_params['crop_n_layers'],
        crop_n_points_downscale_factor=sam_params['crop_n_points_downscale_factor'],
        min_mask_region_area=sam_params['min_mask_region_area'],
        output_mode=sam_params['output_mode']
    )
    
    # Generate automatic mask proposals
    masks = mask_generator.generate(image_rgb)
    
    if not masks:
        print(f"No masks generated for {image_path}")
        return
    
    # Create colored labels image
    h, w = image_rgb.shape[:2]
    labels_image = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Generate random colors for each mask
    np.random.seed(42)  # For reproducible colors
    colors = []
    for i in range(len(masks)):
        # Generate bright, distinct colors
        color = np.random.randint(50, 255, 3).tolist()
        colors.append(color)
    
    # Apply colors to each mask
    for i, mask_data in enumerate(masks):
        mask = mask_data['segmentation']
        color = colors[i]
        
        # Apply color to mask pixels
        for c in range(3):  # RGB channels
            labels_image[:, :, c] = np.where(mask, color[c], labels_image[:, :, c])
    
    # Extract borders from the labels image
    border_image = extract_borders_from_labels_image(labels_image)
    
    # Use the mask with highest area (usually the main object) for reference
    # Sort masks by area in descending order
    masks_sorted = sorted(masks, key=lambda x: x['area'], reverse=True)
    best_mask = masks_sorted[0]
    
    # Get the mask array
    mask = best_mask['segmentation']
    
    # Convert mask to binary (0 and 255)
    binary_mask = mask.astype(np.uint8) * 255
    
    # Create overlay on original image with segmented areas in black
    overlay_image = create_border_overlay(image, border_image, overlay_color=(0, 0, 0))
    
    # Create output filename
    base_name = Path(image_path).stem
    mask_output_path = os.path.join(output_folder, f"{base_name}_mask.png")
    border_output_path = os.path.join(output_folder, f"{base_name}_border.png")
    labels_output_path = os.path.join(output_folder, f"{base_name}_labels.png")
    overlay_output_path = os.path.join(output_folder, f"{base_name}_overlay.png")
    
    # Save images
    cv2.imwrite(mask_output_path, border_image)  # Save border as binary mask
    cv2.imwrite(border_output_path, binary_mask)  # Save original mask as border
    cv2.imwrite(labels_output_path, cv2.cvtColor(labels_image, cv2.COLOR_RGB2BGR))  # Save colored labels
    cv2.imwrite(overlay_output_path, overlay_image)  # Save overlay with borders on original image
    
    print(f"Processed {image_path}")
    print(f"  - Binary mask (borders) saved: {mask_output_path}")
    print(f"  - Original mask saved: {border_output_path}")
    print(f"  - Colored labels saved: {labels_output_path}")
    print(f"  - Border overlay saved: {overlay_output_path}")
    print(f"  - Number of masks: {len(masks)}")
    print(f"  - Best mask area: {best_mask['area']} pixels")

def main():
    """
    Main function to process all images in the input folder
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Update input folder from arguments
    input_folder = args.input_folder
    
    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: Input folder {input_folder} does not exist")
        return
    
    # Create output folder
    output_folder = args.output_folder
    try:
        os.makedirs(output_folder, exist_ok=True)
    except PermissionError:
        print(f"Error: Permission denied when creating output folder {output_folder}")
        print("This might be due to Docker volume permissions.")
        print("Try running the Docker command with proper volume permissions or check the mounted directory.")
        return
    
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
    
    # Print parameters being used
    print("SAM Parameters:")
    for key, value in sam_params.items():
        print(f"  {key}: {value}")
    print()
    
    # Load SAM model
    print("Loading SAM model...")
    sam_model = load_sam_model(args.model_type, args.checkpoint_path)
    if sam_model is None:
        return
    
    # Get all image files
    image_files = get_image_files(input_folder)
    if not image_files:
        print(f"No image files found in {input_folder}")
        return
    
    print(f"Found {len(image_files)} image files")
    
    # Process each image
    for i, image_path in enumerate(image_files, 1):
        print(f"\nProcessing image {i}/{len(image_files)}: {os.path.basename(image_path)}")
        try:
            process_image_with_sam(image_path, sam_model, output_folder, sam_params)
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
    
    print(f"\nProcessing complete! Output saved to: {output_folder}")

if __name__ == "__main__":
    main()

