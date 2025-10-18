#!/usr/bin/env python3
"""
Inference script for trained YOLOv11-seg model.

This script runs inference on a folder of images using a trained YOLOv11-seg model
and visualizes the segmentations with random colors without class names.
Additionally, it can create binary masks where pixels with segmentations are white (255)
and pixels without segmentations are black (0).
"""

import cv2
import numpy as np
import os
import random
import argparse
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb


def generate_random_color():
    """Generate a random bright color."""
    # Generate random HSV values
    h = random.random()  # Hue (0-1)
    s = 0.7 + random.random() * 0.3  # Saturation (0.7-1.0) for bright colors
    v = 0.8 + random.random() * 0.2  # Value (0.8-1.0) for bright colors
    
    # Convert HSV to RGB
    rgb = hsv_to_rgb([h, s, v])
    
    # Convert to BGR for OpenCV (0-255 range)
    bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
    return bgr


def create_binary_mask(image, results, conf_threshold=0.25):
    """
    Create a binary mask from segmentation results.
    
    Args:
        image: Input image (numpy array)
        results: YOLO results object
        conf_threshold: Confidence threshold for filtering detections
    
    Returns:
        Binary mask where 0 = no segmentation, 255 = segmentation found
    """
    height, width = image.shape[:2]
    
    # Initialize binary mask with zeros
    binary_mask = np.zeros((height, width), dtype=np.uint8)
    
    if results.masks is None:
        return binary_mask
    
    # Get masks and boxes
    masks = results.masks.data.cpu().numpy()
    boxes = results.boxes.data.cpu().numpy()
    
    # Filter by confidence threshold
    valid_indices = boxes[:, 4] >= conf_threshold
    masks = masks[valid_indices]
    
    # Combine all valid masks
    for mask in masks:
        # Resize mask to image dimensions
        mask_resized = cv2.resize(mask, (width, height))
        
        # Create binary mask for this segmentation
        mask_binary = (mask_resized > 0.5).astype(np.uint8)
        
        # Add to combined binary mask (OR operation)
        binary_mask = np.logical_or(binary_mask, mask_binary).astype(np.uint8)
    
    # Convert to 0-255 range (0 = no segmentation, 255 = segmentation found)
    binary_mask = binary_mask * 255
    
    return binary_mask


def draw_segmentations(image, results, alpha=0.3, conf_threshold=0.25):
    """
    Draw segmentations on image with random colors.
    
    Args:
        image: Input image (numpy array)
        results: YOLO results object
        alpha: Transparency factor for filled masks
        conf_threshold: Confidence threshold for filtering detections
    
    Returns:
        Image with drawn segmentations
    """
    if results.masks is None:
        return image
    
    height, width = image.shape[:2]
    
    # Create a single overlay for all segmentations
    overlay = np.zeros_like(image)
    
    # Get masks and boxes
    masks = results.masks.data.cpu().numpy()
    boxes = results.boxes.data.cpu().numpy()
    
    # Filter by confidence threshold
    valid_indices = boxes[:, 4] >= conf_threshold
    masks = masks[valid_indices]
    boxes = boxes[valid_indices]
    
    # Draw all segmentations on the overlay
    for i, (mask, box) in enumerate(zip(masks, boxes)):
        # Resize mask to image dimensions
        mask_resized = cv2.resize(mask, (width, height))
        
        # Create binary mask
        binary_mask = (mask_resized > 0.5).astype(np.uint8)
        
        # Generate random color for this segmentation
        color = generate_random_color()
        
        # Draw filled mask on overlay
        overlay[binary_mask == 1] = color
    
    # Apply transparency once at the end
    result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    
    # Draw outlines on the final result
    for i, (mask, box) in enumerate(zip(masks, boxes)):
        # Resize mask to image dimensions
        mask_resized = cv2.resize(mask, (width, height))
        
        # Create binary mask
        binary_mask = (mask_resized > 0.5).astype(np.uint8)
        
        # Generate the same random color (using same seed)
        random.seed(i + 42)  # Use consistent seed for same color
        color = generate_random_color()
        
        # Draw mask outline
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, color, 2)
    
    return result


def process_image(model, image_path, output_path, mask_output_path=None, conf_threshold=0.25, alpha=0.3):
    """
    Process a single image with the trained model.
    
    Args:
        model: Trained YOLO model
        image_path: Path to input image
        output_path: Path to save output image
        mask_output_path: Path to save binary mask (optional)
        conf_threshold: Confidence threshold
        alpha: Transparency factor
    
    Returns:
        Number of detections found
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not load image {image_path}")
        return 0
    
    # Run inference
    results = model(image, conf=conf_threshold)
    
    if len(results) == 0:
        print(f"No detections found in {os.path.basename(image_path)}")
        # Save original image if no detections
        cv2.imwrite(output_path, image)
        # Save empty binary mask if mask path is provided
        if mask_output_path:
            height, width = image.shape[:2]
            empty_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.imwrite(mask_output_path, empty_mask)
        return 0
    
    # Get the first result (assuming single image)
    result = results[0]
    
    # Draw segmentations
    visualized_image = draw_segmentations(image, result, alpha, conf_threshold)
    
    # Save result
    cv2.imwrite(output_path, visualized_image)
    
    # Create and save binary mask if mask path is provided
    if mask_output_path:
        binary_mask = create_binary_mask(image, result, conf_threshold)
        cv2.imwrite(mask_output_path, binary_mask)
    
    # Count detections
    num_detections = len(result.boxes) if result.boxes is not None else 0
    print(f"Processed {os.path.basename(image_path)}: {num_detections} detections")
    
    return num_detections


def process_folder(model, input_folder, output_folder, mask_output_folder=None, input_images_folder=None, conf_threshold=0.25, alpha=0.3):
    """
    Process all images in a folder.
    
    Args:
        model: Trained YOLO model
        input_folder: Path to input folder
        output_folder: Path to output folder
        mask_output_folder: Path to binary mask output folder (optional)
        input_images_folder: Path to save input images (optional)
        conf_threshold: Confidence threshold
        alpha: Transparency factor
    
    Returns:
        Total number of detections across all images
    """
    # Create output directories
    os.makedirs(output_folder, exist_ok=True)
    if mask_output_folder:
        os.makedirs(mask_output_folder, exist_ok=True)
    if input_images_folder:
        os.makedirs(input_images_folder, exist_ok=True)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_files = []
    
    for file in os.listdir(input_folder):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
    
    if not image_files:
        print(f"No image files found in {input_folder}")
        return 0
    
    print(f"Found {len(image_files)} images in {input_folder}")
    print(f"Processing with confidence threshold: {conf_threshold}")
    print(f"Transparency factor: {alpha}")
    if mask_output_folder:
        print(f"Binary masks will be saved to: {mask_output_folder}")
    if input_images_folder:
        print(f"Input images will be saved to: {input_images_folder}")
    
    total_detections = 0
    
    for image_file in image_files:
        input_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, f"segmented_{image_file}")
        
        # Set mask output path if mask folder is provided
        mask_output_path = None
        if mask_output_folder:
            # Get filename without extension and add .png for binary masks
            base_name = os.path.splitext(image_file)[0]
            # Remove existing _mask suffix if present to avoid double _mask_mask
            if base_name.endswith('_mask'):
                base_name = base_name[:-5]  # Remove '_mask' suffix
            mask_output_path = os.path.join(mask_output_folder, f"{base_name}_mask.png")
        
        # Save input image if input images folder is provided
        if input_images_folder:
            input_image_path = os.path.join(input_images_folder, f"input_{image_file}")
            # Copy the input image to the input images folder
            import shutil
            shutil.copy2(input_path, input_image_path)
        
        num_detections = process_image(model, input_path, output_path, mask_output_path, conf_threshold, alpha)
        total_detections += num_detections
    
    print(f"\nProcessing complete!")
    print(f"Total images processed: {len(image_files)}")
    print(f"Total detections: {total_detections}")
    print(f"Segmented images saved to: {output_folder}")
    if mask_output_folder:
        print(f"Binary masks saved to: {mask_output_folder}")
    if input_images_folder:
        print(f"Input images saved to: {input_images_folder}")
    
    return total_detections


def main():
    parser = argparse.ArgumentParser(description='Run YOLOv11-seg inference on folder of images')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model weights (.pt file)')
    parser.add_argument('--input_folder', type=str, required=True,
                        help='Path to input folder containing images')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='Path to output folder for segmented images')
    parser.add_argument('--mask_output_folder', type=str, default=None,
                        help='Path to output folder for binary masks (optional)')
    parser.add_argument('--input_images_folder', type=str, default=None,
                        help='Path to output folder for input images (optional)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold (default: 0.25)')
    parser.add_argument('--alpha', type=float, default=0.3,
                        help='Transparency factor for masks (0.0-1.0, default: 0.3)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size for inference (default: 640)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda, 0, 1, etc.)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible colors')
    
    args = parser.parse_args()
    
    # Set random seed for reproducible colors
    random.seed(args.seed)
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    # Check if input folder exists
    if not os.path.exists(args.input_folder):
        print(f"Error: Input folder not found: {args.input_folder}")
        return
    
    print(f"🚀 Starting inference...")
    print(f"Model: {args.model}")
    print(f"Input folder: {args.input_folder}")
    print(f"Output folder: {args.output_folder}")
    if args.mask_output_folder:
        print(f"Binary mask output folder: {args.mask_output_folder}")
    print(f"Confidence threshold: {args.conf}")
    print(f"Image size: {args.imgsz}")
    print(f"Device: {args.device}")
    
    # Load model
    print(f"\n📦 Loading model...")
    model = YOLO(args.model)
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if hasattr(model, 'device') else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Process folder
    print(f"\n🖼️ Processing images...")
    total_detections = process_folder(
        model, 
        args.input_folder, 
        args.output_folder, 
        args.mask_output_folder,
        args.input_images_folder,
        args.conf, 
        args.alpha
    )
    
    print(f"\n✅ Inference complete!")
    print(f"Total detections found: {total_detections}")
    print(f"Segmented images saved to: {args.output_folder}")
    if args.mask_output_folder:
        print(f"Binary masks saved to: {args.mask_output_folder}")
    if args.input_images_folder:
        print(f"Input images saved to: {args.input_images_folder}")


if __name__ == "__main__":
    main()
