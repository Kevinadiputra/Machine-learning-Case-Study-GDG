"""
Utility functions for visualization and metrics calculation
"""

import cv2
import numpy as np
from typing import Dict, List


def visualize_results(image, results, show_labels=True, show_conf=True):
    """
    Visualize segmentation results on image
    
    Args:
        image (np.ndarray): Input image in RGB
        results: YOLO results object
        show_labels (bool): Whether to show class labels
        show_conf (bool): Whether to show confidence scores
    
    Returns:
        np.ndarray: Image with visualized results
    """
    # Create a copy of the image
    vis_image = image.copy()
    
    # Class names
    class_names = {
        0: 'body',
        1: 'cord',
        2: 'paru kanan',
        3: 'paru kiri'
    }
    
    # Colors for each class (BGR format for cv2)
    colors = {
        0: (255, 0, 0),      # body - red
        1: (0, 255, 0),      # cord - green
        2: (0, 0, 255),      # paru kanan - blue
        3: (255, 255, 0)     # paru kiri - yellow
    }
    
    # Check if there are any detections
    if results.masks is None or len(results.masks) == 0:
        return vis_image
    
    # Draw masks and boxes
    for idx, (mask, box) in enumerate(zip(results.masks.data, results.boxes)):
        # Get class ID and confidence
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        
        # Get color for this class
        color = colors.get(class_id, (255, 255, 255))
        
        # Convert mask to numpy array
        mask_array = mask.cpu().numpy()
        
        # Resize mask to image size
        mask_resized = cv2.resize(mask_array, (image.shape[1], image.shape[0]))
        mask_binary = (mask_resized > 0.5).astype(np.uint8)
        
        # Create colored mask
        colored_mask = np.zeros_like(vis_image)
        colored_mask[mask_binary == 1] = color
        
        # Blend mask with image
        vis_image = cv2.addWeighted(vis_image, 0.7, colored_mask, 0.3, 0)
        
        # Draw contours
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_image, contours, -1, color, 2)
        
        # Draw bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        
        # Add label and confidence
        if show_labels or show_conf:
            label_text = ""
            if show_labels:
                label_text += class_names.get(class_id, 'unknown')
            if show_conf:
                if label_text:
                    label_text += f" {confidence:.2f}"
                else:
                    label_text += f"{confidence:.2f}"
            
            # Draw label background
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                vis_image,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                vis_image,
                label_text,
                (x1, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
    
    return vis_image


def calculate_metrics(results, inference_time):
    """
    Calculate metrics from results
    
    Args:
        results: YOLO results object
        inference_time (float): Time taken for inference
    
    Returns:
        dict: Dictionary containing metrics
    """
    metrics = {
        'total_detections': 0,
        'avg_confidence': 0.0,
        'inference_time': inference_time,
        'detections': []
    }
    
    # Class names
    class_names = {
        0: 'body',
        1: 'cord',
        2: 'paru kanan',
        3: 'paru kiri'
    }
    
    # Check if there are detections
    if results.boxes is None or len(results.boxes) == 0:
        return metrics
    
    # Calculate metrics
    total_conf = 0.0
    
    for idx, box in enumerate(results.boxes):
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        bbox = box.xyxy[0].cpu().numpy().tolist()
        
        detection = {
            'class_id': class_id,
            'class_name': class_names.get(class_id, 'unknown'),
            'confidence': confidence,
            'bbox': bbox
        }
        
        metrics['detections'].append(detection)
        total_conf += confidence
    
    metrics['total_detections'] = len(results.boxes)
    metrics['avg_confidence'] = total_conf / len(results.boxes) if len(results.boxes) > 0 else 0.0
    
    return metrics


def create_comparison_view(original, segmented):
    """
    Create side-by-side comparison view
    
    Args:
        original (np.ndarray): Original image
        segmented (np.ndarray): Segmented image
    
    Returns:
        np.ndarray: Combined comparison image
    """
    # Ensure both images have the same height
    if original.shape[0] != segmented.shape[0]:
        # Resize to match heights
        target_height = min(original.shape[0], segmented.shape[0])
        original = cv2.resize(original, (int(original.shape[1] * target_height / original.shape[0]), target_height))
        segmented = cv2.resize(segmented, (int(segmented.shape[1] * target_height / segmented.shape[0]), target_height))
    
    # Concatenate horizontally
    comparison = np.hstack([original, segmented])
    
    return comparison


def save_results(image, results, output_path):
    """
    Save visualized results to file
    
    Args:
        image (np.ndarray): Original image
        results: YOLO results object
        output_path (str): Path to save the result
    """
    vis_image = visualize_results(image, results)
    
    # Convert RGB to BGR for cv2
    vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
    
    # Save image
    cv2.imwrite(output_path, vis_image_bgr)
    
    print(f"Results saved to: {output_path}")


def get_color_palette():
    """
    Get color palette for classes
    
    Returns:
        dict: Dictionary mapping class IDs to RGB colors
    """
    return {
        0: (255, 0, 0),      # body - red
        1: (0, 255, 0),      # cord - green
        2: (0, 0, 255),      # paru kanan - blue
        3: (255, 255, 0)     # paru kiri - yellow
    }


def format_results_for_display(results):
    """
    Format results for display in UI
    
    Args:
        results: YOLO results object
    
    Returns:
        str: Formatted string with results
    """
    if results.boxes is None or len(results.boxes) == 0:
        return "No detections found"
    
    class_names = {
        0: 'body',
        1: 'cord',
        2: 'paru kanan',
        3: 'paru kiri'
    }
    
    output = f"Found {len(results.boxes)} objects:\n\n"
    
    for idx, box in enumerate(results.boxes):
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = class_names.get(class_id, 'unknown')
        
        output += f"{idx + 1}. {class_name} (confidence: {confidence:.2%})\n"
    
    return output
