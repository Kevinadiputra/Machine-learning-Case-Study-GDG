"""
Inference module for Lung Segmentation
Handles model loading and prediction
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


class LungSegmentationModel:
    """
    Wrapper class for YOLOv8 segmentation model
    """
    
    def __init__(self, model_path):
        """
        Initialize the model
        
        Args:
            model_path (str): Path to the model weights file
        """
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model
        self.model = YOLO(str(self.model_path))
        
        # Class names
        self.class_names = {
            0: 'body',
            1: 'cord',
            2: 'paru kanan',
            3: 'paru kiri'
        }
        
        print(f"✅ Model loaded from: {self.model_path}")
    
    def predict(self, image, conf_threshold=0.25, iou_threshold=0.7):
        """
        Run inference on an image
        
        Args:
            image (np.ndarray): Input image in RGB format
            conf_threshold (float): Confidence threshold for detection
            iou_threshold (float): IoU threshold for NMS
        
        Returns:
            results: YOLO results object
        """
        # Run inference
        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        return results[0]  # Return first result
    
    def predict_batch(self, images, conf_threshold=0.25, iou_threshold=0.7):
        """
        Run inference on multiple images
        
        Args:
            images (list): List of images in RGB format
            conf_threshold (float): Confidence threshold
            iou_threshold (float): IoU threshold
        
        Returns:
            list: List of results
        """
        results = self.model.predict(
            source=images,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        return results
    
    def get_class_name(self, class_id):
        """
        Get class name from class ID
        
        Args:
            class_id (int): Class ID
        
        Returns:
            str: Class name
        """
        return self.class_names.get(int(class_id), 'unknown')


def load_model(model_path='models/best.pt'):
    """
    Load the segmentation model
    
    Args:
        model_path (str): Path to model weights
    
    Returns:
        LungSegmentationModel: Loaded model instance
    """
    try:
        model = LungSegmentationModel(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def run_inference(model, image, conf=0.25, iou=0.7):
    """
    Run inference on a single image
    
    Args:
        model (LungSegmentationModel): Model instance
        image (np.ndarray): Input image
        conf (float): Confidence threshold
        iou (float): IoU threshold
    
    Returns:
        results: Prediction results
    """
    if model is None:
        raise ValueError("Model not loaded")
    
    results = model.predict(image, conf_threshold=conf, iou_threshold=iou)
    return results
