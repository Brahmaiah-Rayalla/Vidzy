"""
Stage 3.2.9F - Motion Evaluator

Quantitative evaluation of motion in generated videos.

Author: Devin
Date: 2026-11-02
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
import json
from PIL import Image
import cv2


class MotionEvaluator:
    """Evaluate motion characteristics in generated videos."""
    
    def __init__(self):
        self.metrics = {}
        
    def load_video_frames(self, frames_dir: Path) -> List[np.ndarray]:
        """Load all frames from a video directory."""
        frames = []
        frame_files = sorted(frames_dir.glob("frame_*.png"))
        
        for frame_file in frame_files:
            frame = np.array(Image.open(frame_file))
            frames.append(frame)
        
        return frames
    
    def detect_object(self, frame: np.ndarray, threshold: float = 0.7) -> Dict[str, Any]:
        """
        Simple object detection using color-based segmentation.
        Detects the main colored object against a white background.
        """
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Background is white, object is darker
        # Create binary mask
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return {
                'detected': False,
                'center': None,
                'bbox': None,
                'area': 0
            }
        
        # Find largest contour (main object)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area < 100:  # Too small, likely noise
            return {
                'detected': False,
                'center': None,
                'bbox': None,
                'area': 0
            }
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        center_x = x + w // 2
        center_y = y + h // 2
        
        return {
            'detected': True,
            'center': (center_x, center_y),
            'bbox': (x, y, w, h),
            'area': area
        }
    
    def evaluate_object_tracking(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Track object across frames and compute trajectory metrics."""
        tracking_data = []
        
        for i, frame in enumerate(frames):
            detection = self.detect_object(frame)
            tracking_data.append({
                'frame': i,
                'detected': detection['detected'],
                'center': detection['center'],
                'bbox': detection['bbox'],
                'area': detection['area']
            })
        
        # Compute tracking metrics
        detected_frames = sum(1 for t in tracking_data if t['detected'])
        detection_rate = detected_frames / len(frames) if frames else 0
        
        # Extract trajectory (centers)
        centers = [t['center'] for t in tracking_data if t['detected']]
        
        if len(centers) < 2:
            return {
                'detection_rate': detection_rate,
                'detected_frames': detected_frames,
                'total_frames': len(frames),
                'trajectory': [],
                'displacements': [],
                'total_displacement': 0,
                'avg_displacement': 0
            }
        
        # Compute frame-to-frame displacements
        displacements = []
        for i in range(len(centers) - 1):
            dx = centers[i+1][0] - centers[i][0]
            dy = centers[i+1][1] - centers[i][1]
            displacement = np.sqrt(dx**2 + dy**2)
            displacements.append(displacement)
        
        total_displacement = sum(displacements)
        avg_displacement = total_displacement / len(displacements) if displacements else 0
        
        return {
            'detection_rate': detection_rate,
            'detected_frames': detected_frames,
            'total_frames': len(frames),
            'trajectory': centers,
            'displacements': displacements,
            'total_displacement': total_displacement,
            'avg_displacement': avg_displacement
        }
    
    def evaluate_motion_direction(self, tracking_data: Dict[str, Any], 
                                  expected_direction: str) -> Dict[str, Any]:
        """
        Evaluate whether motion matches expected direction.
        
        expected_direction: 'left', 'right', 'up', 'down', 'stationary'
        """
        trajectory = tracking_data['trajectory']
        
        if len(trajectory) < 2:
            return {
                'expected_direction': expected_direction,
                'direction_accuracy': 0.0,
                'dominant_direction': 'unknown',
                'direction_x': 0,
                'direction_y': 0
            }
        
        # Compute overall direction
        start = trajectory[0]
        end = trajectory[-1]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        
        # Determine dominant direction
        if abs(dx) > abs(dy):
            dominant_direction = 'right' if dx > 0 else 'left'
        else:
            dominant_direction = 'down' if dy > 0 else 'up'
        
        # Evaluate accuracy
        accuracy = 0.0
        if expected_direction == 'stationary':
            # Stationary: minimal displacement
            total_disp = tracking_data['total_displacement']
            accuracy = 1.0 if total_disp < 5 else max(0, 1.0 - total_disp / 50)
        elif dominant_direction == expected_direction:
            accuracy = 1.0
        elif expected_direction in ['left', 'right'] and dominant_direction in ['left', 'right']:
            # Horizontal direction, opposite
            accuracy = 0.0
        elif expected_direction in ['up', 'down'] and dominant_direction in ['up', 'down']:
            # Vertical direction, opposite
            accuracy = 0.0
        else:
            # Orthogonal direction
            accuracy = 0.5
        
        return {
            'expected_direction': expected_direction,
            'dominant_direction': dominant_direction,
            'direction_x': dx,
            'direction_y': dy,
            'direction_accuracy': accuracy
        }
    
    def evaluate_speed(self, tracking_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate speed characteristics."""
        displacements = tracking_data['displacements']
        
        if not displacements:
            return {
                'avg_pixel_displacement': 0,
                'max_pixel_displacement': 0,
                'speed_variance': 0
            }
        
        avg_disp = np.mean(displacements)
        max_disp = max(displacements)
        speed_variance = np.var(displacements)
        
        return {
            'avg_pixel_displacement': float(avg_disp),
            'max_pixel_displacement': float(max_disp),
            'speed_variance': float(speed_variance)
        }
    
    def evaluate_appearance(self, frames: List[np.ndarray], 
                           tracking_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate appearance consistency."""
        if not tracking_data['trajectory']:
            return {
                'color_consistency': 0,
                'size_variance': 0,
                'visibility_rate': 0
            }
        
        # Extract object regions
        colors = []
        sizes = []
        
        for i, frame in enumerate(frames):
            if tracking_data['trajectory'][i] is not None:
                detection = self.detect_object(frame)
                if detection['detected']:
                    # Extract color from object region
                    x, y, w, h = detection['bbox']
                    region = frame[y:y+h, x:x+w]
                    if region.size > 0:
                        mean_color = np.mean(region, axis=(0, 1))
                        colors.append(mean_color)
                        sizes.append(detection['area'])
        
        if not colors:
            return {
                'color_consistency': 0,
                'size_variance': 0,
                'visibility_rate': tracking_data['detection_rate']
            }
        
        # Color consistency (std of mean colors)
        color_array = np.array(colors)
        color_std = np.std(color_array, axis=0)
        color_consistency = 1.0 / (1.0 + np.mean(color_std))
        
        # Size variance
        size_variance = np.var(sizes) if len(sizes) > 1 else 0
        
        return {
            'color_consistency': float(color_consistency),
            'size_variance': float(size_variance),
            'visibility_rate': tracking_data['detection_rate']
        }
    
    def evaluate_temporal_quality(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Evaluate temporal smoothness and quality."""
        if len(frames) < 2:
            return {
                'frame_to_frame_variation': 0,
                'flicker_score': 0,
                'smoothness': 0
            }
        
        # Compute frame-to-frame differences
        differences = []
        for i in range(len(frames) - 1):
            diff = np.abs(frames[i+1].astype(float) - frames[i].astype(float))
            differences.append(np.mean(diff))
        
        avg_variation = np.mean(differences)
        
        # Flicker: high-frequency changes
        if len(differences) > 1:
            flicker = np.std(differences)
        else:
            flicker = 0
        
        # Smoothness: inverse of variation
        smoothness = 1.0 / (1.0 + avg_variation)
        
        return {
            'frame_to_frame_variation': float(avg_variation),
            'flicker_score': float(flicker),
            'smoothness': float(smoothness)
        }
    
    def evaluate_video(self, frames_dir: Path, expected_direction: str) -> Dict[str, Any]:
        """Complete video evaluation."""
        frames = self.load_video_frames(frames_dir)
        
        if not frames:
            return {
                'error': 'No frames found',
                'valid': False
            }
        
        # Object tracking
        tracking = self.evaluate_object_tracking(frames)
        
        # Motion direction
        direction = self.evaluate_motion_direction(tracking, expected_direction)
        
        # Speed
        speed = self.evaluate_speed(tracking)
        
        # Appearance
        appearance = self.evaluate_appearance(frames, tracking)
        
        # Temporal quality
        temporal = self.evaluate_temporal_quality(frames)
        
        return {
            'valid': True,
            'tracking': tracking,
            'direction': direction,
            'speed': speed,
            'appearance': appearance,
            'temporal': temporal
        }
