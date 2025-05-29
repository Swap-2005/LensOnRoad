import cv2
import os
import numpy as np

class HelmetDetector:
    def __init__(self, cascade_path=None):
        """
        Initialize the helmet detector.
        
        Args:
            cascade_path: Path to the Haar cascade XML file for helmet detection.
                         If None, will look for 'haarcascade_helmet.xml' in the current directory.
        """
        self.cascade_path = cascade_path or 'haarcascade_helmet.xml'
        self.helmet_cascade = self._load_cascade()
        
    def _load_cascade(self):
        """Load the Haar cascade classifier from the specified path."""
        if os.path.exists(self.cascade_path):
            return cv2.CascadeClassifier(self.cascade_path)
        print(f"Warning: Helmet cascade not found at {self.cascade_path}")
        return None
    
    def detect_helmets(self, frame, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
        """
        Detect helmets in the given frame.
        
        Args:
            frame: Input image (BGR format)
            scale_factor: Parameter specifying how much the image size is reduced at each image scale
            min_neighbors: Parameter specifying how many neighbors each candidate rectangle should have
            min_size: Minimum possible object size
            
        Returns:
            list: List of bounding boxes in format (x, y, w, h)
        """
        if self.helmet_cascade is None:
            return []
            
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect helmets
        helmets = self.helmet_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return helmets
    
    def process_frame(self, frame, draw=True):
        """
        Process a frame to detect helmets.
        
        Args:
            frame: Input frame (BGR format)
            draw: Whether to draw detections on the frame
            
        Returns:
            tuple: (frame, has_helmet) where has_helmet is True if any helmets were detected
        """
        helmets = self.detect_helmets(frame)
        has_helmet = len(helmets) > 0
        
        if draw and has_helmet:
            for (x, y, w, h) in helmets:
                # Draw rectangle
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Draw label
                label = "Helmet"
                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x, y - 20), (x + label_width, y), (0, 255, 0), -1)
                cv2.putText(frame, label, (x, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return frame, has_helmet

# For backward compatibility
def detect(frame):
    """
    Detect helmets in the given frame (legacy function).
    
    Args:
        frame: Input image (BGR format)
        
    Returns:
        int: 1 if helmet is detected, 0 otherwise
    """
    detector = HelmetDetector()
    _, has_helmet = detector.process_frame(frame, draw=False)
    return 1 if has_helmet else 0
