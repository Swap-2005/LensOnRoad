"""
Traffic Rules Violation Detection System
Detects vehicles, tracks them, and checks for helmet violations
"""
import cv2
import numpy as np
import time
import math
import os
from collections import deque
from helm import HelmetDetector  # Import the HelmetDetector class

# Initialize classifiers
print("Loading car cascade classifier...")
car_cascade = cv2.CascadeClassifier('cars.xml')
if car_cascade.empty():
    print("Error: Could not load car cascade classifier")
    exit()

print("Loading bike cascade classifier...")
bike_cascade = cv2.CascadeClassifier('motor-v4.xml')
if bike_cascade.empty():
    print("Error: Could not load bike cascade classifier")
    exit()

helmet_detector = None  # Will be initialized when needed

# Video settings
video_path = 'record.mkv'
print(f"Opening video file: {video_path}")
video = cv2.VideoCapture(video_path)
if not video.isOpened():
    print(f"Error: Could not open video {video_path}")
    print("Please make sure the video file exists and is a valid video format.")
    exit()

# Get video properties
fps = video.get(cv2.CAP_PROP_FPS)
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video properties: {width}x{height}, {fps} FPS, {frame_count} frames")

# Validate video properties
if fps <= 0 or fps > 120:  # Assuming a reasonable FPS range
    print(f"Warning: Unusual FPS value: {fps}. Setting to default 30 FPS.")
    fps = 30.0
    
if frame_count <= 0:
    print("Warning: Could not determine frame count. Video might be corrupted.")
    frame_count = 0

# Constants
WIDTH = 1280
HEIGHT = 720
OPTIMISE = 1  # Process every frame for better accuracy
SPEED_LIMIT = 40  # Speed limit in km/h (reduced for better detection)
HELMET_CHECK_INTERVAL = 15  # Check for helmet every N frames (increased frequency)
MIN_DETECTION_SIZE = (40, 40)  # Minimum object size for detection
SCALE_FACTOR = 1.05  # Scale factor for detection
MIN_NEIGHBORS = 3  # Minimum neighbors for detection

# Colors (BGR format)
COLOR_RED = (0, 0, 255)    # Violation
COLOR_GREEN = (0, 255, 0)  # Normal
COLOR_BLUE = (255, 0, 0)   # Information
COLOR_WHITE = (255, 255, 255)  # Text

# Vehicle tracking
class VehicleTracker:
    def __init__(self, bbox, frame, vehicle_type, vehicle_id):
        # Convert bbox to (x, y, w, h) format if needed
        if len(bbox) == 4:
            x, y, w, h = [int(v) for v in bbox]
            self.positions = [(x, y, w, h)]  # List of (x, y, w, h) positions
        else:
            self.positions = [tuple(map(int, bbox))]  # Try to convert if in different format
            
        self.vehicle_type = vehicle_type.lower()  # Ensure consistent casing
        self.id = vehicle_id
        self.speed = 0
        self.speeds = []  # Store speed history for smoothing
        self.frame_count = 0
        self.helmet_detected = False
        self.last_helmet_check = 0
        self.helmet_check_interval = HELMET_CHECK_INTERVAL
        self.missed_detections = 0  # Track consecutive missed detections
        self.active = True  # Track if tracker is still valid
        self.last_seen = 0  # Track when this vehicle was last seen

    def update(self, bbox, frame):
        if not self.active:
            return False
            
        # Convert bbox to (x, y, w, h) if needed
        if len(bbox) == 4:
            x, y, w, h = [int(v) for v in bbox]
            self.positions.append((x, y, w, h))
        else:
            self.positions.append(tuple(map(int, bbox)))
            
        self.frame_count += 1
        self.last_seen = self.frame_count
        self.missed_detections = 0  # Reset missed detections counter
        
        # Limit position history
        if len(self.positions) > 10:
            self.positions.pop(0)
            
        # Calculate speed based on position change
        if len(self.positions) > 1:
            prev_x, prev_y, prev_w, prev_h = self.positions[-2]
            curr_x, curr_y, curr_w, curr_h = self.positions[-1]
            
            # Calculate center points
            prev_center = (prev_x + prev_w//2, prev_y + prev_h//2)
            curr_center = (curr_x + curr_w//2, curr_y + curr_h//2)
            
            # Calculate distance moved (Euclidean distance in pixels)
            distance = math.sqrt((curr_center[0] - prev_center[0])**2 + 
                               (curr_center[1] - prev_center[1])**2)
            
            # Calculate speed in km/h (with calibration factor)
            # Assuming 1 meter = 100 pixels (this needs calibration for your setup)
            # and 1 frame = 1/30 second (for 30 FPS)
            calibration_factor = 0.1  # Adjust this based on your camera setup
            speed_kmh = distance * fps * 3.6 * calibration_factor / 100
            
            # Smooth the speed using moving average
            self.speeds.append(speed_kmh)
            if len(self.speeds) > 5:  # Keep last 5 speed readings
                self.speeds.pop(0)
                
            self.speed = sum(self.speeds) / len(self.speeds)  # Moving average
            
            return True
        return False

    def check_helmet(self, frame):
        # Only check for helmets on bikes and at specified intervals
        if (self.vehicle_type != 'bikes' or 
            self.frame_count - self.last_helmet_check < self.helmet_check_interval or
            not self.active):
            return False
            
        try:
            x, y, w, h = self.positions[-1]
            
            # Expand ROI for better helmet detection (focus on upper half of the bike)
            roi_y1 = max(0, y - h//2)
            roi_y2 = min(y + h//2, frame.shape[0])  # Only check upper half
            roi_x1 = max(0, x - w//4)
            roi_x2 = min(x + w + w//4, frame.shape[1])
            
            roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            
            if roi.size == 0 or roi.shape[0] < 20 or roi.shape[1] < 20:
                return False
                
            # Initialize helmet detector if not already done
            global helmet_detector
            if helmet_detector is None:
                try:
                    helmet_detector = HelmetDetector()
                    print("Helmet detector initialized successfully")
                except Exception as e:
                    print(f"Error initializing HelmetDetector: {e}")
                    return False
            
            # Check for helmet
            _, has_helmet = helmet_detector.process_frame(roi, draw=False)
            self.helmet_detected = has_helmet
            self.last_helmet_check = self.frame_count
            
            # Debug: Draw ROI for helmet detection
            debug = False
            if debug:
                cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 0), 2)
                label = "Helmet Check"
                cv2.putText(frame, label, (roi_x1, roi_y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            return has_helmet
            
        except Exception as e:
            print(f"Error in helmet detection: {e}")
            return False

def draw_vehicle_info(image, x, y, w, h, vehicle_id, vehicle_type, speed=None, has_helmet=None):
    """
    Draw vehicle information on the image.
    
    Args:
        image: Image to draw on
        x, y, w, h: Bounding box coordinates
        vehicle_id: Unique ID for the vehicle
        vehicle_type: Type of vehicle ('cars' or 'bikes')
        speed: Current speed in km/h
        has_helmet: Whether the rider is wearing a helmet (for bikes)
    """
    # Draw bounding box
    color = (0, 255, 0)  # Default green
    
    # Change color based on violations
    if speed is not None and speed > SPEED_LIMIT:
        color = (0, 0, 255)  # Red for speeding
    
    if vehicle_type == 'bikes' and has_helmet is False:
        color = (0, 0, 255)  # Red for no helmet
    
    cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
    
    # Prepare text
    text = f"ID: {vehicle_id} {vehicle_type}"
    if speed is not None:
        text += f" {speed:.1f}km/h"
    if vehicle_type == 'bikes':
        helmet_text = "Helmet: Yes" if has_helmet else "Helmet: No"
        text = f"{text} | {helmet_text}"
    
    # Draw text background
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(image, (x, y - 20), (x + text_width, y), color, -1)
    
    # Draw text
    cv2.putText(image, text, (x, y - 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def estimate_speed(location1, location2, fps):
    """
    Estimate the speed of a vehicle between two points.
    
    Args:
        location1: First position [x, y, w, h]
        location2: Second position [x, y, w, h]
        fps: Frames per second
        
    Returns:
        float: Estimated speed in km/h
    """
    # Calculate the center points
    x1, y1, w1, h1 = location1
    x2, y2, w2, h2 = location2
    
    center1 = (x1 + w1//2, y1 + h1//2)
    center2 = (x2 + w2//2, y2 + h2//2)
    
    # Calculate distance in pixels
    distance_px = math.sqrt((center2[0] - center1[0])**2 + (center2[1] - center1[1])**2)
    
    # Convert to meters (assuming 1 meter = 100 pixels)
    distance_m = distance_px / 100.0
    
    # Calculate time in seconds
    time_sec = 1.0 / fps if fps > 0 else 1.0
    
    # Calculate speed in m/s
    speed_mps = distance_m / time_sec
    
    # Convert to km/h
    speed_kmh = speed_mps * 3.6
    
    return speed_kmh

def detect_vehicles(frame, cascade, vehicle_type):
    """
    Detect vehicles in the frame using the specified cascade classifier.
    
    Args:
        frame: Input frame (BGR format)
        cascade: Cascade classifier
        vehicle_type: Type of vehicle ('cars' or 'bikes')
        
    Returns:
        list: List of detected vehicles as (x, y, w, h, vehicle_type) tuples
    """
    try:
        # Convert to grayscale and equalize histogram for better detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        # Detect vehicles with optimized parameters
        vehicles = cascade.detectMultiScale(
            gray,
            scaleFactor=SCALE_FACTOR,
            minNeighbors=MIN_NEIGHBORS,
            minSize=MIN_DETECTION_SIZE,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Filter out small detections and return with vehicle type
        min_area = MIN_DETECTION_SIZE[0] * MIN_DETECTION_SIZE[1]
        return [
            (x, y, w, h, vehicle_type) 
            for (x, y, w, h) in vehicles 
            if w * h >= min_area * 0.8  # Allow slightly smaller detections
        ]
        
    except Exception as e:
        print(f"Error in vehicle detection ({vehicle_type}): {e}")
        return []

def trackMultipleObjects():
    """
    Main function to track multiple vehicles and detect traffic violations.
    Uses Haar cascades for detection and simple tracking without external model dependencies.
    """
    print("Starting vehicle tracking...")
    
    # Initialize variables
    frame_count = 0
    vehicle_trackers = []  # List to store active vehicle trackers
    next_vehicle_id = 1  # Counter for assigning unique IDs to vehicles
    
    # Performance metrics
    total_frames = 0
    processing_times = []
    
    # Create output window
    cv2.namedWindow("Traffic Monitoring", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Traffic Monitoring", WIDTH, HEIGHT)
    
    # Calculate delay based on desired FPS (30 FPS = ~33ms delay between frames)
    delay = int(1000 / 30)  # 30 FPS
    
    # Define detection parameters
    SCALE_FACTOR = 1.1
    MIN_NEIGHBORS = 3
    MIN_DETECTION_SIZE = (30, 30)
    
    # Define colors
    COLOR_GREEN = (0, 255, 0)
    COLOR_RED = (0, 0, 255)
    COLOR_BLUE = (255, 0, 0)
    COLOR_WHITE = (255, 255, 255)
    
    # Define speed limit
    SPEED_LIMIT = 30  # km/h
    
    print("Starting video playback. Press 'q' to quit...")
    
    # Main processing loop
    while True:
        try:
            start_time = time.time()
            
            # Read frame
            ret, frame = video.read()
            if not ret:
                print("End of video")
                break
                
            total_frames += 1
                
            # Resize frame for consistent processing
            frame = cv2.resize(frame, (WIDTH, HEIGHT))
            frame_count += 1
            
            # Create a copy for drawing
            display_frame = frame.copy()
        
            # Skip frames for better performance
            if frame_count % OPTIMISE != 0:
                # Still process some frames for display to maintain responsiveness
                if frame_count % (OPTIMISE * 2) == 0:
                    cv2.imshow('Traffic Monitoring', frame)
                    
                # Add a small delay to maintain playback speed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("User requested to quit")
                    break
                    
                continue
                
            # Calculate and display FPS
            process_time = time.time() - start_time
            current_fps = 1.0 / process_time if process_time > 0 else 0
            
            # Display FPS on frame
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_GREEN, 2)
            
            # Convert to grayscale for detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect vehicles in the current frame
            cars = detect_vehicles(frame, car_cascade, 'cars')
            bikes = detect_vehicles(frame, bike_cascade, 'bikes')
            
            # Update existing trackers
            active_trackers = []
            
            for tracker in vehicle_trackers:
                x, y, w, h = tracker.positions[-1]
                
                # Check if this tracker is still valid (vehicle is still in frame)
                if (x < 0 or y < 0 or x + w > WIDTH or y + h > HEIGHT):
                    continue
                    
                # Update position based on detection
                found_match = False
                
                # Check for matches with new detections
                detections = cars if tracker.vehicle_type == 'cars' else bikes
                
                for (nx, ny, nw, nh, _) in detections:
                    # Calculate intersection over union
                    x1 = max(x, nx)
                    y1 = max(y, ny)
                    x2 = min(x + w, nx + nw)
                    y2 = min(y + h, ny + nh)
                    
                    if x2 > x1 and y2 > y1:  # If there is an overlap
                        # Update position
                        tracker.update((nx, ny, nw, nh), frame)
                        found_match = True
                        break
                
                # If no match found, keep the last known position
                if not found_match and len(tracker.positions) > 0:
                    tracker.update((x, y, w, h), frame)
                
                # Check for helmet if it's a bike
                has_helmet = False
                if tracker.vehicle_type == 'bikes':
                    has_helmet = tracker.check_helmet(frame)
                
                # Determine color based on violations
                color = COLOR_GREEN
                violations = []
                
                # Check for speed violation
                if tracker.speed > SPEED_LIMIT:
                    violations.append(f"SPEEDING: {tracker.speed:.1f} km/h")
                    color = COLOR_RED
                
                # Check for helmet violation (only for bikes)
                if tracker.vehicle_type == 'bikes' and not has_helmet:
                    violations.append("NO HELMET")
                    color = COLOR_RED
                
                # Draw vehicle information
                draw_vehicle_info(
                    frame, x, y, w, h, 
                    tracker.id, tracker.vehicle_type, 
                    tracker.speed, has_helmet
                )
                
                # Draw violation text if any
                for i, violation in enumerate(violations):
                    cv2.putText(
                        frame, 
                        violation, 
                        (x, y - 10 - (i * 20)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        color, 
                        2
                    )
            
            # Add new detections as trackers
            for (x, y, w, h, _) in cars:
                is_new = True
                for tracker in vehicle_trackers:
                    tx, ty, tw, th = tracker.positions[-1]
                    if (abs(x - tx) < w and abs(y - ty) < h):
                        is_new = False
                        break
                
                if is_new:
                    vehicle_trackers.append(VehicleTracker((x, y, w, h), frame, 'cars', next_vehicle_id))
                    next_vehicle_id += 1
            
            for (x, y, w, h, _) in bikes:
                is_new = True
                for tracker in vehicle_trackers:
                    tx, ty, tw, th = tracker.positions[-1]
                    if (abs(x - tx) < w and abs(y - ty) < h):
                        is_new = False
                        break
                
                if is_new:
                    vehicle_trackers.append(VehicleTracker((x, y, w, h), frame, 'bikes', next_vehicle_id))
                    next_vehicle_id += 1
            
            # Calculate and display FPS
            process_time = time.time() - start_time
            processing_times.append(process_time)
            if len(processing_times) > 30:  # Keep last 30 frames
                processing_times.pop(0)
                
            avg_fps = 1.0 / (sum(processing_times) / len(processing_times)) if processing_times else 0
            
            # Display FPS and vehicle count
            cv2.putText(
                frame, 
                f"FPS: {avg_fps:.1f} | Vehicles: {len(vehicle_trackers)}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                COLOR_WHITE, 
                2
            )
            
            # Display the resulting frame
            cv2.imshow('Traffic Monitoring', frame)
            
            # Calculate processing time and maintain consistent FPS
            process_time = time.time() - start_time
            wait_time = max(1, int((1000.0 / 30) - (process_time * 1000)))  # Target 30 FPS
            
            # Exit on 'q' press or window close
            key = cv2.waitKey(wait_time) & 0xFF
            if key == ord('q'):
                print("User requested to quit")
                break
                
            # Check if window was closed
            if cv2.getWindowProperty('Traffic Monitoring', cv2.WND_PROP_VISIBLE) < 1:
                print("Window closed by user")
                break
                
        except Exception as e:
            print(f"Error in main loop: {e}")
            continue
    
    # Clean up
    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
	trackMultipleObjects()
