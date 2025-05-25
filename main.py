import cv2
import mediapipe as mp
import numpy as np
import torch
import time
import math

def load_yolo_model():
    """Load YOLOv5 model from torch hub"""
    print("Loading YOLOv5 model...")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    model.conf = 0.4  # Confidence threshold
    return model

def calculate_pointing_direction(hand_landmarks):
    """Calculate pointing direction vector from wrist to index fingertip"""
    # Get wrist and index fingertip positions
    wrist = np.array([
        hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].x,
        hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].y
    ])
    
    index_tip = np.array([
        hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].x,
        hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y
    ])
    
    # Create direction vector (from wrist to fingertip)
    direction = index_tip - wrist
    direction = direction / np.linalg.norm(direction)  # Normalize to unit vector
    
    return wrist, index_tip, direction

def find_pointed_object(wrist, direction, detections, frame_shape):
    """Determine which object the hand is pointing at"""
    if len(detections.xyxy[0]) == 0:  # No objects detected
        return None
    
    closest_obj = None
    min_angle = float('inf')
    min_distance = float('inf')
    
    height, width = frame_shape[:2]
    wrist_abs = np.array([wrist[0] * width, wrist[1] * height])
    
    for det in detections.xyxy[0]:
        x1, y1, x2, y2, conf, cls = det.cpu().numpy()
        
        # Calculate object center
        obj_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        
        # Vector from wrist to object center
        to_obj = obj_center - wrist_abs
        
        # Calculate angle between direction vector and object vector
        if np.linalg.norm(to_obj) > 0:
            # Normalize to_obj vector
            to_obj_norm = to_obj / np.linalg.norm(to_obj)
            
            # Calculate cosine similarity between vectors (dot product of unit vectors)
            cos_angle = np.dot(direction, to_obj_norm)
            
            # Constrain to valid range for arccos
            cos_angle = max(-1, min(cos_angle, 1))
            
            # Calculate angle in degrees
            angle = math.degrees(math.acos(cos_angle))
            
            # Consider object if angle is small enough (within pointing cone)
            if angle < 30:  # 30 degree cone
                dist = np.linalg.norm(to_obj)
                # Prioritize objects that are closer and more in line with pointing direction
                score = angle * 0.1 + dist * 0.0001
                
                if score < min_angle:
                    min_angle = score
                    closest_obj = det.cpu().numpy()
    
    return closest_obj

def draw_pointing_line(frame, wrist, index_tip, frame_shape):
    """Draw line showing pointing direction"""
    height, width = frame_shape[:2]
    
    start_point = (int(wrist[0] * width), int(wrist[1] * height))
    end_point = (int(index_tip[0] * width), int(index_tip[1] * height))
    
    # Extend the line further in the pointing direction
    direction = np.array(end_point) - np.array(start_point)
    norm = np.linalg.norm(direction)
    if norm > 0:
        unit_direction = direction / norm
        extension_factor = 1000  # Extend line by this many pixels
        extended_end = np.array(end_point) + unit_direction * extension_factor
        extended_end = tuple(map(int, extended_end))
        
        cv2.line(frame, start_point, extended_end, (0, 0, 255), 2)  # Red line

def visualize_results(frame, hand_landmarks, detections, pointed_object):
    """Draw hand landmarks, pointing line, and object boxes on the frame"""
    height, width = frame.shape[:2]
    
    # 1. Draw hand landmarks first
    if hand_landmarks:
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(
            frame, 
            hand_landmarks, 
            mp.solutions.hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
        )
        
        # Get wrist and fingertip positions
        wrist, index_tip, direction = calculate_pointing_direction(hand_landmarks)
        
        # Draw pointing line
        draw_pointing_line(frame, wrist, index_tip, frame.shape)
    
    # 2. Draw all detected objects (dimmed if not pointed at)
    if detections is not None:
        for i, det in enumerate(detections.xyxy[0]):
            det = det.cpu().numpy()
            x1, y1, x2, y2, conf, cls = det
            
            # Check if this is the pointed object
            is_pointed = pointed_object is not None and np.array_equal(det, pointed_object)
            
            # Define box color and thickness based on whether object is pointed at
            if is_pointed:
                color = (0, 255, 0)  # Green for pointed object
                thickness = 3
                label = f"Pointed: {detections.names[int(cls)]}"
            else:
                color = (128, 128, 128)  # Gray for other objects
                thickness = 1
                label = f"{detections.names[int(cls)]}"
            
            # Draw bounding box
            cv2.rectangle(
                frame, 
                (int(x1), int(y1)), 
                (int(x2), int(y2)), 
                color, 
                thickness
            )
            
            # Add label
            cv2.putText(
                frame, 
                label, 
                (int(x1), int(y1) - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                color, 
                thickness
            )
    
    # 3. Add FPS info
    fps_text = f"FPS: {int(fps)}"
    cv2.putText(
        frame, 
        fps_text, 
        (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (0, 255, 255), 
        2
    )
    
    return frame

def main():
    global fps  # Make fps available for display
    fps = 0
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Initialize MediaPipe Hands
    print("Initializing MediaPipe Hands...")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    # Load YOLO model
    model = load_yolo_model()
    
    # FPS calculation variables
    prev_time = time.time()
    fps_alpha = 0.1  # Smoothing factor
    
    print("Starting gesture-based object detection. Press 'q' to quit.")
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe Hands
        hand_results = hands.process(rgb_frame)
        
        # Object detection with YOLOv5
        detections = model(rgb_frame)
        
        # Initialize pointed_object to None
        pointed_object = None
        
        # If hands detected, find pointing direction
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Calculate pointing direction
                wrist, index_tip, direction = calculate_pointing_direction(hand_landmarks)
                
                # Find object being pointed at
                pointed_object = find_pointed_object(wrist, direction, detections, frame.shape)
        
        # Visualize results
        frame = visualize_results(frame, hand_results.multi_hand_landmarks[0] if hand_results.multi_hand_landmarks else None, 
                                  detections, pointed_object)
        
        # Calculate and smooth FPS
        current_time = time.time()
        fps = fps_alpha * (1 / (current_time - prev_time)) + (1 - fps_alpha) * fps
        prev_time = current_time
        
        # Display the frame
        cv2.imshow('Gesture-Based Object Detection', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Application terminated.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have installed all required libraries:")
        print("pip install opencv-python mediapipe torch torchvision numpy")
