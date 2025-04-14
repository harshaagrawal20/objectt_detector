import cv2
import numpy as np
import time

def draw_bounding_boxes(frame, boxes, labels, scores):
    """
    Draw bounding boxes and labels on the video frame.

    Parameters:
    - frame: The video frame to annotate.
    - boxes: List of bounding box coordinates (x1, y1, x2, y2).
    - labels: List of labels corresponding to the detected objects.
    - scores: List of confidence scores for each detection.
    
    Returns:
    - Annotated frame with bounding boxes and labels.
    """
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        # Draw the bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Prepare the label with confidence score
        label_with_score = f"{label}: {score:.2f}"
        # Draw the label
        cv2.putText(frame, label_with_score, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def draw_predictions(frame, predictions, fps=None):
    """Draw bounding boxes and labels on the frame"""
    height, width = frame.shape[:2]
    annotated = frame.copy()
    
    # Draw predictions
    for i, pred in enumerate(predictions[:3]):  # Show top 3 predictions
        confidence = pred['confidence']
        category = pred['category']
        
        # Draw text with better visibility
        text = f"{category}: {confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        color = (0, 255, 0)  # Green
        
        # Get text size and position
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = 20
        text_y = 50 + (i * (text_height + 20))
        
        # Draw text background
        cv2.rectangle(annotated, 
                     (text_x - 10, text_y - text_height - 10),
                     (text_x + text_width + 10, text_y + 10),
                     (0, 0, 0),
                     -1)
        
        # Draw text
        cv2.putText(annotated, text, (text_x, text_y),
                    font, font_scale, color, thickness)
    
    # Add FPS counter
    if fps:
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(annotated, fps_text, (width - 200, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    return annotated

def display_annotated_video(video_source):
    """
    Display the annotated video in a window.

    Parameters:
    - video_source: The source of the video (webcam or file).
    """
    cap = cv2.VideoCapture(video_source)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Here you would call your model to get boxes, labels, and scores
        # For example:
        # boxes, labels, scores = model.predict(frame)
        
        # Annotate the frame (dummy values for demonstration)
        boxes = [(50, 50, 200, 200)]  # Example bounding box
        labels = ["Lightbulb"]  # Example label
        scores = [0.95]  # Example confidence score
        
        annotated_frame = draw_bounding_boxes(frame, boxes, labels, scores)
        
        cv2.imshow('Annotated Video', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()