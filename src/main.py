import cv2
import torch
from PIL import Image
import time
from model.zero_shot import ZeroShotDetector
from utils.video import VideoStream
from utils.visualization import draw_predictions
import json
import argparse
import os

class ObjectDetectionApp:
    def __init__(self, config_path="data/prompts.json"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\nInitializing object detection on {self.device}...")
        
        # Set up model cache
        cache_dir = os.path.join(os.getcwd(), "model_cache")
        self.model = ZeroShotDetector(device=self.device, cache_dir=cache_dir)
        self.video_stream = VideoStream()
        
        # Load custom object categories
        with open(config_path, 'r') as f:
            self.categories = json.load(f)['categories']
        
    def run(self, source=0, skip_frames=2, resize_factor=0.5):
        try:
            self.video_stream.start(source)
            start_time = time.time()
            frame_count = 0
            fps = 0.0
            
            print(f"Using device: {self.device}")
            print(f"Looking for categories: {self.categories}")
            print("Press 'q' to quit, 's' to save a screenshot")
            
            while True:
                frame = self.video_stream.read()
                if frame is None:
                    break
                
                frame_count += 1
                current_time = time.time()
                fps = frame_count / (current_time - start_time)
                
                # Process frames
                if frame_count % skip_frames == 0:
                    # Resize frame for faster processing
                    height, width = frame.shape[:2]
                    new_width = int(width * resize_factor)
                    new_height = int(height * resize_factor)
                    frame = cv2.resize(frame, (new_width, new_height))
                    
                    # Convert and predict
                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    predictions = self.model.predict(image, self.categories, confidence_threshold=0.20)
                    
                    if predictions:
                        print(f"\nDetections (FPS: {fps:.1f}):")
                        for pred in predictions[:3]:
                            print(f"{pred['category']}: {pred['confidence']:.2f}")
                    
                    # Draw predictions
                    annotated_frame = draw_predictions(frame, predictions, fps)
                    cv2.imshow('Object Detection', annotated_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"detection_{timestamp}.jpg", frame)
                    print(f"\nSaved screenshot: detection_{timestamp}.jpg")
                    
        except KeyboardInterrupt:
            print("\nStopping detection...")
        finally:
            self.video_stream.stop()
            cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default=0,
                      help="Video source (0 for webcam or path to video file)")
    parser.add_argument("--skip-frames", type=int, default=2,
                      help="Number of frames to skip")
    parser.add_argument("--resize", type=float, default=0.5,
                      help="Resize factor")
    args = parser.parse_args()

    try:
        # Convert video path to absolute path
        source = args.source
        if isinstance(source, str) and source != "0":
            source = os.path.abspath(os.path.join(os.getcwd(), source))
            if not os.path.exists(source):
                raise FileNotFoundError(f"Video file not found: {source}")
            print(f"Using video file: {source}")

        app = ObjectDetectionApp()
        app.run(source=source, skip_frames=args.skip_frames, resize_factor=args.resize)
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()