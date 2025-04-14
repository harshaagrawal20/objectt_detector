# Zero-Shot Object Detection with CLIP

Real-time object detection using OpenAI's CLIP model for zero-shot classification of custom objects.

## Overview
The Object Detection App is a Python application that utilizes a zero-shot vision model to recognize and annotate objects from real-time or pre-recorded video. The application is designed to test the model's generalization ability by using custom object categories that are not part of the common COCO dataset.

## Features
- Real-time object detection from webcam or video files
- Zero-shot detection of custom object categories
- FPS counter and performance metrics
- Screenshot capability
- Support for both CPU and CUDA acceleration
- Ensures that none of the detected objects are from the COCO dataset.

## Technical Requirements
- Implemented in Python.
- Uses OpenCV for video input.
- Utilizes PyTorch and pre-trained zero-shot models like CLIP or OWL-ViT.
- Code is clean, modular, and well-commented.
- Results can be displayed in a live window or printed to the console.

## Bonus Features
- Live prompt editing for changing detection classes during runtime.
- Frame rate optimization (>=10 FPS).
- Logging predictions to a file (JSON or CSV).
- Using ONNX or TorchScript to accelerate inference.
- Visualizing detections with a minimal dashboard or UI.

## Setup Instructions

1. Create a conda environment:
```bash
conda create -n object_detection python=3.9
conda activate object_detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the CLIP model (happens automatically on first run)

## Usage

Run with webcam:
```bash
python src/main.py
```

Run with video file:
```bash
python src/main.py --source "path/to/video.mp4" --skip-frames 3 --resize 0.4
```

Controls:
- Press 'q' to quit
- Press 's' to save screenshot

## Project Structure
```
object-detection-app/
├── src/
│   ├── main.py
│   ├── model/
│   │   └── zero_shot.py
│   └── utils/
│       ├── video.py
│       └── visualization.py
├── data/
│   └── prompts.json
└── test_video/
    └── example.mp4
```

## How It Works

This application uses OpenAI's CLIP (Contrastive Language-Image Pre-training) model for zero-shot object detection. CLIP is trained on a wide variety of image-text pairs, allowing it to recognize objects it hasn't been specifically trained on. The application processes video frames in real-time, passing them through CLIP along with text prompts for desired object categories. CLIP compares the visual features with text descriptions to detect objects without needing category-specific training data.

## Challenges and Solutions

1. Performance Optimization
   - Challenge: Initial FPS was very low (2-3 FPS)
   - Solution: Implemented frame skipping and resizing for better performance

2. Model Loading
   - Challenge: Slow model loading and high memory usage
   - Solution: Added model caching and optimized batch processing

3. Detection Accuracy
   - Challenge: False positives in complex scenes
   - Solution: Improved text prompts and added confidence thresholds

## Future Improvements

1. Technical Enhancements:
   - ONNX runtime integration for faster inference
   - GPU optimization for real-time processing
   - Multi-threading for parallel frame processing

2. Features to Add:
   - Dynamic category addition during runtime
   - Tracking of detected objects across frames
   - Web interface for remote monitoring
   - Detection logging and analytics

## Challenges and Improvements
This project aims to explore the capabilities of zero-shot models in recognizing non-COCO objects. Future improvements could include enhancing the model's accuracy, expanding the list of custom object categories, and optimizing performance for real-time applications.

## Video Demonstration
A video demonstration of the application will be provided to showcase its functionality and performance.