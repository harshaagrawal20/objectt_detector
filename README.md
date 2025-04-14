# Zero-Shot Object Detection with CLIP

Real-time object detection system using CLIP for detecting custom objects in video streams.

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

1. Create and activate conda environment:
```bash
conda create -n object_detection python=3.9
conda activate object_detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Test camera setup:
```bash
python test_camera.py
```

## Usage

Run with webcam:
```bash
python src/main.py
```

Run with video file:
```bash
python src/main.py --source "test_video/example.mp4" --skip-frames 3 --resize 0.4
```

### Controls
- `q`: Quit application
- `s`: Save screenshot

## Project Structure
```
object-detection-app/
├── src/
│   ├── main.py           # Main application
│   ├── model/           # Model implementation
│   └── utils/           # Helper utilities
├── data/
│   └── prompts.json     # Detection categories
├── test_video/         # Test videos
└── README.md
```

## How It Works

This application uses OpenAI's CLIP (Contrastive Language-Image Pre-training) model for zero-shot object detection. CLIP is trained on a wide variety of image-text pairs, allowing it to recognize objects it hasn't been specifically trained on. The application processes video frames in real-time, passing them through CLIP along with text prompts for desired object categories. CLIP compares the visual features with text descriptions to detect objects without needing category-specific training data.

## Technical Implementation & Challenges

The application leverages OpenAI's CLIP (Contrastive Language-Image Pre-training) model to perform zero-shot object detection. At its core, CLIP processes video frames alongside text prompts describing target objects, computing similarity scores between visual and textual features. The implementation uses PyTorch for model inference and OpenCV for video handling. Video frames are processed through a pipeline that includes frame resizing (to 0.4-0.5x original size) and frame skipping (processing every 2-3 frames) to maintain real-time performance. Text prompts are enhanced with contextual descriptions (e.g., "a clear photo of a {object}") to improve detection accuracy.

Major challenges included optimizing inference speed on CPU (initially 2-3 FPS), handling model loading efficiently (1.7GB model size), and reducing false positives in complex scenes. These were addressed through several optimizations: implementing frame skipping and resizing reduced processing overhead, model caching improved startup time, and refined prompt engineering with confidence thresholds (>0.20) reduced false detections. Future improvements could focus on ONNX runtime integration for faster inference, implementing object tracking across frames, and adding a web-based UI for remote monitoring. The current implementation achieves 4-5 FPS on CPU while maintaining reasonable detection accuracy for custom object categories.

## Technical Implementation

The system uses OpenAI's CLIP model for zero-shot object detection, enabling recognition of objects without traditional training data. Key features:

- Real-time video processing with OpenCV
- Zero-shot detection using CLIP
- Custom object category support
- FPS optimization through frame skipping
- Confidence-based detection filtering

### Challenges & Solutions

1. Performance Optimization
   - Initial FPS: 2-3 frames/second
   - Solution: Implemented frame skipping and resizing
   - Result: Achieved 4-5 FPS on CPU

2. Model Loading
   - Challenge: 1.7GB model size
   - Solution: Added model caching
   - Result: Faster subsequent startups

3. Detection Accuracy
   - Challenge: False positives
   - Solution: Enhanced prompts and confidence thresholds
   - Result: More reliable detections

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

[Link to demo video - Coming soon]