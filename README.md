# Object Detection App

## Overview
The Object Detection App is a Python application that utilizes a zero-shot vision model to recognize and annotate objects from real-time or pre-recorded video. The application is designed to test the model's generalization ability by using custom object categories that are not part of the common COCO dataset.

## Features
- Accepts input from a webcam or a local video file.
- Uses a list of custom object categories as text prompts.
- Processes each frame through a zero-shot model.
- Displays annotated video with bounding boxes, labels, and confidence scores.
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
1. Clone the repository:
   ```
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```
   cd object-detection-app
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Model Usage
To run the application, execute the following command:
```
python src/main.py
```
Make sure to have your webcam connected or specify a local video file in the code.

## Challenges and Improvements
This project aims to explore the capabilities of zero-shot models in recognizing non-COCO objects. Future improvements could include enhancing the model's accuracy, expanding the list of custom object categories, and optimizing performance for real-time applications.

## Video Demonstration
A video demonstration of the application will be provided to showcase its functionality and performance.