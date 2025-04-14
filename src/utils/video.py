import cv2
import os
import threading
import time

class VideoStream:
    def __init__(self, skip_frames=2, resize_factor=0.5):
        self.skip_frames = skip_frames
        self.resize_factor = resize_factor
        self.stream = None
        self.stopped = False
        self._thread = None
        self.frame = None
        self.lock = threading.Lock()

    def start(self, source=0):
        # Handle string path for video file
        if isinstance(source, str):
            # Convert relative path to absolute
            if not os.path.isabs(source):
                root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                source = os.path.join(root_dir, source)
            
            if not os.path.exists(source):
                raise RuntimeError(f"Video file not found: {source}")
        
        self.stream = cv2.VideoCapture(source)
        if not self.stream.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")
        
        # Read first frame to ensure camera is working
        ret, frame = self.stream.read()
        if not ret:
            raise RuntimeError("Could not read from video source")
            
        self.frame = frame
        self.stopped = False
        self._thread = threading.Thread(target=self.update, daemon=True)
        self._thread.start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.stream.read()
            if not ret:
                self.stop()
                break
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return None
            
            # Skip frames
            for _ in range(self.skip_frames):
                self.stream.read()
                
            # Resize frame
            frame = self.frame.copy()
            if self.resize_factor != 1.0:
                width = int(frame.shape[1] * self.resize_factor)
                height = int(frame.shape[0] * self.resize_factor)
                frame = cv2.resize(frame, (width, height))
            
            return frame

    def stop(self):
        self.stopped = True
        if self._thread is not None and self._thread != threading.current_thread():
            self._thread.join()
        if self.stream is not None:
            self.stream.release()