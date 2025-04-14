import unittest
from src.utils.video import capture_video, save_annotated_frame
from src.utils.visualization import draw_bounding_boxes

class TestUtils(unittest.TestCase):

    def test_capture_video(self):
        # Test if the video capture function works correctly
        video_source = 0  # Assuming 0 is the default webcam
        cap = capture_video(video_source)
        self.assertIsNotNone(cap)
        cap.release()

    def test_save_annotated_frame(self):
        # Test if the annotated frame can be saved correctly
        frame = None  # Placeholder for an actual frame
        output_path = 'test_output.jpg'
        result = save_annotated_frame(frame, output_path)
        self.assertTrue(result)

    def test_draw_bounding_boxes(self):
        # Test if bounding boxes are drawn correctly
        frame = None  # Placeholder for an actual frame
        boxes = [(10, 10, 50, 50)]
        labels = ['Test Object']
        confidences = [0.9]
        annotated_frame = draw_bounding_boxes(frame, boxes, labels, confidences)
        self.assertIsNotNone(annotated_frame)

if __name__ == '__main__':
    unittest.main()