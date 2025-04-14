import unittest
from src.model.zero_shot import ZeroShotModel

class TestZeroShotModel(unittest.TestCase):

    def setUp(self):
        self.model = ZeroShotModel()
        self.model.load_model('path/to/model')  # Adjust the path as necessary

    def test_load_model(self):
        self.assertIsNotNone(self.model.model, "Model should be loaded successfully.")

    def test_process_frame(self):
        test_frame = ...  # Load or create a test frame
        predictions = self.model.process_frame(test_frame)
        self.assertIsInstance(predictions, list, "Predictions should be a list.")
        self.assertGreater(len(predictions), 0, "Predictions should not be empty.")

if __name__ == '__main__':
    unittest.main()