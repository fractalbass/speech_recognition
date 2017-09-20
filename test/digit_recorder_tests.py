import unittest
from digit_recognizer import digit_recognizer


class TestDigitRecognizer(unittest.TestCase):
    def setUp(self):
        pass

    def test_process_directory(self):
        dr = digit_recognizer()
        features = dr.process_directory()
        #self.assertEqual(len(features), 10)

    def test_get_filename(self):
        dr = digit_recognizer()
        self.assertEqual(dr.get_label("1_miles_blah.wav"), "1")

if __name__ == '__main__':
    unittest.main()
