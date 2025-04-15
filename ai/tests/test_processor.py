import unittest
from src.processor import TextProcessor

class TestProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = TextProcessor()
    
    def test_tokenize(self):
        tokens = self.processor.tokenize("Merhaba")
        self.assertTrue(len(tokens) > 0)
    
    def test_detect_language(self):
        lang = self.processor.detect_language("Merhaba")
        self.assertEqual(lang, "tr")

if __name__ == "__main__":
    unittest.main()