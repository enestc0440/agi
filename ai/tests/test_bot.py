import unittest
from src.bot import ChatBot

class TestBot(unittest.TestCase):
    def setUp(self):
        self.bot = ChatBot(model_path="checkpoints/chatbot.pkl")
    
    def test_empty_input(self):
        response = self.bot.respond("")
        self.assertEqual(response, "Lütfen geçerli bir giriş yapın.")

if __name__ == "__main__":
    unittest.main()