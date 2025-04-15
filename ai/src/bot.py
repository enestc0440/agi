import jax
import jax.numpy as jnp
import numpy as np
from gtts import gTTS
from .processor import TextProcessor
from .integrations import get_weather, get_news
from .cache import Cache
from .inference import InferenceRunner

class ChatBot:
    def __init__(self, model_path='checkpoints/chatbot.pkl', model_name="xlm-roberta-base"):
        self.processor = TextProcessor(model_name)
        self.memory = []
        self.memory_size = 5
        self.cache = Cache()
        self.inference_runner = InferenceRunner(
            name="UltraLightAI",
            runner=None,  # ModelRunner sonra bağlanacak
            load=model_path,
            tokenizer_path="/tmp/xai_data/tokenizer.model"
        )
        self.inference_runner.initialize()
    
    def respond(self, input_text, use_speech=False):
        if not input_text.strip():
            return "Lütfen geçerli bir giriş yapın."
        
        cached_response = self.cache.get(input_text)
        if cached_response:
            return cached_response
        
        lang = self.processor.detect_language(input_text)
        intent, confidence = self.processor.classify_intent(input_text)
        sentiment = self.processor.analyze_sentiment(input_text)
        
        self.memory.append(input_text)
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        
        if intent == "hava durumu" and confidence > 0.7:
            response = get_weather("Istanbul")
        elif intent == "haber" and confidence > 0.7:
            response = get_news()
        else:
            response = self.inference_runner.run_prompt(input_text, max_len=50, temperature=1.0)
        
        if sentiment == "negative":
            response += " Üzülme, her şey düzelir! 😊"
        
        self.cache.set(input_text, response)
        
        if use_speech:
            tts = gTTS(text=response, lang=lang)
            tts.save("response.mp3")
        
        self.memory.append(response)
        return response