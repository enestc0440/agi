from transformers import AutoTokenizer, pipeline
from langdetect import detect
import speech_recognition as sr

class TextProcessor:
    def __init__(self, model_name="xlm-roberta-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.intent_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="savasy/bert-base-turkish-sentiment")
        self.recognizer = sr.Recognizer()
    
    def detect_language(self, text):
        try:
            return detect(text)
        except:
            return "tr"
    
    def tokenize(self, text, max_length=128):
        return self.tokenizer.encode(text, add_special_tokens=False, max_length=max_length, truncation=True)
    
    def detokenize(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
    
    def classify_intent(self, text):
        candidate_labels = ["selamlaşma", "soru", "talep", "sohbet", "hava durumu", "haber"]
        result = self.intent_classifier(text, candidate_labels)
        return result["labels"][0], result["scores"][0]
    
    def analyze_sentiment(self, text):
        if self.detect_language(text) == "tr":
            return self.sentiment_analyzer(text)[0]["label"]
        return "neutral"
    
    def speech_to_text(self):
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source)
            print("Dinliyorum...")
            audio = self.recognizer.listen(source)
            try:
                text = self.recognizer.recognize_google(audio, language="tr-TR")
                return text
            except sr.UnknownValueError:
                return "Anlamadım, tekrar edebilir misiniz?"
            except sr.RequestError:
                return "Ses servisi şu anda kullanılamıyor."