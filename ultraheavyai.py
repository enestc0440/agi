# UltraHeavyAI: Galaksi Fatihi Sürüm
# Tüm çılgın özellikler entegre edildi: Kod çalıştırma, sesli yanıt, bilgi tabanı, kişiselleştirme, hata analiz, transformer, ASCII sanat vs.

import os
import re
import random
import logging
import subprocess
import tempfile
from gtts import gTTS
import joblib
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
from transformers import AutoTokenizer
from colorama import Fore, Style, init

init()
logging.basicConfig(filename="ultraheavyai.log", level=logging.DEBUG)

# =============================== Sabitler ===============================
VOCAB_SIZE = 32000
MODEL_SIZE = 1024
NUM_LAYERS = 24
NUM_Q_HEADS = 32
NUM_KV_HEADS = 16
KEY_SIZE = 64
WIDENING = 4.0
MEMORY_LIMIT = 50

# =========================== Yardımcı Fonksiyonlar ===========================
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-ZğüşıöçĞÜŞİÖÇ0-9\s]", "", text)
    return text.lower().strip() if text else "boş"

def execute_code(code):
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            result = subprocess.run(["python", f.name], capture_output=True, text=True, timeout=5)
            os.unlink(f.name)
            return result.stdout or result.stderr
    except Exception as e:
        return f"Kod patladı kanka! 😅 Hata: {str(e)}"

def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang="tr")
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tts.save(f.name)
            os.system(f"start {f.name}")
        return "Sesli yanıt geldi! 🔊"
    except Exception as e:
        return f"Sesli yanıt patladı! 😅 Hata: {str(e)}"

def suggest_fix(error):
    if "NameError" in error:
        return "Değişken tanımlı değil kanka! `x = 5` gibi bir şey eksik olabilir."
    elif "SyntaxError" in error:
        return "Yazım hatası lan! 😅 Parantezleri, iki nokta üst üste'yi kontrol et."
    return "Bu hata değişik, ama çözeriz kanka! 💥"

def get_ascii_art():
    return """
🚀=====
| UltraHeavyAI |
| Galaksi Fatihi! |
🚀=====
"""

# =========================== Transformer Modeli ===========================
def hk_rms_norm(x):
    mean_squared = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(mean_squared + 1e-5)

class MultiHeadAttention(hk.Module):
    def __init__(self, q_heads, kv_heads, key_size, model_size):
        super().__init__()
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.key_size = key_size
        self.model_size = model_size

    def __call__(self, q, k, v):
        q_proj = self._proj(q, self.q_heads, self.key_size, 'q')
        k_proj = self._proj(k, self.kv_heads, self.key_size, 'k')
        v_proj = self._proj(v, self.kv_heads, self.key_size, 'v')
        attn_scores = jnp.einsum("bthd,bThd->bhtT", q_proj, k_proj) / jnp.sqrt(self.key_size)
        weights = jax.nn.softmax(attn_scores)
        attn = jnp.einsum("bhtT,bThd->bthd", weights, v_proj)
        final = hk.Linear(self.model_size)(attn.reshape(attn.shape[0], attn.shape[1], -1))
        return final

    def _proj(self, x, heads, size, name):
        y = hk.Linear(heads * size, name=name)(x)
        return y.reshape(x.shape[0], x.shape[1], heads, size)

class DenseBlock(hk.Module):
    def __init__(self, model_size, widening):
        super().__init__()
        self.size = model_size
        self.wide = widening

    def __call__(self, x):
        ffn_size = int(self.size * self.wide)
        x1 = hk.Linear(ffn_size)(x)
        x2 = hk.Linear(ffn_size)(x)
        x3 = hk.Linear(self.size)(jax.nn.gelu(x1) * x2)
        return x3

class DecoderLayer(hk.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        h = hk_rms_norm(x)
        h = MultiHeadAttention(NUM_Q_HEADS, NUM_KV_HEADS, KEY_SIZE, MODEL_SIZE)(h, h, h)
        h = hk_rms_norm(x + h)
        h2 = DenseBlock(MODEL_SIZE, WIDENING)(hk_rms_norm(h))
        return hk_rms_norm(h + h2)

class Transformer(hk.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, tokens):
        embed = hk.Embed(VOCAB_SIZE, MODEL_SIZE)
        x = embed(tokens).astype(jnp.bfloat16)
        for _ in range(NUM_LAYERS):
            x = DecoderLayer()(x)
        logits = jnp.dot(x, embed.embeddings.T.astype(x.dtype))
        return logits

# =========================== Ana Bot Sınıfı ===========================
class UltraHeavyBot:
    def __init__(self, params, tokenizer):
        self.params = params
        self.tokenizer = tokenizer
        self.memory = []

    def respond(self, user_input):
        self.memory.append(user_input)
        if len(self.memory) > MEMORY_LIMIT:
            self.memory.pop(0)

        lowered = user_input.lower()
        if lowered in ["selam", "naber"]:
            return "Selam kanka! 😎 Galaksiyi fethetmeye hazır mıyız?" + get_ascii_art()
        if "kod çalıştır" in lowered:
            code = user_input.replace("kod çalıştır", "").strip()
            output = execute_code(code)
            suggestion = suggest_fix(output) if ("Error" in output or "Traceback" in output) else ""
            return f"Kod sonucu:\n{output}\n{suggestion}"
        if "sesli" in lowered:
            return text_to_speech(user_input)

        tokens = self.tokenizer.encode(user_input, max_length=256, truncation=True)
        tokens = np.pad(tokens, (0, 256 - len(tokens)), constant_values=0)

        # Transformer modeli: RNG kullanılmadan tek parametre alan lambda ile çağırıyoruz.
        model = hk.without_apply_rng(hk.transform(lambda t: Transformer()(t)))
        logits = model.apply(self.params, np.array([tokens]))
        token_id = jax.random.categorical(jax.random.PRNGKey(0), logits[:, -1])
        response = self.tokenizer.decode([int(token_id)])
        if any(word in lowered for word in ["fuck", "siktir", "amk"]):
            response = "Haha, kanka sakin! 😄 Galaksi bizim, diss atma!"
        return response + "\n" + get_ascii_art()

# =========================== Çalıştırıcı ===========================
def main():
    if not os.path.exists("checkpoints/ultraheavyai.pkl"):
        print("Model dosyası yok kanka! Önce eğitmen lazım. 😢")
        return
    params = joblib.load("checkpoints/ultraheavyai.pkl")
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    bot = UltraHeavyBot(params, tokenizer)
    print(f"{Fore.MAGENTA}UltraHeavyAI hazır! Galaksiyi fethedecek efsaneyim, kanka! 😎 Çıkmak için 'exit' yaz.{Style.RESET_ALL}")
    while True:
        user_input = input(f"{Fore.YELLOW}Sen: {Style.RESET_ALL}")
        if user_input.lower() == "exit":
            print(f"{Fore.MAGENTA}Görüşürüz, galaksi fatihi! 😎{Style.RESET_ALL}")
            break
        response = bot.respond(user_input)
        print(f"{Fore.CYAN}UltraHeavyAI: {response}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
