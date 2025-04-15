# UltraLightAI

JAX tabanlı, Triton quantization ile optimize edilmiş, çok dilli bir sohbet botu.

## Özellikler
- Türkçe ve İngilizce destek
- Sesli giriş/çıkış
- Hava durumu ve haber entegrasyonları
- Web arayüzü (tema desteği)
- Niyet ve duygu analizi
- Redis önbellekleme
- Birim testleri
- xAI transformer modeli

## Kurulum

1. Bağımlılıkları yükleyin:
   ```bash
   pip install -r requirements.txt
   ```

2. Redis’i başlatın:
   ```bash
   redis-server
   ```

3. API anahtarlarını ayarlayın:
   - `src/integrations.py` içinde `YOUR_API_KEY` yerlerini güncelleyin.

## Çalıştırma

1. Modeli eğitin:
   ```bash
   python src/train.py
   ```

2. Botu başlatın:
   ```bash
   python main.py
   ```

3. Web arayüzünü başlatın:
   ```bash
   python src/api.py
   ```
   - Tarayıcıda `http://localhost:5000` adresine gidin.

## Örnek Kullanım

**Terminal:**
```
Sen: selam
Bot: Merhaba! 😊 Nasılsın?
Sen: hava nasıl
Bot: İstanbul için hava durumu: güneşli, 20°C.
```

**Web:**
- Mesaj yazın, sesli giriş kullanın veya temayı değiştirin.