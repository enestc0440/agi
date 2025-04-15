import requests

def get_weather(city="Istanbul"):
    api_key = "YOUR_API_KEY"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric&lang=tr"
    try:
        response = requests.get(url).json()
        if response.get("cod") != 200:
            return "Hava durumu bilgisi alınamadı."
        weather = response["weather"][0]["description"]
        temp = response["main"]["temp"]
        return f"{city} için hava durumu: {weather}, {temp}°C."
    except:
        return "Hava durumu servisi şu anda kullanılamıyor."

def get_news():
    api_key = "YOUR_API_KEY"
    url = f"https://newsapi.org/v2/top-headlines?country=tr&apiKey={api_key}"
    try:
        response = requests.get(url).json()
        articles = response.get("articles", [])
        if not articles:
            return "Haber bulunamadı."
        return articles[0]["title"]
    except:
        return "Haber servisi şu anda kullanılamıyor."