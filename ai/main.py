import threading
from src.bot import ChatBot
from src.utils import memory_guard

def main():
    bot = ChatBot()
    print("Sohbet Botu Başlatıldı! Çıkmak için 'exit' yazın.")
    
    while True:
        user_input = input("Sen: ")
        if user_input.lower() == 'exit':
            break
        response = bot.respond(user_input, use_speech=True)
        print(f"Bot: {response}")

if __name__ == "__main__":
    threading.Thread(target=memory_guard, daemon=True).start()
    main()