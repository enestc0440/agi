from flask import Flask, request, jsonify, render_template
from .bot import ChatBot

app = Flask(__name__, template_folder="../templates", static_folder="../static")
bot = ChatBot()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
async def chat():
    data = request.get_json()
    user_input = data.get("message", "")
    use_speech = data.get("use_speech", False)
    response = bot.respond(user_input, use_speech)
    return jsonify({"response": response})

@app.route("/speech", methods=["POST"])
async def speech():
    user_input = bot.processor.speech_to_text()
    response = bot.respond(user_input, use_speech=True)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(port=5000)