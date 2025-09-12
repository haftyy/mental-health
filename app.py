from flask import Flask, jsonify, render_template, request

from chatbot import chatbot_reply

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/stress")
def stress():

    return render_template("stress.html")

@app.route("/anxiety")
def anxiety():
    return render_template("anxiety.html")

@app.route("/resources")
def resources():
    return render_template("resources.html")

@app.route("/chat")
def chat():
    return render_template("chat.html")
@app.route("/get_reply", methods=["POST"])
def get_bot_response():
    # safer JSON parsing
    data = request.get_json(silent=True) or {}
    user_input = data.get("message")   # <-- matches frontend JS

    if not user_input:
        return jsonify({"reply": "⚠️ Please type a message."})

    try:
        response = chatbot_reply(user_input)
    except Exception as e:
        app.logger.exception("Error generating reply")
        return jsonify({"reply": "⚠️ Sorry, something went wrong on the server."})

    return jsonify({"reply": response})   # <-- matches frontend JS expectation


if __name__ == "__main__":
    app.run(debug=True)