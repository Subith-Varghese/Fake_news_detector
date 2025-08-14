from flask import Flask, request, jsonify, render_template
from src.predict_lstm import predict_lstm

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json()
    text = data.get("text", "")
    result = predict_lstm(text)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
