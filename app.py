import os
from flask import Flask, request, jsonify, render_template
from openai import OpenAI

app = Flask(__name__)
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

@app.route("/")
def index():
    return render_template("index.html")  # Serve the HTML page

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    prompt = data.get("prompt", "")

    if not prompt:
        return jsonify({"error": "Prompt is empty"}), 400

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response.choices[0].message.content
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
