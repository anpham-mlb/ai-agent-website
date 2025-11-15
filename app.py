import os
from flask import Flask, request, render_template
from openai import OpenAI

app = Flask(__name__)
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

PORT = int(os.environ.get("PORT", 5000))  # default 5000 locally

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        prompt = request.form.get("prompt")
        # Example OpenAI call (adjust as needed)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content
        return render_template("index.html", answer=answer)
    return render_template("index.html", answer=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
