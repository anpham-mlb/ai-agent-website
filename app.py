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
        print(f"Received prompt: {prompt}")  # Logs to Render

        try:
            import os
            from openai import OpenAI

            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            print("API key loaded")  # Logs to Render

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"Error: {e}"
            print(f"Error: {e}")  # Logs the error

        return render_template("index.html", answer=answer)
    return render_template("index.html", answer=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
