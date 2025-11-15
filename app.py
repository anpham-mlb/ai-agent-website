import os
import pandas as pd
from flask import Flask, request, jsonify, render_template
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# -----------------------------
# 1. LOAD ALL CSV DATA
# -----------------------------
DATA_FILES = {
    "cost": "data/cost.csv",
    "harvest_summary": "data/harvest_summary.csv",
    "xero_actual_revenue": "data/xero_actual_revenue.csv",
    "xero_budget_revenue": "data/xero_budget_revenue.csv",
}

dataframes = {}
for name, path in DATA_FILES.items():
    try:
        dataframes[name] = pd.read_csv(path)
        print(f"Loaded {name}: {dataframes[name].shape[0]} rows")
    except Exception as e:
        print(f"Failed to load {name} from {path}: {e}")

# -----------------------------
# 2. CONVERT DATA TO TEXT CHUNKS
# -----------------------------
def df_to_chunks(df, source_name):
    chunks = []
    for i, row in df.iterrows():
        text = f"[{source_name} | Row {i}] " + ", ".join([f"{col}: {row[col]}" for col in df.columns])
        chunks.append(text)
    return chunks

all_chunks = []
for name, df in dataframes.items():
    all_chunks.extend(df_to_chunks(df, name))

# -----------------------------
# 3. GENERATE EMBEDDINGS FOR ALL CHUNKS
# -----------------------------
def embed(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [np.array(e.embedding) for e in response.data]

print("Generating embeddings... This happens only once.")
chunk_embeddings = embed(all_chunks)
print(f"Created {len(chunk_embeddings)} embeddings.")

# -----------------------------
# 4. FIND TOP RELEVANT CHUNKS
# -----------------------------
def search(query, top_k=5):
    query_emb = embed([query])[0]
    sims = cosine_similarity([query_emb], chunk_embeddings)[0]
    top_idx = sims.argsort()[-top_k:][::-1]

    results = []
    for idx in top_idx:
        results.append(all_chunks[idx])
    return results

# -----------------------------
# 5. FLASK ROUTES
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    prompt = data.get("prompt", "")

    if not prompt:
        return jsonify({"error": "Prompt is empty"}), 400

    try:
        # Retrieve relevant rows
        context_chunks = search(prompt, top_k=6)
        context_text = "\n".join(context_chunks)

        # Send data + question to GPT
        system_message = (
            "You are a financial assistant. Answer ONLY using the data in context.\n"
            "If the answer is not in the data, say you cannot find it.\n"
            "Dataset context:\n" + context_text
        )

        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
        )

        answer = response.choices[0].message.content
        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------
# 6. RUN APP
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
