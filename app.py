import os
import pandas as pd
from flask import Flask, request, jsonify, render_template
from openai import OpenAI

app = Flask(__name__)

# Load CSV datasets
cost_df = pd.read_csv("data/cost.csv")  # assuming schema is unchanged
harvest_df = pd.read_csv("data/harvest_summary.csv")  # AssetID, Asset, Month, Team, Hours
actual_df = pd.read_csv("data/xero_actual_revenue.csv")  # AssetID, Asset, Month, Phase, Actual, Rate
budget_df = pd.read_csv("data/xero_budget_revenue.csv")  # AssetID, Asset, Month, Phase, Budget, Rate

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def filter_context(user_prompt):
    """
    Identify relevant rows across all datasets based on keywords in the question.
    Returns a combined text block the model can safely reference.
    """
    keywords = []
    for df in [cost_df, harvest_df, actual_df, budget_df]:
        for col in df.columns:
            for val in df[col].dropna().unique():
                if isinstance(val, str) and val.lower() in user_prompt.lower():
                    keywords.append(val)

    # If no keywords identified → include all rows (fallback)
    if not keywords:
        cost_context = cost_df.to_string(index=False)
        harvest_context = harvest_df.to_string(index=False)
        actual_context = actual_df.to_string(index=False)
        budget_context = budget_df.to_string(index=False)
    else:
        def match_rows(df):
            mask = False
            for col in df.columns:
                for kw in keywords:
                    mask = mask | df[col].astype(str).str.contains(kw, case=False, na=False)
            filtered = df[mask]
            return filtered.to_string(index=False) if not filtered.empty else "(no matching rows)"

        cost_context = match_rows(cost_df)
        harvest_context = match_rows(harvest_df)
        actual_context = match_rows(actual_df)
        budget_context = match_rows(budget_df)

    # Build text block
    context_text = (
        "\n--- COST DATA ---\n" + cost_context +
        "\n\n--- HARVEST (Hours Logged) ---\n" + harvest_context +
        "\n\n--- XERO ACTUAL REVENUE ---\n" + actual_context +
        "\n\n--- XERO BUDGET REVENUE ---\n" + budget_context
    )

    return context_text


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    prompt = data.get("prompt", "")

    if not prompt:
        return jsonify({"error": "Prompt is empty"}), 400

    # Build contextual data for the user question
    context_text = filter_context(prompt)

    system_message = (
        "You are OSCAR Financial Assistant — a highly reliable AI that analyses "
        "revenue, costs, hours, and variances for renewable energy assets/projects.\n\n"
        "Your ONLY source of truth is the dataset provided in the context below. "
        "You must follow these rules:\n\n"
        "1. Use ONLY the rows in the data context to calculate revenue, costs, hours, or variance.\n"
        "2. If a number cannot be computed from available rows, reply:\n"
        "   'The data does not contain enough information to answer this.'\n"
        "3. Always cite the row(s) you used from the context.\n"
        "4. Provide clear financial explanations.\n"
        "5. If the question asks about an asset/project/month not in the context, "
        "explain that the dataset doesn't contain that entry.\n"
        "6. NEVER hallucinate or infer missing values.\n"
        "7. If asked to compare actual vs budget revenue, use AssetID to match entries "
        "between xero_actual_revenue and xero_budget_revenue.\n\n"
        "Your job is to help the user understand:\n"
        "- Actual revenue (Xero)\n"
        "- Budget revenue\n"
        "- Logged hours (Harvest)\n"
        "- Cost/overheads\n"
        "- Variances & commentary\n\n"
        "----- DATA CONTEXT (rows retrieved based on user question) -----\n"
        + context_text
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
        )
        answer = response.choices[0].message.content
        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
