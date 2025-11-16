import os
import pandas as pd
from flask import Flask, request, jsonify, render_template
from openai import OpenAI

app = Flask(__name__)
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# -------------------------------------------------------
# 1. LOAD YOUR CSV DATA
# -------------------------------------------------------
cost_df = pd.read_csv("cost.csv")
harvest_df = pd.read_csv("harvest_summary.csv")
actual_df = pd.read_csv("xero_actual_revenue.csv")
budget_df = pd.read_csv("xero_budget_revenue.csv")

# -------------------------------------------------------
# 2. PREP MERGED DATA FOR VARIANCE ANALYSIS
# -------------------------------------------------------
merged_df = pd.merge(
    actual_df,
    budget_df,
    on=["AssetID", "Month"],
    how="outer",
    suffixes=("_Actual", "_Budget")
)

merged_df["Variance"] = merged_df["Revenue_Actual"].fillna(0) - merged_df["Revenue_Budget"].fillna(0)


# -------------------------------------------------------
# 3. HELPER FUNCTION: FILTER RELEVANT ROWS
# -------------------------------------------------------
def filter_context(prompt):
    """
    Returns only the relevant rows among cost_df, harvest_df,
    actual_df, budget_df, merged_df based on keywords found in the prompt.
    """

    keywords = prompt.lower()

    context_parts = []

    def df_to_context(df, df_name):
        if df.empty:
            return f"[{df_name}] — NO MATCHING ROWS\n"
        return f"[{df_name}]\n" + df.to_string(index=False) + "\n\n"

    # Filter by AssetID or Month if mentioned
    def filter_df(df):
        filtered = df.copy()

        # Check ID
        for col in df.columns:
            if col.lower() in ["assetid", "projectid"]:
                for word in keywords.split():
                    if word.isdigit():
                        filtered = filtered[filtered[col] == int(word)]

        # Check month (e.g., 2024-07, July, 07/24)
        for col in df.columns:
            if "month" in col.lower():
                for word in keywords.split():
                    if word in str(df[col].values):
                        filtered = filtered[filtered[col].astype(str).str.contains(word)]

        return filtered

    # Detect which datasets the question should use
    if "cost" in keywords:
        context_parts.append(df_to_context(filter_df(cost_df), "COST DATA"))

    if "hour" in keywords or "harvest" in keywords:
        context_parts.append(df_to_context(filter_df(harvest_df), "HOURS (HARVEST)"))

    if "actual" in keywords:
        context_parts.append(df_to_context(filter_df(actual_df), "ACTUAL REVENUE"))

    if "budget" in keywords:
        context_parts.append(df_to_context(filter_df(budget_df), "BUDGET REVENUE"))

    if "variance" in keywords or "compare" in keywords:
        context_parts.append(df_to_context(filter_df(merged_df), "ACTUAL vs BUDGET (MERGED)"))

    # If no keyword matched, include minimal context
    if not context_parts:
        context_parts.append("[No matching dataset — include direct question instead]\n")

    return "\n".join(context_parts)


# -------------------------------------------------------
# 4. SYSTEM PROMPT FOR THE FINANCIAL ASSISTANT
# -------------------------------------------------------
def build_system_prompt(context_text):

    return f"""
You are OSCAR Financial Assistant — a highly reliable AI that analyses
revenue, costs, hours, and variances for renewable energy assets/projects.

Your ONLY source of truth is the dataset provided in the context below.
Never make assumptions. Follow these rules strictly:

1. Use ONLY the data in the context to calculate revenue, hours, costs, or variance.
2. If a number cannot be computed from the available rows, respond:
   'The data does not contain enough information to answer this.'
3. When answering, always cite the specific rows you used from the context.
4. Provide clear, concise financial explanations.
5. If the question refers to a month, asset, or project not in the context,
   explain that the dataset doesn't contain that entry.
6. Do NOT hallucinate, estimate, or infer missing values.

---------------- DATA YOU HAVE ACCESS TO ----------------

You have the following datasets:
- cost_df: overhead and cost data
- harvest_df: logged timesheet hours
- actual_df: actual revenue from Xero
- budget_df: planned/budgeted revenue
- merged_df: actual vs budget revenue with variance

Your role is to help the user understand:
- Actual revenue
- Budget revenue
- Variance
- Hours logged
- Cost breakdown
- Asset/project performance

------------------- CONTEXT BELOW -----------------------
{context_text}
---------------------------------------------------------
"""


# -------------------------------------------------------
# 5. ROUTES
# -------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.get_json().get("prompt", "")

    if not user_input:
        return jsonify({"error": "Prompt is empty"}), 400

    # Generate contextual subset of data
    context_text = filter_context(user_input)

    # Build system message
    system_prompt = build_system_prompt(context_text)

    # Call OpenAI
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
        )

        answer = response.choices[0].message.content
        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------------------------------
# 6. FLASK SERVER
# -------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
