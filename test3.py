import pandas as pd
import sys
import io 
import contextlib 

# --- 1. Data Loading ---

def load_data():
    """Reads all four CSV files into pandas DataFrames."""
    data = {}
    
    # List of files to load
    files = [
        "harvest_summary.csv",
        "ebitda_summary.csv",
        "xero_actual_revenue.csv",
        "xero_budget_revenue.csv"
    ]
    
    print("--- Loading Data Files ---")
    for file_name in files:
        try:
            # Note: Assuming files are in the same directory as the script.
            df = pd.read_csv(file_name)
            data[file_name] = df
            print(f"Loaded: {file_name} with {len(df)} rows.")
        except FileNotFoundError:
            print(f"Error: File not found: {file_name}. Exiting.")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading {file_name}: {e}. Exiting.")
            sys.exit(1)
            
    return data

# --- 2. Context Building (Schema-Based for Agent) ---

def build_context_schema(data_frames):
    """
    Builds a context string containing only the schema (columns and a sample) 
    of the data for the LLM to write analysis code.
    """
    context_parts = []
    
    # System Instruction for a Code-Writing Agent
    system_instruction = (
        "You are an Analytical Agent. Your task is to write a single block of Python code "
        "to answer the user's question based on the provided data schemas. "
        "The code must use the 'data_frames' dictionary where keys are filenames (e.g., 'harvest_summary.csv') "
        "and values are Pandas DataFrames. Your final answer MUST be printed to stdout using a single print() call. "
        "DO NOT include any explanation or markdown outside the code block."
    )
    context_parts.append(system_instruction)

    # Provide schemas and a small sample
    for file_name, df in data_frames.items():
        context_parts.append(f"\n### SCHEMA: {file_name} ###")
        context_parts.append(f"Columns: {list(df.columns)}")
        context_parts.append(f"Sample (First 3 rows):\n{df.head(3)}")
    
    context_parts.append("\n--- END SCHEMA CONTEXT ---")
    
    return "".join(context_parts)

# --- 3. Agent Helper Function (Simulated Code Execution) ---

def ask_agent(question, context, data_frames):
    """
    Simulates a true agent workflow:
    1. Sends schema and question to LLM. (LLM generates Python code)
    2. Executes the generated code locally using the real DataFrames.
    3. Returns the captured stdout result.
    """
    
    # 1. Build the prompt for the code-generating LLM
    code_generation_prompt = f"{context}\n\nUSER QUESTION: {question}"
    
    print("\n--- AGENT STEP 1: Sending Schema and Question to LLM ---")
    print(code_generation_prompt[:1500] + "...\n")
    
    # --- Actual LLM API Call (Get Code) Would Go Here ---
    # response = gemini_client.models.generate_content(...)
    # generated_code = response.text.strip('`').strip('python') 
    
    # MOCK GENERATED CODE: Code to filter actual revenue for Sep-25 and group by Asset.
    # Defined using a clean triple-quoted string ("""...""") to avoid escaping issues.
    generated_code = """
import pandas as pd
import io

# Define the target month
target_month = 'Sep-25'

# --- Analysis: Actual Revenue by Asset for Sep-25 ---
df_actual_rev = data_frames['xero_actual_revenue.csv']

# 1. Filter data for the target month
sep_data = df_actual_rev[df_actual_rev['Month'] == target_month]

# 2. Group by Asset and sum the Actual revenue
revenue_by_asset = sep_data.groupby('Asset')['Actual'].sum().reset_index()

# 3. Rename columns and sort
revenue_by_asset.columns = ['Asset', 'Total Actual Revenue (Sep-25)']
revenue_by_asset = revenue_by_asset.sort_values(by='Total Actual Revenue (Sep-25)', ascending=False)

# 4. Format revenue for readability
# Using single quotes for the f-string inside the mock code to avoid conflicting with the outer triple quotes.
revenue_by_asset['Total Actual Revenue (Sep-25)'] = revenue_by_asset['Total Actual Revenue (Sep-25)'].apply(lambda x: f'${x:,.2f}')

# 5. Prepare output table
output = io.StringIO()
revenue_by_asset.to_string(output, index=False)
revenue_table = output.getvalue()

# Build the final result string
result = f\"\"\"
## Actual Revenue for September 2025 (Data for Graph)

This table provides the filtered and aggregated data required to create your bar graph:

{revenue_table}
\"\"\"

print(result)
"""
    
    print("--- AGENT STEP 2: Generated Code ---")
    print(generated_code)
    
    # 2. Execute the generated code locally (The "Tool" execution)
    print("\n--- AGENT STEP 3: Executing Code ---")
    
    f = io.StringIO()
    try:
        # Redirect stdout to the StringIO object
        with contextlib.redirect_stdout(f):
            # Use exec to run the code in a scope where data_frames, pd, and io are available
            local_scope = {'data_frames': data_frames, 'pd': pd, 'io': io}
            exec(generated_code, globals(), local_scope)
        
        # Get the captured output
        output = f.getvalue().strip()
        print("Captured Result:")
        print(output)
        return output
        
    except Exception as e:
        error_output = f"ERROR during code execution: {e}"
        print(error_output)
        return error_output

# --- Main Execution ---

if __name__ == "__main__":
    # 1. Load Data
    data_frames = load_data()
    
    # 2. Build Context (Schema-based for the Agent)
    context = build_context_schema(data_frames)
    
    # 3. Test Question (Graph Data)
    question = "Provide the total actual revenue for each asset in September 2025, which will be used to create a bar graph."
    
    # 4. Ask Agent (Execute Agent Workflow)
    answer = ask_agent(question, context, data_frames)
    
    # 5. Print Results
    print("\nQ:", question)
    print("A: The agent executed the code tool. The final answer is:\n", answer)