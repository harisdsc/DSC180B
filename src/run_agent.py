import os
import subprocess
import requests
import re
import time
import shutil

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5-coder:14b"
BEST_AUC = 0.0

def get_new_feature_code(current_features, failed_history=""):
    memory_block = ""
    if failed_history:
        memory_block = f"""
    CRITICAL - PAST FAILED ATTEMPTS:
    You have already tried the following code snippets which FAILED. 
    Review the errors and DO NOT repeat these exact approaches:
    {failed_history}
    """

    schema_block = """
    DATA SCHEMA:
    Your function `build_novel_feature(history_subset, acct_df)` receives two DataFrames.
    
    1. `history_subset` (Merged Transaction & Consumer Data):
    - prism_consumer_id: string 
    - posted_date: datetime 
    - amount: float 
    - credit_or_debit: string ("CREDIT" or "DEBIT")
    - signed_amount: float (Debits negative, credits positive)
    - category: string ("ATM_CASH", "OVERDRAFT", "PAYCHECK", "RENT", etc.)
    - running_balance: float 
    
    2. `acct_df` (Account Snapshots):
    - prism_consumer_id: string 
    - account_type: string ("CHECKING", "SAVINGS")
    - balance: float 
    - balance_date: datetime 
    """

    prompt = f"""
    You are an expert Quant Data Scientist optimizing an XGBoost credit risk model.
    Here are the current feature extraction functions:
    {current_features}
    {memory_block}
    {schema_block}
    
    INSTRUCTIONS:
    1. Formulate a novel quantitative hypothesis using the provided schema.
    2. Write a single Python function named `build_novel_feature(history_subset, acct_df)`.
    3. The function MUST return a pandas DataFrame containing `prism_consumer_id` and the new numeric feature(s).
    4. Handle Division by Zero by adding + 1e-6 to denominators.
    5. Do NOT use any target columns or predict the future.
    6. ONLY output the raw Python code. NO markdown formatting, NO backticks, NO explanations.
    """
    
    try:
        response = requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": prompt, "stream": False})
        code = response.json().get('response', '')
        code = re.sub(r'^```[a-zA-Z]*\n', '', code, flags=re.MULTILINE)
        code = re.sub(r'```\n?$', '', code, flags=re.MULTILINE)
        return code.strip()
    except Exception as e:
        print(f"API Error: {e}")
        return None

def evaluate_pipeline():
    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH", "") + ":."
    result = subprocess.run(["python3", "-m", "src.main"], capture_output=True, text=True, env=env)
    match = re.search(r"Full Test ROC AUC:\s*([0-9.]+)", result.stdout)
    if match:
        return float(match.group(1)), "Success"
    else:
        error_msg = result.stderr.strip()[-1000:] if result.stderr else result.stdout.strip()[-1000:]
        return 0.0, f"Pipeline Error:\n{error_msg}"

def inject_feature(new_code, iteration):
    new_code = re.sub(r'def build_novel_feature\(', f'def build_novel_feature_{iteration}(', new_code)
    with open("src/features.py", "r") as f:
        content = f.read()
    modified_content = content + "\n\n" + new_code + "\n"
    injection_logic = f"""
# --- AUTO INJECTED ---
try:
    novel_feats = build_novel_feature_{iteration}(history_subset, acct_df)
    if 'prism_consumer_id' in novel_feats.columns:
        novel_feats = novel_feats.set_index('prism_consumer_id')
    features_df = features_df.join(novel_feats, how='left')
except Exception as e:
    print(f"Novel feature {iteration} failed: {{e}}")
# ---------------------
"""
    injection_indented = "\n".join("    " + line for line in injection_logic.strip().split("\n"))
    modified_content = re.sub(
        r'([ \t]*)(return features_df\.reset_index\(\))',
        lambda m: injection_indented + "\n" + m.group(1) + m.group(2),
        modified_content
    )
    with open("src/features.py", "w") as f:
        f.write(modified_content)

def run_loop(max_iterations=100):
    global BEST_AUC
    print("\n--- STARTING AUTONOMOUS ML AGENT ---")
    BEST_AUC, _ = evaluate_pipeline()
    print(f"🎯 Baseline AUC established: {BEST_AUC:.4f}")
    failed_logs = []
    try:
        for i in range(1, max_iterations + 1):
            print(f"\n======================================")
            print(f"🔄 ITERATION {i}/{max_iterations}")
            shutil.copy("src/features.py", "src/features.py.bak")
            with open("src/features.py", "r") as f:
                current_features = f.read()
            history_string = "\n---\n".join(failed_logs[-3:]) 
            new_code = get_new_feature_code(current_features, history_string)
            if not new_code or "def build_novel_feature" not in new_code:
                continue
            inject_feature(new_code, i)
            start_time = time.time()
            new_auc, status_msg = evaluate_pipeline()
            elapsed = time.time() - start_time
            
            # 95% Wiggle Room Logic
            required_auc = BEST_AUC * 1.005
            if new_auc >= required_auc:
                print(f"✅ SUCCESS! New AUC {new_auc:.4f} met hurdle {required_auc:.4f}.")
                if new_auc > BEST_AUC:
                    BEST_AUC = new_auc
                failed_logs.clear()
                with open("agent_log.txt", "a") as f:
                    f.write(f"Iteration {i}: WIN - AUC {new_auc:.4f}\nCode:\n{new_code}\n\n")
            else:
                print(f"❌ FAILED. AUC {new_auc:.4f} < {required_auc:.4f}. Reverting.")
                shutil.move("src/features.py.bak", "src/features.py")
                reason = status_msg if new_auc == 0.0 else f"Failed hurdle. AUC: {new_auc:.4f}"
                failed_logs.append(f"Attempted Code:\n{new_code}\nResult: {reason}")
    except KeyboardInterrupt:
        if os.path.exists("src/features.py.bak"):
            shutil.move("src/features.py.bak", "src/features.py")
        print(f"Exiting safely. Best AUC: {BEST_AUC:.4f}")

if __name__ == "__main__":
    run_loop()