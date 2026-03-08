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
    - running_balance: float (CAN BE NEGATIVE)
    
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
    6. CRITICAL: Ensure all newly created column names are entirely unique.
    7. CRITICAL (PRUNING SYSTEM): Our pipeline automatically drops features with >0.85 correlation to existing features. Your new feature MUST be mathematically distinct from existing standard deviation, count, or simple ratio metrics. Think non-linear or complex temporal behaviors.
    8. PREVENT INFINITY & NAN: You must cap bounds if using np.exp(). NEVER use np.log(), np.log1p(), np.sqrt(), or mathematical operations that fail on negative numbers, because `running_balance` can be negative.
    9. ONLY output the raw Python code. NO markdown formatting, NO backticks, NO explanations.
    """
    
    try:
        response = requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": prompt, "stream": False})
        code = response.json().get('response', '')
        
        # Safely remove markdown formatting without triggering parser issues
        code = code.replace("```python", "").replace("```", "").strip()
        
        return code
    except Exception as e:
        print(f"API Error: {e}")
        return None

def evaluate_pipeline():
    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH", "") + ":."
    
    # Use Popen to stream the output to the console in real-time
    process = subprocess.Popen(
        ["python3", "-m", "src.main"], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True, 
        env=env
    )
    
    output_log = ""
    for line in process.stdout:
        print(f"  [Pipeline] {line}", end="")
        output_log += line
        
    process.wait()
    
    match = re.search(r"Full Test ROC AUC:\s*([0-9.]+)", output_log)
    if match:
        return float(match.group(1)), "Success"
    else:
        error_msg = output_log.strip()[-1000:]
        return 0.0, f"Pipeline Error:\n{error_msg}"

def inject_feature(new_code, iteration):
    # FORCEFULLY rename whatever function name the LLM hallucinated
    new_code = re.sub(r'def\s+[a-zA-Z0-9_]+\s*\(', f'def build_novel_feature_{iteration}(', new_code, count=1)
    with open("src/features.py", "r") as f:
        content = f.read()
    modified_content = content + "\n\n" + new_code + "\n"
    
    # ADDED SAFETY NET: Eradicate infs, NaNs, and Overlapping columns
    injection_logic = f"""
try:
    novel_feats = build_novel_feature_{iteration}(history_subset, acct_df)
    import numpy as np
    novel_feats = novel_feats.replace([np.inf, -np.inf], 0).fillna(0)
    
    if 'prism_consumer_id' in novel_feats.columns:
        novel_feats = novel_feats.set_index('prism_consumer_id')
        
    # Drop any columns the LLM generated that ALREADY exist in the matrix to prevent overlap crashes
    cols_to_drop = [c for c in novel_feats.columns if c in features_df.columns]
    if cols_to_drop:
        novel_feats = novel_feats.drop(columns=cols_to_drop)
        
    features_df = features_df.join(novel_feats, how='left')
except Exception as e:
    print(f"Novel feature {iteration} failed: {{e}}")
"""
    injection_indented = "\n".join("    " + line for line in injection_logic.strip().split("\n"))
    
    # CRITICAL FIX: count=1 stops exponential duplication permanently!
    modified_content = re.sub(
        r'([ \t]*)(return features_df\.reset_index\(\))',
        lambda m: injection_indented + "\n" + m.group(1) + m.group(2),
        modified_content,
        count=1
    )
    with open("src/features.py", "w") as f:
        f.write(modified_content)

def run_loop(max_iterations=200):
    global BEST_AUC
    print("\n--- STARTING AUTONOMOUS ML AGENT ---")
    
    # --- NEW MASTER BACKUP LOGIC TO PREVENT STACKING ---
    if not os.path.exists("src/features_master.py"):
        print("💾 Creating master clean backup of features.py...")
        shutil.copy("src/features.py", "src/features_master.py")
    else:
        print("🧹 Restoring clean features.py from master backup to prevent iteration stacking...")
        shutil.copy("src/features_master.py", "src/features.py")
        
    print("⏳ Running baseline model... (Streaming pipeline logs below)")
    BEST_AUC, _ = evaluate_pipeline()
    print(f"\n🎯 Baseline AUC established: {BEST_AUC:.4f}")
    
    failed_logs = []
    try:
        for i in range(1, max_iterations + 1):
            print(f"\n======================================")
            print(f"🔄 ITERATION {i}/{max_iterations}")
            
            # Temporary backup just for this single iteration
            shutil.copy("src/features.py", "src/features_iter_backup.py")
            with open("src/features.py", "r") as f:
                current_features = f.read()
            history_string = "\n---\n".join(failed_logs[-3:]) 
            
            print("🧠 Requesting new feature from LLM...")
            new_code = get_new_feature_code(current_features, history_string)
            if not new_code or "def build_novel_feature" not in new_code:
                print("⚠️ Agent failed to generate valid code. Skipping iteration.")
                continue
            
            inject_feature(new_code, i)
            start_time = time.time()
            
            print("⏳ Evaluating new feature pipeline...")
            new_auc, status_msg = evaluate_pipeline()
            
            elapsed = time.time() - start_time
            print(f"\n⏱️ Iteration {i} completed in {elapsed:.2f} seconds.")
            
            # --- DYNAMIC HURDLE LOGIC (Simulated Annealing) ---
            # Odd iterations: Strict 1.05 multiplier (must improve by 5%)
            # Even iterations: Relaxed 0.95 multiplier (can drop by 5% to escape local minima)
            multiplier = 1.05 if i % 2 != 0 else 0.95
            required_auc = BEST_AUC * multiplier
            mode = "STRICT (1.05x)" if multiplier == 1.05 else "RELAXED (0.95x)"
            
            print(f"📏 Hurdle Mode: {mode} -> Target AUC: {required_auc:.4f}")
            
            if new_auc >= required_auc:
                print(f"✅ SUCCESS! New AUC {new_auc:.4f} met hurdle {required_auc:.4f}.")
                
                # Only raise the ceiling if we actually beat our global high score
                if new_auc > BEST_AUC:
                    BEST_AUC = new_auc
                    print(f"🏆 NEW GLOBAL HIGH SCORE! Baseline raised to {BEST_AUC:.4f}")
                    
                failed_logs.clear()
                
                # WIN! We update the master backup to lock in this new feature
                shutil.copy("src/features.py", "src/features_master.py")
                
                with open("agent_log.txt", "a") as f:
                    f.write(f"Iteration {i}: WIN - AUC {new_auc:.4f} (Mode: {mode})\nCode:\n{new_code}\n\n")
            else:
                print(f"❌ FAILED. AUC {new_auc:.4f} < {required_auc:.4f}. Reverting.")
                if new_auc == 0.0:
                    # Explicitly print the fatal python error so you aren't blind!
                    print(f"   [Error Details]: {status_msg.splitlines()[-1] if status_msg else 'Unknown Fatal Error'}")
                
                # Revert to the state right before this iteration
                shutil.move("src/features_iter_backup.py", "src/features.py")
                reason = status_msg if new_auc == 0.0 else f"Failed hurdle. AUC: {new_auc:.4f}"
                failed_logs.append(f"Attempted Code:\n{new_code}\nResult: {reason}")
    except KeyboardInterrupt:
        if os.path.exists("src/features_iter_backup.py"):
            shutil.move("src/features_iter_backup.py", "src/features.py")
        print(f"\n🛑 Exiting safely. Best AUC achieved: {BEST_AUC:.4f}")

if __name__ == "__main__":
    run_loop()