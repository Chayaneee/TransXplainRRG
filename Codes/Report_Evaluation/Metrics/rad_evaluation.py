from RadEval import RadEval
import json
import pandas as pd


# === Step 1: Load CSV files ===
# Replace with your actual file paths
generated_path = "/home/chayan/TransXplainNet+/Codes/Report_Evaluation/Metrics/IU-CXR/Generated_Reports_IU_CA1.csv"
reference_path = "/home/chayan/TransXplainNet+/Codes/Report_Evaluation/Metrics/IU-CXR/Reference_Reports_IU_CA1.csv"

# Read both CSV files
generated_df = pd.read_csv(generated_path)
reference_df = pd.read_csv(reference_path)

# Ensure both have the same number of samples
assert len(generated_df) == len(reference_df), "Mismatch in number of samples"

# Extract impressions (you can adjust column name if it's different)
gen_impressions = generated_df['Report Impression'].fillna("").tolist()
ref_impressions = reference_df['Report Impression'].fillna("").tolist()


evaluator = RadEval(
    do_radgraph=True,
    do_chexbert=True,
    do_rouge=True,
    do_bertscore=True,
    #do_radcliq=True,
    #do_green = True
    do_bleu=True
)

results = evaluator(refs=ref_impressions, hyps=gen_impressions)

print(results)