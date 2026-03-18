import pandas as pd

# Path to the input .txt file (each line = one report)
input_txt_path = "/home/chayan/CGI_Net/outputs/x_NLMCXR_ClsGen_Swin_MaxView2_NumLabel114_History_Hyp.txt"

# Path to the output .csv file
output_csv_path = "/home/chayan/TransXplainNet+/Codes/Report_Evaluation/Metrics/IU-CXR/Generated_Reports_IU_CA1.csv"

# Read the text file (each line is a report)
with open(input_txt_path, "r") as file:
    reports = [line.strip() for line in file if line.strip()]

# Create a DataFrame with the required column name
df = pd.DataFrame(reports, columns=["Report Impression"])

# Save to CSV
df.to_csv(output_csv_path, index=False)

print(f"Saved {len(df)} reports to {output_csv_path}")