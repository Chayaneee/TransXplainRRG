import pandas as pd
from tqdm import tqdm
import csv

from openai import OpenAI
client = OpenAI()

data = pd.read_csv("/home/chayan/CGI_Net/outputs/SwinT-52M/Abnormal_Findings/Data/Final_MIMIC_test_Swin52M_GPT4-Abnormal-Findings_v2.csv")

Generated_report = data["Generated_Report"][0:100].reset_index(drop=True)  
Abnormal_findings = data["Abnormal_Findings_Generated"][0:100].reset_index(drop=True) ## for testing [0:10] [20000:25000]

print(Generated_report[2])
print(len(Generated_report))

responses = []


print("Scoring Evaluation Metrics......")


system_prompt = f""" You are given two radiology reports: the original ground-truth report and a filtered abnormal findings-only report. Your task is to evaluate the abnormal findings-only report based on the following criteria:


Errorneous of Abnormal Findings (0, 1, 2, or 3):
Score 0: If all the abnormal findings in the original report are correctly identified in the abnormal findings-only report without missing any. Any support devices like catheter or wire should be considered as abnormal findings.
Score 1: If any abnormal finding from the original report is missing in the abnormal findings-only report.
Score 2: If any false finding (i.e., a finding that is not mentioned in the original report) is present in the abnormal findings-only report.
Score 3: If any abnormal finding is missed (Score 0) and a false finding is present (Score 2).

Clinically Safe or Unsafe (1 or 0):
Score 1: If no critical findings are missed, and the abnormal findings-only report is not affecting patient care. 
Score 0: If critical findings are missed, which could affect the patient's diagnosis or treatment, leading to an unsafe report.

Based on these criterion, write only the score as the following format:
Errorneous: either 0 or 1 or 2 or 3
Safety: either 0 or 1

"""



for j in tqdm(range(len(Generated_report))):
  completion = client.chat.completions.create(
  model="gpt-4-turbo",  ### gpt-4-turbo
  messages=[
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": f"Generated_report: {Generated_report[j]}, \n Output Report: {Abnormal_findings[j]}"}
   ],
   temperature = 0.2)
  response = completion.choices[0].message.content 
  print(response)
  responses.append( [Generated_report[j], Abnormal_findings[j], response])



csv_file_path = "/home/chayan/CGI_Net/outputs/SwinT-52M/Abnormal_Findings/Data/GPT/Final_MIMIC_test_Swin52M_GPT4-Abnormal-Findings_Evaluation.csv"

# Write the results to a CSV file
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header row
    writer.writerow(['Findings', 'Abnormal_Findings', 'Score'])
    # Write each row of the results list to the CSV file
    for result in responses:
        writer.writerow(result)

print("CSV file created successfully.")
