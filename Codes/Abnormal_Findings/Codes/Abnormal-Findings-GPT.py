import pandas as pd
from tqdm import tqdm
import csv

from openai import OpenAI
client = OpenAI()

#data = pd.read_csv("/home/chayan/OpenAI-API/Filtered-mimic-cxr-2.0.0-chexpert-label-FIndings-test.csv") ### Mimic Test Set Original


#Original_report = data["FINDINGS:"].reset_index(drop=True)[0:100]  ### Mimic Test Set Original
#file_path = data["File"].reset_index(drop=True)[0:100] ## for testing [0:10] [20000:25000]
#csv_file_path = "/home/chayan/MIMIC-Dataset/Data/Abnormal_Findings/GPT-API/Final_MIMIC_test_GPT-4_API-Abnormal-FIndings-V3.csv" 



data = pd.read_csv("/home/chayan/CGI_Net/outputs/SwinT-52M/Original_vs_Generated_Report.csv") ### Mimic Test Set Generated Report

Original_report = data["Actual_Report"].reset_index(drop=True)[0:100]   ### Mimic Test Set Generated Report
Generated_report = data["Generated_Report"].reset_index(drop=True)[0:100]
pid = data["Pid"].reset_index(drop=True)[0:100] ## for testing [0:10] 
sid = data["Pid"].reset_index(drop=True)[0:100]

print(Original_report[2])
print(len(Original_report))

responses = []


print(" Refinining Original Report according to Abnormal Findings......")


system_prompt = f""" You are reading a radiology report. Extract concisely only the abnormal findings from the following report mentioned in user content. Any support devices like catheter, pacemaker, orthopedic hardware, wire should be considered as abnormal findings.\
Any recommendations or uncertainty related sentences or post operative status should be also considered as abnormal. Exclude any normal findings. If no abnormality is present, indicate Everything is Normal. 
"""


#system_prompt = f""" You are reading a radiology report. Extract concisely only the abnormal findings from the following report mentioned in user content. Any support devices like catheter or wire should be considered as abnormal findings.
#Exclude any normal findings. If no abnormality is present, indicate Everything is Normal. Write within a paragraph.
#"""

responses = []  # Initialize the list to store results

for j in tqdm(range(len(Original_report))):
    # Process the Actual Report
    completion_actual = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": Original_report[j]}
        ],
        temperature=0.2
    )
    response_actual = completion_actual.choices[0].message.content
    
    # Process the Generated Report
    completion_generated = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": Generated_report[j]}
        ],
        temperature=0.2
    )
    response_generated = completion_generated.choices[0].message.content
    print(response_generated)
    
    # Store both responses
    responses.append([pid[j], sid[j], Original_report[j], response_actual, Generated_report[j], response_generated])

# Define the CSV file path
csv_file_path = "/home/chayan/CGI_Net/outputs/SwinT-52M/Abnormal_Findings/Data/GPT/Final_MIMIC_test_Swin52M_GPT3.5-Abnormal-Findings.csv"

# Write the results to a CSV file
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header row
    writer.writerow(['Pid', 'Sid', 'Actual_Report', 'Abnormal_Findings_Actual', 'Generated_Report', 'Abnormal_Findings_Generated'])
    # Write each row of the results list to the CSV file
    for result in responses:
        writer.writerow(result)

print("CSV file created successfully.")
