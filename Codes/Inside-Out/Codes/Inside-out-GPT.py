import pandas as pd
from tqdm import tqdm
import csv

from openai import OpenAI
client = OpenAI()

data = pd.read_csv("/home/chayan/CGI_Net/outputs/SwinT-52M/Original_vs_Generated_Report.csv") ### Mimic Test Set Generated Report

Original_report = data["Actual_Report"].reset_index(drop=True)[0:100]   ### Mimic Test Set Generated Report
Generated_report = data["Generated_Report"].reset_index(drop=True)[0:100]
pid = data["Pid"].reset_index(drop=True)[0:100] ## for testing [0:10] 
sid = data["Pid"].reset_index(drop=True)[0:100]

print(Original_report[2])
print(len(Original_report))

responses = []


print(" Refinining Original Report according to inside-out approach......")


system_prompt = f""" You are reading a radiology report. Rearrange the sentences of the findings within 1 paragraph maintaining the following order:
1. Any support devices like catheter, pacemaker, orthopedic hardware, wire related sentences
2. Heart related sentences
3. Mediastinal or hili-related sentences
4. Lung related sentences
5. Chest wall or thoracic spine or rib-related sentences
6. Abdomen related sentences
7. Rest sentences.

For your help,
any sentence related to heart, cardio, pacemaker, atrium and ventricle are considered heart related sentence.
any sentence related to pleural effusion, pneumothorax, or pulmonary edema are considered lung related sentence.

Write exactly same sentences. 
If any information about the order is missing, don't write anything about that.

"""



for j in tqdm(range(len(Original_report))):
    # Process the Actual Report
    completion_actual = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": Original_report[j]}
        ],
        temperature=0.2
    )
    response_actual = completion_actual.choices[0].message.content
    
    # Process the Generated Report
    completion_generated = client.chat.completions.create(
        model="gpt-4-turbo",
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
csv_file_path = "/home/chayan/CGI_Net/outputs/SwinT-52M/Inside-Out/Data/Final_MIMIC_test_GPT4-Inside-Out-apprach.csv"

# Write the results to a CSV file
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header row
    writer.writerow(['Pid', 'Sid', 'Actual_Report', 'In_Out_Actual', 'Generated_Report', 'In_Out__Generated'])
    # Write each row of the results list to the CSV file
    for result in responses:
        writer.writerow(result)

print("CSV file created successfully.")