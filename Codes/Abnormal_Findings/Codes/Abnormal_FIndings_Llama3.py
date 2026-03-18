from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)

import pandas as pd
from tqdm import tqdm
import csv
import re

#### Data

data = pd.read_csv("/home/chayan/CGI_Net/outputs/SwinT-52M/Original_vs_Generated_Report.csv")

Original_report = data["Actual_Report"].reset_index(drop=True)[0:100]   ### Mimic Test Set Generated Report
Generated_report = data["Generated_Report"].reset_index(drop=True)[0:100]
pid = data["Pid"].reset_index(drop=True)[0:100] ## for testing [0:10] 
sid = data["Pid"].reset_index(drop=True)[0:100]

#print(Original_report[9])
print(len(Original_report))


terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

# Open CSV file for writing

csv_file_path = "/home/chayan/CGI_Net/outputs/SwinT-52M/Abnormal_Findings/Data/Llama/Final_MIMIC_test_Swin52M_Llama3-8B-Abnormal-Findings-v2.csv"


with open(csv_file_path, mode='w', newline='') as file:
     writer = csv.writer(file)
     writer.writerow(['Pid', 'Sid', 'Generated_Report', 'Abnormal_Findings_Generated'])


for j in tqdm(range(len(Generated_report))):
    messages = [
        {"role": "system", "content": "You are analyzing a radiology report for a chest X-ray examination. Identify **all abnormalities** (including mild or subtle findings such as mild enlargement or changes in size, shape, or structure, post operative status.) and **support and monitoring devices** (e.g., catheters, wires, pacemakers, NG tubes). \
                    Categorize your output into two sections only avoiding any note or extra writings:\
    1. **Abnormal Findings**: Include all findings that suggest any deviation from normal, even if described as mild or subtle. If no abnormalities present, just write 'No abnormal findings'.\
    2. **Support and Monitoring Devices**: Identify all devices mentioned in the report. If support devices are remain in place, mention that also."},
        {"role": "user", "content": f"Find the abnormalities of this report. {Generated_report[j]}." 
    },
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
    )
    output = outputs[0][input_ids.shape[-1]:]
    response = tokenizer.decode(output, skip_special_tokens=True)

    # Compile results and write to CSV
    results = [pid[j], sid[j], Generated_report[j], response]
    with open(csv_file_path, mode='a', newline='') as file:
         writer = csv.writer(file)
         writer.writerow(results)