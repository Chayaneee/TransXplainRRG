from transformers import AutoTokenizer
import transformers
import torch

from transformers import LlamaTokenizer


import pandas as pd
from tqdm import tqdm
import csv
import re

#### Data

data = pd.read_csv("/home/chayan/CGI_Net/outputs/SwinT-52M/Original_vs_Generated_Report.csv")

Original_report = data["Actual_Report"].reset_index(drop=True)[0:2]   ### Mimic Test Set Generated Report
Generated_report = data["Generated_Report"].reset_index(drop=True)[0:2]
pid = data["Pid"].reset_index(drop=True)[0:2] ## for testing [0:10] 
sid = data["Pid"].reset_index(drop=True)[0:2]

#print(Original_report[9])
print(len(Original_report))



# Hugging face repo name
model = "meta-llama/Meta-Llama-3-8B-Instruct" #chat-hf (hugging face wrapper version) mistralai/Mistral-7B-v0.1 meta-llama/Llama-2-7b-chat-hf/"meta-llama/Meta-Llama-3.1-8B-Instruct"

#tokenizer = LlamaTokenizer.from_pretrained(model, truncation=True)

tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto" # if you have GPU
)

print(" Refinining Original Report according to Abnormal Findings......")

base_prompt = "<s>[INST]\n<<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt}[/INST]"


csv_file_path = "/home/chayan/CGI_Net/outputs/SwinT-52M/Abnormal_Findings/Data/Llama/Final_MIMIC_test_Swin52M_Llama2-8B-Abnormal-Findings.csv"


     
     

# Open CSV file for writing
system_prompts= ["You are analyzing a radiology report for a chest X-ray examination. Identify **all abnormalities** (including mild or subtle findings such as mild enlargement or changes in size, shape, or structure) and **support and monitoring devices** (e.g., catheters, wires, pacemakers, NG tubes). \
                 Categorize your output into two sections:\
1. **Abnormal Findings**: Include all findings that suggest any deviation from normal, even if described as mild or subtle.\
2. **Support and Monitoring Devices**: Identify all devices mentioned in the report.\
If no abnormalities are found in the report just write a single sentence 'No abnormal findings.' in Abnormal Findings category.",
                "You are analyzing a radiology report for a chest X-ray examination. Just write down the abnormal findings and support devices within a single paragraph without any introduction. Remove all the normal findings from paragraph if any."]
 
   
#system_prompts= [ "You are analyzing a radiology report for a chest X-ray examination. Just mention the abnormal findings and support devices within a single paragraph without any introduction. Remove all the normal findings from paragraph if any. If no abnormalities are found in the report just write a single sentence 'No abnormal findings.'"]
 
#system_prompts= ["Task: Extract Abnormal Findings and Support Devices related sentences.\
#                  Scenario: You are analyzing a radiology report for a chest X-ray examination.\
#                  Instructions:\
#                  1. Extract only list of all the abnormal findings including support devices (e.g., catheters, sternotomy wires, ng tube, pacemakers) related sentences as concisely as possible.\
#                  2. Exclude normal findings entirely.\
#                  3. Include Recommendations, uncertainty-related sentences, or postoperative status related sentence.\
#                  4. If the report states no abnormal findings, respond with only this single sentence: 'There are no abnormal findings'.\
#                  "
#                  ]

# Open CSV file for writing

#with open(csv_file_path, mode='w', newline='') as file:
#     writer = csv.writer(file)
#    writer.writerow(['Pid', 'Sid', 'Generated_Report', 'Abnormal_Findings_Generated'])


# Iterate over reports
#for j in tqdm(range(len(Original_report))):

    # Process the Actual Report
    # Generate response for the first system prompt
    # first_prompt_input_org = base_prompt.format(system_prompt=system_prompts[0], user_prompt=Original_report[j])
    # first_sequences = pipeline(
    #     first_prompt_input_org,
    #     do_sample=True,
    #     top_k=10,
    #     num_return_sequences=1,
    #     eos_token_id=tokenizer.eos_token_id,
    #     max_length=1000,
    # )
    # first_response_original = first_sequences[0]['generated_text'].split('[/INST]')[-1].strip()

    # # Use the response of the first prompt as the user prompt for the second system prompt
    # second_prompt_input_org = base_prompt.format(system_prompt=system_prompts[1], user_prompt=first_response_original)
    # second_sequences = pipeline(
    #     second_prompt_input_org,
    #     do_sample=True,
    #     top_k=10,
    #     num_return_sequences=1,
    #     eos_token_id=tokenizer.eos_token_id,
    #     max_length=1000,
    # )
    # second_response_original = second_sequences[0]['generated_text'].split('[/INST]')[-1].strip()
    
    # Process the Generated Report
    
     #first_prompt_input_gen = base_prompt.format(system_prompt=system_prompts[0], user_prompt=Generated_report[j])
#     first_sequences = pipeline(
#         first_prompt_input_gen,
#         do_sample=True,
#         temperature = 0.5,
#         top_k=10,
#         num_return_sequences=1,
#         eos_token_id=tokenizer.eos_token_id,
#         max_length=1000,
#         truncation=True,
        
#     )
#     first_response_generated = first_sequences[0]['generated_text'].split('[/INST]')[-1].strip()

#     # Use the response of the first prompt as the user prompt for the second system prompt
#     second_prompt_input_gen = base_prompt.format(system_prompt=system_prompts[1], user_prompt=first_response_generated)
#     second_sequences = pipeline(
#         second_prompt_input_gen,
#         do_sample=True,
#         temperature = 0.5,
#         top_k=10,
#         num_return_sequences=1,
#         eos_token_id=tokenizer.eos_token_id,
#         max_length=1000,
#         truncation=True,
        
#     )
#     second_response_generated = second_sequences[0]['generated_text'].split('[/INST]')[-1].strip()

#     response = re.sub(r"\b(Sure|Sure!|Okay|Great),.*?:", "", second_response_generated).strip()

#     # Compile results and write to CSV
#     results = [pid[j], sid[j], Generated_report[j], response]
#     with open(csv_file_path, mode='a', newline='') as file:
#          writer = csv.writer(file)
#          writer.writerow(results)





## Test a single report
report = " in comparison with the study of <unk> , there is little overall change . again there is evidence of pneumothorax with substantial decrease in the degree of right pleural effusion and underlying compressive atelectasis . cardiac silhouette remains mildly enlarged . no definite vascular congestion . the right ij catheter tip extends to the mid portion of the svc ."

first_prompt_input_gen = base_prompt.format(system_prompt=system_prompts[0], user_prompt= report)
first_sequences = pipeline(
    first_prompt_input_gen,
    do_sample=True,
    temperature = 0.2,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=1000,
    truncation=True
)

first_response_generated = first_sequences[0]['generated_text'].split('[/INST]')[-1].strip()
first_response_generated = re.sub(r"\b(Sure|Sure!|Okay|Great),.*?:", "", first_response_generated).strip()

# Use the response of the first prompt as the user prompt for the second system prompt
second_prompt_input_gen = base_prompt.format(system_prompt=system_prompts[1], user_prompt=first_response_generated)
second_sequences = pipeline(
    second_prompt_input_gen,
    do_sample=True,
    temperature = 0.2,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=1000,
    truncation=True
)
second_response_generated = second_sequences[0]['generated_text'].split('[/INST]')[-1].strip()

response = re.sub(r"\b(Sure|Sure!|Okay|Great),.*?:", "", second_response_generated).strip()
print(response)

