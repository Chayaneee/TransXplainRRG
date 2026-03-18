from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
from tqdm import tqdm
import csv

# Load model and tokenizer
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Load data
data = pd.read_csv("/home/chayan/CGI_Net/outputs/SwinT-52M/Original_vs_Generated_Report.csv")
Original_report = data["Actual_Report"].reset_index(drop=True)[0:100]
Generated_report = data["Generated_Report"].reset_index(drop=True)[0:100]
pid = data["Pid"].reset_index(drop=True)[0:100]
sid = data["Pid"].reset_index(drop=True)[0:100]

# Define terminators
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

# CSV output path
csv_file_path = "/home/chayan/CGI_Net/outputs/SwinT-52M/Inside-Out/Data/Final_MIMIC_test_Swin52M_Llama3-8B-inside-out-trial.csv"

# Open CSV file for writing
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Pid', 'Sid', 'Generated_Report', 'Middle_Part1', 'Middle_Part2', 'Abnormal_Findings_Generated'])

for j in tqdm(range(len(Generated_report))):
    # First LLM message: Reorder the sentences
    messages_first = [
        {"role": "system", "content": ''' You are reading a radiology report. Your task is to reorder sentences into categories as instructed.

Instructions:
1. Write the exact sentences given in the report maintaining sequence of order. 
2. No extra sentences are allowed to write.
3. Don't write like this: 'any of the orders are not mentioned'
4. No extra notes or introductory sentences are allowed.
5.  If any of the orders are not mentioned in report, just don't write anything about that.

Sequential orders:
 
1. Any support devices (includes monitoring and support devices, catheter, pacemaker, tube, wire, cabg) related sentences. 
2. Heart (includes heart, cardio, cardiac silhouette, cardiomegaly) related sentences. 
3. Mediastinal or hili-related sentences. 
4. Lung (includes pleural effusion, pneumothorax, or pulmonary edema, lung diseases, lung volumes) related sentences. 
5. Chest wall or thoracic spine or rib-related sentences. 
6. Abdomen related sentences. 
7. Rest sentences.
            
        '''},
        {"role": "user", "content": f"Reorder the sentences in the following report: {Generated_report[j]}."}
    ]

    input_ids_first = tokenizer.apply_chat_template(
        messages_first,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    outputs_first = model.generate(
        input_ids_first,
        max_new_tokens=1024,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
    )
    output_first = outputs_first[0][input_ids_first.shape[-1]:]
    reordered_response = tokenizer.decode(output_first, skip_special_tokens=True)

    # Process the first response to remove "None" labels
    def filter_reordered_response(response):
        filtered_lines = []
        for line in response.splitlines():
        # Skip lines with unwanted content
            if (
                "None" not in line
                and "related sentences." not in line
                and "related sentences:" not in line
                and line.strip()
            ):
                # If the line contains a colon, extract the content after it; otherwise, use the whole line
                if ":" in line:
                    filtered_lines.append(line.split(":", 1)[-1].strip())
                else:
                    filtered_lines.append(line.strip())
        # Join all filtered sentences into a single paragraph
        return " ".join(filtered_lines)

    filtered_response = filter_reordered_response(reordered_response)
    #print(filtered_response)

    # Second LLM message: Generate final findings without headings
    messages_second = [
        {"role": "system", "content": "You are reading the reordered radiology report. Your task is to combine the exact sentences into a single paragraph without altering them."},
        {"role": "user", "content": f"Combine the following reordered sentences into a single paragraph: {filtered_response}"}
    ]

    input_ids_second = tokenizer.apply_chat_template(
        messages_second,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    outputs_second = model.generate(
        input_ids_second,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.07,
        top_p=0.9,
    )
    output_second = outputs_second[0][input_ids_second.shape[-1]:]
    final_response = tokenizer.decode(output_second, skip_special_tokens=True)

    # Compile results and write to CSV
    results = [pid[j], sid[j], Generated_report[j], reordered_response, filtered_response, final_response]
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(results)
