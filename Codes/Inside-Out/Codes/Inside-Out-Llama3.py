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

Original_report = data["Actual_Report"].reset_index(drop=True)[0:10]   ### Mimic Test Set Generated Report
Generated_report = data["Generated_Report"].reset_index(drop=True)[0:10]
pid = data["Pid"].reset_index(drop=True)[0:10] ## for testing [0:10] 
sid = data["Pid"].reset_index(drop=True)[0:10]

#print(Original_report[9])
print(len(Original_report))


terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

# Open CSV file for writing

csv_file_path = "/home/chayan/CGI_Net/outputs/SwinT-52M/Inside-Out/Data/Final_MIMIC_test_Swin52M_Llama3-8B-inside-out-trial.csv"


with open(csv_file_path, mode='w', newline='') as file:
     writer = csv.writer(file)
     writer.writerow(['Pid', 'Sid', 'Generated_Report', 'Abnormal_Findings_Generated'])


for j in tqdm(range(len(Generated_report))):

#    messages = [
#         {"role": "system", "content": "You are reading a radiology report. Rearrange the sentences in the findings section based on the following order. Include only the mentioned categories in the exact same wording from the findings:\
# 1. Sentences about support devices like catheters, pacemakers, tubes or wires. \
# 2. Heart-related sentences (e.g., mentioning the heart, cardiac silhouette, cardiomegaly). \
# 3. Mediastinal or hili-related sentences.\
# 4. Lung-related sentences (e.g., pleural effusion, pneumothorax, pulmonary edema).\
# 5. Sentences about the chest wall, thoracic spine, or ribs.\
# 6. Sentences about the abdomen.\
# 7. Any other remaining sentences.\
# Do not add or modify any text. Skip categories that are not mentioned."},

# {"role": "user", "content": f"Rearrange the sentences from the findings section of the report into one paragraph following the above order and instructions: {Generated_report[j]}." 
#     },
#     ]

    messages = [
        {"role": "system", "content": ''' You are reading a radiology report as normal reader. Your task is to rewriting the exact sentences of the report maintaining a sequential order and carefully following instruction given below:
Instructions:
1. Write the exact sentences given in the report maintaining sequence of order. 
2. No extra sentences are allowed to write.
3. Don't write like this: 'any of the orders are not mentioned'
4. No extra notes or introductory sentences are allowed.
5. If any of the orders are not mentioned in report, just don't write anything about that.

Sequential orders:
 
1. Any support devices (includes monitoring and support devices, catheter, pacemaker, tube, wire, cabg) related sentences. 
2. Heart (includes heart, cardio, cardiac silhouette, cardiomegaly) related sentences. 
3. Mediastinal or hili-related sentences. 
4. Lung (includes pleural effusion, pneumothorax, or pulmonary edema, lung diseases, lung volumes) related sentences. 
5. Chest wall or thoracic spine or rib-related sentences. 
6. Abdomen related sentences. 
7. Rest sentences.

'''
},

        {"role": "user", "content": f''' 

Write the reordered report of the following radiology report maintaining the sequential orders and instructions given in system content. 
{Generated_report[j]} 
''' 
}
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=1024,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.05,
        top_p=0.9,
    )
    output = outputs[0][input_ids.shape[-1]:]
    response = tokenizer.decode(output, skip_special_tokens=True)

    # Compile results and write to CSV
    results = [pid[j], sid[j], Generated_report[j], response]
    with open(csv_file_path, mode='a', newline='') as file:
         writer = csv.writer(file)
         writer.writerow(results)
         
         
#Input Findings: right chest tube remains in place , with a small right apical pneumothorax and a small right apical pneumothorax . cardiomediastinal and hilar contours are stable . persistent small right pleural effusion with adjacent atelectasis . persistent small right pleural effusion.\
#Output: Right chest tube remains in place. Cardiomediastinal and hilar contours are stable. Persistent small right pleural effusion with adjacent atelectasis. Persistent small right pleural effusion. A small right apical pneumothorax and a small right apical pneumothorax.\

#Instructions:\
#If no information about any of the orders is mentioned, don't write anything about that like 'No support devices like catheter, pacemaker, wire are mentioned.'\
#any sentence related to heart, cardio, cardiac silhouette, cardiomegaly are considered heart related sentence.\
#any sentence related to pleural effusion, pneumothorax, or pulmonary edema are considered lung related sentence.\
#Rewrite the report in order, ensuring all key details are preserved but without needing to write extra sentences.\


# Input Report: frontal radiograph of the chest demonstrates low lung volumes with bibasilar atelectasis . there is mild pulmonary vascular congestion without frank interstitial edema . the heart is top normal in size . the mediastinal and hilar contours are normal . there are no definite pleural effusions . there is no pneumothorax . there is a focal area of increased opacity in the right lower lobe which could represent pneumonia or aspiration .
#         Reordered Report: The heart is top normal in size. The mediastinal and hilar contours are normal. Frontal radiograph of the chest demonstrates low lung volumes with bibasilar atelectasis. There is mild pulmonary vascular congestion without frank interstitial edema. There are no definite pleural effusions. There is no pneumothorax. There is a focal area of increased opacity in the right lower lobe which could represent pneumonia or aspiration.\

# Input Report: endotracheal tube terminates <num> cm above the carina .  right internal jugular catheter terminates in the mid svc . lungs are low in volume with stable right upper lung opacities which are better assessed on the recent chest ct but suspicious for pneumonia . there is no pneumothorax or pleural effusion . heart is normal in size . normal cardiomediastinal silhouette .
# Reordered Report: Endotracheal tube terminates <num> cm above the carina. Right internal jugular catheter terminates in the mid SVC. Heart is normal in size. Normal cardiomediastinal silhouette. Lungs are low in volume with stable right upper lung opacities which are better assessed on the recent chest CT but suspicious for pneumonia. There is no pneumothorax or pleural effusion.

# Input Report: right chest tube remains in place , with a small right apical pneumothorax and a small right apical pneumothorax . cardiomediastinal and hilar contours are stable . persistent small right pleural effusion with adjacent atelectasis . persistent small right pleural effusion.
# Reordered Report: Right chest tube remains in place. Cardiomediastinal and hilar contours are stable. Persistent small right pleural effusion with adjacent atelectasis. Persistent small right pleural effusion. A small right apical pneumothorax and a small right apical pneumothorax.

# Input Report: interval extubation . large right upper lobe mass has slightly worsened in the interval , with residual patchy opacity in the right upper lobe and right upper lobe . this may reflect a combination of pleural effusion and collapse and or aspiration in the appropriate clinical setting . left lung is clear . heart and mediastinal contours are stable in appearance . no definite pneumothorax.
# Reordered Report: Interval extubation. Heart and mediastinal contours are stable in appearance. Large right upper lobe mass has slightly worsened in the interval, with residual patchy opacity in the right upper lobe and right upper lobe. This may reflect a combination of pleural effusion and collapse and or aspiration in the appropriate clinical setting. Left lung is clear. No definite pneumothorax.

# Input Report: {Generated_report[j]}
# Reordered Report: