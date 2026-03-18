import pandas as pd
from sklearn.metrics import cohen_kappa_score, accuracy_score

# load file
df = pd.read_csv("/home/chayan/TransXplainNet+/Report_Evaluation/Clinical_Evaluation.csv")

r1 = df["R1(Vincent)"]
r2 = df["R2(Ashu)"]

# Cohen's Kappa
kappa = cohen_kappa_score(r1, r2)

# Percentage agreement
agreement = accuracy_score(r1, r2)

print("Cohen's Kappa:", round(kappa,3))
print("Percentage Agreement:", round(agreement*100,2), "%")