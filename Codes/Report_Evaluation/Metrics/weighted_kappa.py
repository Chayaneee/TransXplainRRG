import pandas as pd
from sklearn.metrics import cohen_kappa_score, accuracy_score

# load csv
df = pd.read_csv("Inside-Out-Kappa.csv")

r2 = df["Score(Vincent)"] # .str.strip()
r1 = df["Score(Ashu)"]  # .str.strip()

# percentage agreement
agreement = accuracy_score(r1, r2)

# standard kappa
kappa = cohen_kappa_score(r1, r2)

# weighted kappa (quadratic)
weighted_kappa = cohen_kappa_score(r1, r2, weights="quadratic")

print("Agreement:", round(agreement*100,2), "%")
print("Cohen Kappa:", round(kappa,3))
print("Weighted Kappa:", round(weighted_kappa,3))