import pandas as pd

from src.agents.semantic_analyzer import analyze_semantics

df = pd.read_csv("data/uploads/Churn_Modelling.csv")

semantic = analyze_semantics(
    dataframe=df,
    use_llm=False,
)

print("\n========== CHURN DATASET ==========\n")

print("Columns:")
print(df.columns.tolist())

print("\nTarget Candidates:")
print(semantic["target_candidates"])

print("\nRoles:\n")

for column, info in semantic["columns"].items():
    print(f"{column:20} -> {info['role']}")