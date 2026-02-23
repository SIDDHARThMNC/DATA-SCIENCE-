import pandas as pd

df = pd.read_csv("telecom_churn - Sheet1.csv")

print(df.shape[0])

print(df.isnull().sum())

print(df["Monthly_Charges"].mean())

print(df["Contract_Type"].value_counts())

print(df["Churn"].value_counts())

print(pd.crosstab(df["Contract_Type"], df["Churn"]))

total = df.shape[0]
avg = df["Monthly_Charges"].mean()
max_contract = df["Contract_Type"].value_counts().idxmax()

print(total)
print(avg)
print(max_contract)