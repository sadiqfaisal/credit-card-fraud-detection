import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('../data/creditcard.csv')

print('Shape:', df.shape)
print(df['Class'].value_counts())
print('Fraud %:', df['Class'].mean() * 100)

plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=df)
plt.title('Class Distribution (0 = Normal, 1 = Fraud)')
plt.savefig('../reports/class_distribution.png')
plt.close()

scaler = StandardScaler()
df['Amount_scaled'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['Time_scaled'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

df.drop(['Amount', 'Time'], axis=1, inplace=True)

df.to_csv('../data/creditcard_cleaned.csv', index=False)
print('Saved cleaned dataset.')
print(df.head())
