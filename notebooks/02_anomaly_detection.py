import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('../data/creditcard_cleaned.csv')

X = df.drop('Class', axis=1)
y = df['Class']

iso_forest = IsolationForest(
    n_estimators=100,
    contamination=y.mean(),
    random_state=42
)
iso_pred = iso_forest.fit_predict(X)
iso_pred = np.where(iso_pred == -1, 1, 0)

print('=== Isolation Forest Results ===')
print(confusion_matrix(y, iso_pred))
print(classification_report(y, iso_pred))

lof = LocalOutlierFactor(n_neighbors=20, contamination=y.mean())
lof_pred = lof.fit_predict(X)
lof_pred = np.where(lof_pred == -1, 1, 0)

print('=== Local Outlier Factor Results ===')
print(confusion_matrix(y, lof_pred))
print(classification_report(y, lof_pred))
