import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Генерация данных
np.random.seed(42)
n_patients = 600
data = pd.DataFrame({
    'Pregnancies': np.random.randint(0, 10, n_patients),
    'Glucose': np.random.randint(70, 200, n_patients),
    'BloodPressure': np.random.randint(60, 110, n_patients),
    'BMI': np.random.uniform(18, 45, n_patients),
    'Outcome': np.random.choice([0, 1], n_patients, p=[0.65, 0.35])
})


data.loc[data.sample(frac=0.08).index, 'Glucose'] = np.nan
data.loc[data.sample(frac=0.05).index, 'BloodPressure'] = 0 

BloodPressure_mean = round(data['BloodPressure'].mean(),2)
data['BloodPressure'] = data['BloodPressure'].replace(0,BloodPressure_mean)
print(data[data['BloodPressure'] == BloodPressure_mean])

sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Тепловая карта корреляций признаков')
plt.show()


X = data.drop('Outcome', axis=1)
y = data['Outcome']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)

print("Отчет о классификации (Classification Report):")
print(classification_report(y_test, y_pred))

# 4. График важности признаков (Feature Importances)
importances = rf_classifier.feature_importances_
print(importances)
features = X.columns
print(features)

plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=features, palette="viridis")
plt.title('Важность признаков (Feature Importances)')
plt.xlabel('Относительная важность')
plt.ylabel('Признаки')
plt.show()