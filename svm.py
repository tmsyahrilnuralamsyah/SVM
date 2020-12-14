import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics

df = pd.read_csv("diabetes.csv")

X = df[df.columns[:8]]
y = df['Outcome']

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

data = np.array(X)
df = pd.DataFrame(data=data, columns=['Pregnancies', 'Glucose', 'BloodPressure', \
    'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = SVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy :", metrics.accuracy_score(y_test, y_pred))
print("Precision :", metrics.precision_score(y_test, y_pred))
print("Recall :", metrics.recall_score(y_test, y_pred))
print("Confusion Matrix :", metrics.confusion_matrix(y_test, y_pred))
print("Classification Report :\n", metrics.classification_report(y_test, y_pred))