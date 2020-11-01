import numpy as np
import pandas as pd

data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churnutf8SMALL.csv')

# x: alles außer 'Churn'
# y: nur 'Churn'
x = data.drop(['Churn'], axis=1)

# Wichtig für später
x_list = list(x.columns)

# in np.arrays umformen
x = np.array(x)
y = np.array(data['Churn'])


from sklearn.model_selection import train_test_split

# Unterteilung in Training und Test Data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
# random_state=int: um Zufälligkeit herauszunehmen

# Größe der Unterteilungen überprüfen
print('Training x Größe:', x_train.shape)
print('Training y Größe:', y_train.shape)
print('Testing x Größe:', x_test.shape)
print('Testing y Größe:', y_test.shape)


# Support Vector Machine
from sklearn import svm, metrics

# SVC: support vector classifier
SVMmodel = svm.SVC(kernel='linear')

# Wichtige Parameter: Kernel: basic classifications Funktion DEF: rbf
#                     Regularization: C Parameter: Wie viel Prozent Error ist in Ordnung DEF: 1.0
#                     Gamma: fitting DEF: scale

SVMmodel.fit(x_train, y_train)
SVMpredictions = SVMmodel.predict(x_test)


# Präzision berechnen
score = metrics.accuracy_score(y_test, SVMpredictions)
print("score: ", round(score, 5) * 100, "%")
print("Precision:", metrics.precision_score(y_test, SVMpredictions))
print("Recall:", metrics.recall_score(y_test, SVMpredictions))

from sklearn.metrics import classification_report
print(classification_report(y_test, SVMpredictions))


'''
LAUFZEIT CA 20-25 MINUTEN:

Training x Größe: (1456, 24)
Training y Größe: (1456,)
Testing x Größe: (364, 24)
Testing y Größe: (364,)
score:  82.692 %
Precision: 0.6046511627906976
Recall: 0.3611111111111111
              precision    recall  f1-score   support

           0       0.86      0.94      0.90       292
           1       0.60      0.36      0.45        72

    accuracy                           0.83       364
   macro avg       0.73      0.65      0.67       364
weighted avg       0.81      0.83      0.81       364

Process finished with exit code 0
'''