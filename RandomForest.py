import numpy as np
import pandas as pd

data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churnutf8.csv')

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

# Größe der Unterteilungen überprüfen
print('Training x Größe:', x_train.shape)
print('Training y Größe:', y_train.shape)
print('Testing x Größe:', x_test.shape)
print('Testing y Größe:', y_test.shape)
print("\n")

# RF parameter: ob Bootstrapped Datensätze verwendet werden sollen, Anzahl Bäume, Methode zur Auswahl zufälliger
# Variablen, Random Faktor
parameters = {'bootstrap': True,
              'n_estimators': 200,
              'max_features': 'sqrt',
              'random_state': 0,
              'criterion': 'gini',
              'oob_score': True}

lokOptParameter = {'n_estimators': 1600,
                   'min_samples_split': 2,
                   'min_samples_leaf': 4,
                   'max_features': 'sqrt',
                   'max_depth': 10,
                   'bootstrap': True,
                   'oob_score': True}


from sklearn.ensemble import RandomForestClassifier

# Forest bauen anhand der oberen Parameter
RFmodel = RandomForestClassifier(**lokOptParameter)

# Modell trainieren
RFmodel.fit(x_train, y_train)

# Vorhersagen erstellen
RFpredictions = RFmodel.predict(x_test)

from sklearn.metrics import accuracy_score

# Präzision berechnen
score = accuracy_score(y_test, RFpredictions)
print("score: ", round(score, 5) * 100, "%")

# OOB score CHECK
print("OOB-Score : ", round(RFmodel.oob_score_, 5) * 100, "%")
print("\n")

# key feature extract
feature_importance = pd.Series(RFmodel.feature_importances_, index=x_list).sort_values(ascending=False)
print(feature_importance)
print("\n")

# Confusion Matrix
from sklearn.metrics import confusion_matrix
xAx = np.array(y_test)
yAx = np.array(RFpredictions)
cm = confusion_matrix(xAx, yAx)
print(cm)
TP = cm[0, 0]
FP = cm[1, 0]
FN = cm[0, 1]
TN = cm[1, 1]
print("True Positive: ", TP, "\nFalse Negative: ", FN, "\nTrue Negative :", TN, "\nFalse Positive :", FP)


from sklearn.metrics import classification_report
print(classification_report(y_test, RFpredictions))


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

logit_roc_auc = roc_auc_score(y_test, RFmodel.predict(x_test))
fpr, tpr, thresholds = roc_curve(y_test, RFmodel.predict_proba(x_test)[:, 1])
plt.figure()
plt.plot(fpr, tpr, label='Random Forest (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('RF_ROC')
plt.show()

print("ROC Score :", round(roc_auc_score(y_test, RFmodel.predict(x_test)),5)*100,"%")


import plotly.offline as py
import plotly.graph_objs as go

#Korrelation
correlation = data.corr()
#Labels
matrix_cols = correlation.columns.tolist()
#Array Umformung
corr_array  = np.array(correlation)

#Plotting
trace = go.Heatmap(z = corr_array,
                   x = matrix_cols,
                   y = matrix_cols,
                   colorscale = "Inferno",
                   colorbar   = dict(title = "Pearson Korrelationskoeffizient",
                                     titleside = "right"
                                    ) ,
                  )

layout = go.Layout(dict(title = "Korrelationsmatrix für Variablen",
                        autosize = False,
                        height  = 720,
                        width   = 800,
                        margin  = dict(r = 0 ,l = 210,
                                       t = 25,b = 210,
                                      ),
                        yaxis   = dict(tickfont = dict(size = 9)),
                        xaxis   = dict(tickfont = dict(size = 9))
                       )
                  )

data = [trace]
fig = go.Figure(data=data,layout=layout)
py.plot(fig)


