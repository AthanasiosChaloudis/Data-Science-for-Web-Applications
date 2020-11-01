import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot

data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churnutf8.csv')
heads = list(data.columns.values)

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
print("\n")

from sklearn import metrics
from sklearn.linear_model import LogisticRegression

LRModel = LogisticRegression()
LRModel.fit(x_train, y_train)
LRpredictions = LRModel.predict(x_test)

feature_importance = LRModel.coef_[0]
for i, v in enumerate(feature_importance):
    print(heads[i], 'Score: %.5f' % v)

pyplot.bar([x for x in range(len(feature_importance))], feature_importance)
#pyplot.show()

print("\n")
# Präzision berechnen
score = metrics.accuracy_score(y_test, LRpredictions)
print("score: ", round(score, 5) * 100, "%")
print("Precision:", metrics.precision_score(y_test, LRpredictions))
print("Recall:", metrics.recall_score(y_test, LRpredictions))

from sklearn.metrics import classification_report

print(classification_report(y_test, LRpredictions))

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

# ROC AUC Berechnung und Plotting
logit_roc_auc = roc_auc_score(y_test, LRModel.predict(x_test))
fpr, tpr, thresholds = roc_curve(y_test, LRModel.predict_proba(x_test)[:, 1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

print("ROC Score :", round(roc_auc_score(y_test, LRModel.predict(x_test)),5)*100,"%")
'''
[2.72321656e-02 3.82929195e-01 9.51413919e-02 2.63652268e-01
 7.80972187e-02 1.00000000e+00 2.61666843e-01 1.90312735e-01
 7.37886864e-01 2.80733558e-01 1.96901031e-01 6.49946584e-01
 2.70106436e-02 3.69729762e-02 4.17652247e-01 4.06075028e-01
 7.35982397e-01 5.58138836e-01 1.33909426e-01 2.09860865e-01
 3.84050767e-02 4.19039963e-01 2.12620123e-02 4.49086525e-04]
'''
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
