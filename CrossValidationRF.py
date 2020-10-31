import numpy as np
import pandas as pd

data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churnutf8.csv')
# data = pd.read_csv('testRFsmall.csv')

# print(data.head())

# x: alles außer 'Churn'
# y: nur 'Churn'
x = data.drop('Churn', axis=1)

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

# CROSS VALIDATION

from sklearn.model_selection import RandomizedSearchCV
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
criterion = ['gini', 'entropy']
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
max_features = ['auto', 'sqrt', 'log2']
bootstrap = [True, False]
oob_score = [True]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'criterion': criterion,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
               'oob_score': oob_score}

from pprint import pprint
pprint(random_grid)

# RF parameter: ob Bootstrapped Datensätze verwendet werden sollen, Anzahl Bäume, Methode zur Auswahl zufälliger
# Variablen, Random Faktor

from sklearn.ensemble import RandomForestClassifier

RFmodel = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator=RFmodel, param_distributions=random_grid,
                               n_iter=2, scoring='f1',
                               cv=3, verbose=2, random_state=42, n_jobs=-1,
                               return_train_score=True)

# Modell trainieren
rf_random.fit(x_train, y_train)
print(rf_random.best_params_)
print("\n")

from sklearn.metrics import accuracy_score


def evaluate(RFmodel, x_test, y_test):
    RFpredictions = RFmodel.predict(x_test)
    errors = abs(RFpredictions - y_test)
    score = accuracy_score(y_test, RFpredictions)
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(score))
    print("\n")

    return score


base_model = RandomForestClassifier(n_estimators=10, random_state=42)
base_model.fit(x_train, y_train)
base_accuracy = evaluate(base_model, x_test, y_test)
best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, x_test, y_test)
print('Improvement of {:0.2f}%.'.format(100 * (random_accuracy - base_accuracy) / base_accuracy))
