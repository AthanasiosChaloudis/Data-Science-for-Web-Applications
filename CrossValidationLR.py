import numpy as np
import pandas as pd

data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churnutf8.csv')
x = data.drop('Churn', axis=1)
x_list = list(x.columns)
x = np.array(x)
y = np.array(data['Churn'])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LogisticRegression


# CROSS VALIDATION
from sklearn.model_selection import RandomizedSearchCV
penalty = ['l1', 'l2']
C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
max_iter = [100,200,400]

# Create the random grid
random_grid = {'penalty': penalty,
               'C': C,
               'solver': solver,
               'max_iter': max_iter
               }

from pprint import pprint
pprint(random_grid)

LRModel = LogisticRegression()
lr_random = RandomizedSearchCV(estimator=LRModel, param_distributions=random_grid,
                               scoring='f1',
                               verbose=2, random_state=42, n_jobs=-1, n_iter=100)

# Modell trainieren
lr_random.fit(x_train, y_train)
print(lr_random.best_params_)
print("\n")

from sklearn.metrics import accuracy_score


def evaluate(LRModel, x_test, y_test):
    LRpredictions = LRModel.predict(x_test)
    errors = abs(LRpredictions - y_test)
    score = accuracy_score(y_test, LRpredictions)
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(score))
    print("\n")

    return score


base_model = LogisticRegression(random_state=42)
base_model.fit(x_train, y_train)
base_accuracy = evaluate(base_model, x_test, y_test)
best_random = lr_random.best_estimator_
random_accuracy = evaluate(best_random, x_test, y_test)
print('Improvement of {:0.2f}%.'.format(100 * (random_accuracy - base_accuracy) / base_accuracy))
