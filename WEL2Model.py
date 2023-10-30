from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# load df from data/pmdata/p01/fitbit/hr_sleep.csv
df = None

processed_no = int(input('¿Qué conjunto de datos desea utilizar para el modelo LSTM?: '))
validation = int(input('¿Qué conjunto de datos de participante desea utilizar para la validación (El resto será utilizado para entrenamiento y pruebas)?: '))

for i in range(1, 17):
    if i == validation:
        continue
    participant_df = pd.read_csv(f'data/processed{processed_no}/data/all_hr_sleep_p{i:02d}.csv')
    # concatenate the participant dataframe to the main dataframe
    if df is None:
        df = participant_df
    else:
        df = pd.concat([df, participant_df], ignore_index=True)

label = 'level'
X = df.drop(columns=[label])
Y = df[label]

# Normalize the variables
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)
joblib.dump(scaler, f'data/processed{processed_no}/scaler.pkl')

X_train_full, X_test, y_train_full, y_test = train_test_split(scaled_X, Y, test_size=0.5, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.33, random_state=1)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

models = [
    ('knn', KNeighborsRegressor()),
    ('cart', DecisionTreeRegressor()),
    ('svm', SVR()),
]

from sklearn.metrics import mean_absolute_error

def evaluate_models(models, X_train, X_val, y_train, y_val):
    # fit and evaluate the models
    scores = list()
    for name, model in models:
        # fit the model
        model.fit(X_train, y_train)
        # evaluate the model
        yhat = model.predict(X_val)
        mae = mean_absolute_error(y_val, yhat)
        # store the performance
        scores.append(-mae)
        # report model performance
    return scores

from sklearn.ensemble import VotingRegressor

scores = evaluate_models(models, X_train, X_val, y_train, y_val)
print(scores)
# create the ensemble
ensemble = VotingRegressor(estimators=models, weights=scores)
# fit the ensemble on the training dataset
ensemble.fit(X_train_full, y_train_full)
# make predictions on test set
yhat = ensemble.predict(X_test)
# evaluate predictions
score = mean_absolute_error(y_test, yhat)
print('Weighted Avg MAE: %.3f' % (score))
# evaluate each standalone model
scores = evaluate_models(models, X_train_full, X_test, y_train_full, y_test)
for i in range(len(models)):
    print('>%s: %.3f' % (models[i][0], scores[i]))
# evaluate equal weighting
ensemble = VotingRegressor(estimators=models)
ensemble.fit(X_train_full, y_train_full)
yhat = ensemble.predict(X_test)
score = mean_absolute_error(y_test, yhat)
print('Voting MAE: %.3f' % (score))