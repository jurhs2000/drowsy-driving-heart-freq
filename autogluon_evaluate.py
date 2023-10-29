from autogluon.tabular import TabularPredictor
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

processed_no = int(input("Ingrese el número del dataset a usar: "))

model_no = int(input("Ingrese el número del modelo a usar: "))

# load TabularPredictor from directory
predictor = TabularPredictor.load(f'data/processed{processed_no}/AutogluonModels/model{model_no}')
results = predictor.leaderboard()
print(results)

validation_participant = int(input("Ingrese el número del participante a usar como validación: "))

# validation data
validation_df = pd.read_csv(f'data/processed{processed_no}/data/all_hr_sleep_p{validation_participant:02d}.csv')

label = 'level'
X = validation_df.drop(columns=[label])
y_test = validation_df[label]

# Normalize the variables
scaler = joblib.load(f'data/processed{processed_no}/scaler.pkl')
scaled_X = scaler.fit_transform(X)

scaled_df = pd.DataFrame(scaled_X, columns=X.columns)

y_pred = predictor.predict(scaled_df)

print(y_pred)

# evaluate performance
perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)

print(perf)

total_df = scaled_df.copy()
total_df[label] = y_test

# performance
performance = predictor.evaluate(total_df)
print(performance)

# leaderboard
leaderboard = predictor.leaderboard(total_df, silent=True)
print(leaderboard)
# feature importance
print(predictor.feature_importance(total_df))

# confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualiza la matriz de confusión
plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix, annot=True, cmap="YlGnBu", fmt='g')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()