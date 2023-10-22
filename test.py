import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

predictor = TabularPredictor.load('AutogluonModels/ag-20231022_143722/')
loaded_scaler = joblib.load('scaler.pkl')

# resumen de modelos
print(predictor.leaderboard())

# data
# load df from data/pmdata/p01/fitbit/hr_sleep.csv
df = pd.read_csv('data/pmdata/count_all_phases_only_first_cycle_both_types_both_sleeps_random_early_time.csv')

# convert the dateTime column to numeric
df['dateTime'] = df['dateTime'].apply(lambda x: pd.to_datetime(x).timestamp())

label = 'level'
X = df.drop(columns=[label])
Y = df[label]

# Normalize the variables
scaler = StandardScaler()
print('Columns: ', type(X), X)
scaled_X = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler.pkl')

print('Scaled: ', type(scaled_X), scaled_X)

scaled_df = pd.DataFrame(scaled_X, columns=X.columns)
scaled_df[label] = Y

train_data, test_data = train_test_split(scaled_df, test_size=0.2, random_state=42)

# predictions
y_test = test_data[label]
test_data_nolabel = test_data.drop(columns=[label])
y_pred = predictor.predict(test_data_nolabel)

# confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualiza la matriz de confusión
plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix, annot=True, cmap="YlGnBu", fmt='g')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# performance
performance = predictor.evaluate(test_data)
print(performance)