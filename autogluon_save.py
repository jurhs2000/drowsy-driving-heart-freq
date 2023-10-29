from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# load df from data/pmdata/p01/fitbit/hr_sleep.csv
df = None

processed_dataset_no = int(input('¿Qué conjunto de datos desea utilizar para los modelos?: '))
validation = int(input('¿Qué conjunto de datos desea utilizar para la validación (El resto será utilizado para entrenamiento y pruebas)?: '))

for i in range(1, 17):
    if i == validation:
        continue
    participant_df = pd.read_csv(f'data/processed{processed_dataset_no}/data/all_hr_sleep_p{i:02d}.csv')
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
joblib.dump(scaler, f'data/processed{processed_dataset_no}/scaler.pkl')

scaled_df = pd.DataFrame(scaled_X, columns=X.columns)
scaled_df[label] = Y

# tabular and leaderboard
train_data = TabularDataset(scaled_df)
predictor = TabularPredictor(label=label).fit(train_data, time_limit=1200)
leaderboard = predictor.leaderboard(train_data, silent=True)
print(leaderboard)