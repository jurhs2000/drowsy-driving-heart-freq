from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# load df from data/pmdata/p01/fitbit/hr_sleep.csv
df = pd.read_csv('data/pmdata/count_all_phases_only_first_cycle_both_types_both_sleeps_random_early_time.csv')

# convert the dateTime column to numeric
df['dateTime'] = df['dateTime'].apply(lambda x: pd.to_datetime(x).timestamp())

label = 'level'
X = df.drop(columns=[label])
Y = df[label]

# Normalize the variables
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler.pkl')

scaled_df = pd.DataFrame(scaled_X, columns=X.columns)
scaled_df[label] = Y

# tabular and leaderboard
train_data = TabularDataset(scaled_df)
predictor = TabularPredictor(label=label).fit(train_data, time_limit=1200)
leaderboard = predictor.leaderboard(train_data, silent=True)
print(leaderboard)
predictor.feature_importance(train_data)
