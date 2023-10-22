from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd

# load TabularPredictor from directory
predictor = TabularPredictor.load('AutogluonModels/ag-20231021_042054/')
results = predictor.leaderboard()
print(results)
# predict on new data
'''test_data = TabularDataset('data/pmdata/all_phases_all_cycles_both_types_both_sleeps.csv')
# convert the dateTime column to numeric
test_data['dateTime'] = test_data['dateTime'].apply(lambda x: pd.to_datetime(x).timestamp())
y_pred = predictor.predict(test_data)

# evaluate performance
label = 'level'
y_test = test_data[label]
perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)

# leaderboard
leaderboard = predictor.leaderboard(test_data, silent=True)
print(leaderboard)
# feature importance
predictor.feature_importance(test_data)'''
