from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import joblib

# 1. Load the dataset
df = pd.read_csv('AutogluonModels/ag-20231025_040442/all_phases_only_first_cycle.csv')

label = 'level'
X = df.drop(columns=[label])
y = df[label]

# Normalize the variables
scaler = StandardScaler()
print('Columns: ', type(X), X)
scaled_X = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler.pkl')

print('Scaled: ', type(scaled_X), scaled_X)

scaled_df = pd.DataFrame(scaled_X, columns=X.columns)
#scaled_df[label] = y

X_train, X_test, y_train, y_test = train_test_split(scaled_df, y, test_size=0.2, random_state=42)

# 2. Train two classifiers
clf1 = LogisticRegression(max_iter=5000).fit(X_train, y_train)
clf2 = SVC(probability=True).fit(X_train, y_train)

# 3. Weighted ensemble with L2 regularization
# Let's assume initial weights as 0.5 for both classifiers
weights = np.array([0.5, 0.5])
lambda_reg = 0.1  # L2 regularization term

probas1 = clf1.predict_proba(X_test)[:, 1]
probas2 = clf2.predict_proba(X_test)[:, 1]

ensemble_probas = weights[0] * probas1 + weights[1] * probas2
ensemble_preds = (ensemble_probas > 0.5).astype(int)

# Calculate the accuracy
accuracy = np.mean(ensemble_preds == y_test)
print(f"Ensemble accuracy: {accuracy:.2f}")

# If you want to use gradient-based optimization to tune the weights with L2 regularization,
# you can use optimization libraries like Scipy's `minimize` function. The objective
# would be to minimize the negative log likelihood of the ensemble predictions, 
# plus the L2 regularization term.
