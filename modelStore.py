import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Define the LSTM model with 5 layers: 1 perceptron layer, 3 LSTM layers, and 2 perceptron layers
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(1, 4)))
model.add(tf.keras.layers.LSTM(128, activation='relu', return_sequences=True))
model.add(tf.keras.layers.LSTM(128, activation='relu', return_sequences=True))
model.add(tf.keras.layers.LSTM(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(3, activation='softmax'))

model.build()
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# load df from data/pmdata/p01/fitbit/hr_sleep.csv
df = pd.read_csv('data/pmdata/all_phases_only_first_cycle_both_types_both_sleeps_3600.csv')

# convert the dateTime column to numeric
df['dateTime'] = df['dateTime'].apply(lambda x: pd.to_datetime(x).timestamp())

X = df[['participant', 'logId', 'dateTime', 'bpm', 'age', 'sex', 'person']].values
y = df['level'].values

# drop age, sex and person columns
X = df[['participant', 'logId', 'dateTime', 'bpm']].values

# Normalize the variables
X = tf.keras.utils.normalize(X, axis=1)

print(X)

# Reshape X to match the model's input shape
X = X.reshape(-1, 1, 4)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
# print the accuracy
print('Test accuracy: {:2.2f}%'.format(test_accuracy*100))

# confusion matrix
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_test, y_pred)
print(cm)
plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(test_accuracy)
plt.title(all_sample_title, size=15)
plt.show()

# Save the model
model.save('model.h5', save_format="h5")
