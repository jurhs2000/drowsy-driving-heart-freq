import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# load df from data/pmdata/p01/fitbit/hr_sleep.csv
df = None

processed_no = int(input('¿Qué conjunto de datos desea utilizar para los modelos?: '))
validation = int(input('¿Qué conjunto de datos desea utilizar para la validación (El resto será utilizado para entrenamiento y pruebas)?: '))

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

X_train, X_test, y_train, y_test = train_test_split(scaled_X, Y, test_size=0.2, random_state=42)

# Define the LSTM model with 5 layers: 1 perceptron layer, 3 LSTM layers, and 2 perceptron layers
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(1, scaled_X.shape[1])))
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

# Train the model
model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
# print the accuracy
print('Test accuracy: {:2.2f}%'.format(test_accuracy*100))

# confusion matrix
# validation data
validation_df = pd.read_csv(f'data/processed{processed_no}/data/all_hr_sleep_p{validation:02d}.csv')

X = validation_df.drop(columns=[label])
y_test = validation_df[label]

scaled_X = scaler.fit_transform(X)

scaled_df = pd.DataFrame(scaled_X, columns=X.columns)

y_pred = model.predict(scaled_df)
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
model.save('model-LSTM.h5', save_format="h5")
