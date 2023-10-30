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

scaled_X = scaled_X.reshape(-1, 1, scaled_X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(scaled_X, Y, test_size=0.2, random_state=42)

# Define the LSTM model with 5 layers: 1 perceptron layer, 3 LSTM layers, and 2 perceptron layers
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(1, X_train.shape[2])))
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
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save the model
model.save('model-LSTM.h5', save_format="h5")
