import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Define the LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(1, 7)),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(3, activation='softmax')  # Assuming 3 event classes
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# load df from data/pmdata/p01/fitbit/hr_sleep.csv
df = pd.read_csv('data/pmdata/all_hr_sleep.csv')

# convert the dateTime column to numeric
df['dateTime'] = df['dateTime'].apply(lambda x: pd.to_datetime(x).timestamp())

X = df[['participant', 'logId', 'dateTime', 'bpm', 'age', 'sex', 'person']].values
y = df['level'].values

# Reshape X to match the model's input shape
X = X.reshape(-1, 1, 7)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
# print the accuracy
print('Test accuracy: {:2.2f}%'.format(test_accuracy*100))

# Save the model
model.save('model.h5', save_format="h5")

# Load the model
model = tf.keras.models.load_model('model.h5')
