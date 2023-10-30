from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import pandas as pd
import seaborn as sns
import numpy as np
import joblib

processed_no = int(input('¿Qué conjunto de datos desea utilizar para validar el modelo LSTM?: '))
validation = int(input('¿Qué conjunto de datos de participante desea utilizar para la validación (El resto será utilizado para entrenamiento y pruebas)?: '))

# validation data
validation_df = pd.read_csv(f'data/processed{processed_no}/data/all_hr_sleep_p{validation:02d}.csv')

label = 'level'
X = validation_df.drop(columns=[label])
y_test = validation_df[label]

scaler = joblib.load(f'data/processed{processed_no}/scaler.pkl')
scaled_X = scaler.fit_transform(X)

scaled_X = scaled_X.reshape(-1, 1, scaled_X.shape[1])

model = tf.keras.models.load_model('model-LSTM.h5')

print(model.summary())

y_pred = model.predict(scaled_X)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(scaled_X, y_test)
# print the accuracy
print('Test accuracy: {:2.2f}%'.format(test_accuracy*100))

y_pred = np.argmax(y_pred, axis=1)
# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(test_accuracy)
plt.title(all_sample_title, size=15)
plt.show()
