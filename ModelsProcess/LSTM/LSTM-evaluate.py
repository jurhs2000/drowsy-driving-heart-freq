import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

# Parámetros
sequence_length = 24
batch_size = 500  # Esto es importante para mantener consistencia con el entrenamiento
num_non_seq_features = 9 # height, fatigue, sleep_duration, sleep_quality, stress, Age_Binned_<25, Age_Binned_25_40, Age_Binned_40_60 y Age_Binned_>60
num_seq_features = 12 # Interval, Time, BPM, BPM_Avg, BPM_Var, HeartRateDiff, BPM_Avg_Diff, Time_SleepStage, SleepStage_Changes, BPM_Trend, BPM_Range y BPM_Std

# Cargar el modelo entrenado
model = tf.keras.models.load_model('out/lstm/best_lstm_model')

def create_dataset_from_tfrecord(tfrecord_path, batch_size):
    """
    Función para cargar datos desde un archivo TFRecord.
    """
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_tfrecord_fn)
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE) # tf.data.experimental.AUTOTUNE
    # dejar solo el 5% para prueba
    #dataset = dataset.take(3)
    return dataset

def parse_tfrecord_fn(example):
    """
    Función para parsear los ejemplos almacenados en TFRecord.
    """
    feature_description = {
        'sequence': tf.io.FixedLenFeature([sequence_length * num_seq_features], tf.float32),
        'non_seq_features': tf.io.FixedLenFeature([num_non_seq_features], tf.float32),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }
    example = tf.io.parse_single_example(example, feature_description)
    
    # Reformatear las características secuenciales en una secuencia (sequence_length, num_seq_features)
    sequence = tf.reshape(example['sequence'], (sequence_length, num_seq_features))
    
    # Las características no secuenciales se mantienen como un vector
    non_seq_features = example['non_seq_features']
    
    label = example['label']
    
    return (sequence, non_seq_features), label

def evaluate_in_chunks(tfrecord_files, model, batch_size):
    """
    Evaluar el modelo usando los archivos TFRecord por chunks.
    """
    y_true_all = []
    y_pred_all = []

    # Barra de progreso para mostrar cuántos chunks se están evaluando
    for tfrecord_file in tqdm(tfrecord_files, desc="Evaluando los chunks"):
        dataset = create_dataset_from_tfrecord(tfrecord_file, batch_size)

        y_true_chunk = []
        y_pred_chunk = []
        
        # Realizar predicciones por chunks
        for (X_seq_chunk, X_non_seq_chunk), y_chunk in tqdm(dataset, desc="Prediciendo"):
            # Realizar las predicciones para este chunk
            y_pred = model.predict([X_seq_chunk, X_non_seq_chunk], verbose=0)  # verbose=0 para evitar mensajes por cada batch
            y_pred = np.argmax(y_pred, axis=1)  # Tomar la clase con mayor probabilidad

            y_true_chunk.extend(y_chunk.numpy())
            y_pred_chunk.extend(y_pred)

        y_true_all.extend(y_true_chunk)
        y_pred_all.extend(y_pred_chunk)

    return np.array(y_true_all), np.array(y_pred_all)

if __name__ == '__main__':
    # Obtener lista de archivos TFRecord de validación
    validation_tfrecord_files = [f'out/lstm/data/{f}' for f in os.listdir('out/lstm/data') if f.startswith("validation_chunk") and f.endswith('.tfrecord')]
    validation_tfrecord_files = sorted(validation_tfrecord_files)
    #validation_tfrecord_files = validation_tfrecord_files[:2] # Solo para pruebas

    model.summary()

    # Evaluar el modelo chunk por chunk
    y_true, y_pred = evaluate_in_chunks(validation_tfrecord_files, model, batch_size)

    # Calcular métricas
    accuracy = accuracy_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    # classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

    print("Accuracy:", accuracy)
    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)