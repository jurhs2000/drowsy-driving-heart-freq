import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import os
from tqdm import tqdm
import shap

# Parámetros
sequence_length = 24
batch_size = 500
num_non_seq_features = 9
num_seq_features = 12

# Cargar el modelo entrenado
model = tf.keras.models.load_model('out/lstm/best_lstm_model', compile=False)

def create_dataset_from_tfrecord(tfrecord_path, batch_size):
    """
    Función para cargar datos desde un archivo TFRecord.
    """
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_tfrecord_fn)
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.take(50)
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
    
    sequence = tf.reshape(example['sequence'], (sequence_length, num_seq_features))
    non_seq_features = example['non_seq_features']
    label = example['label']
    
    return (sequence, non_seq_features), label

def evaluate_in_chunks(tfrecord_files, model, batch_size):
    """
    Evaluar el modelo usando los archivos TFRecord por chunks.
    """
    y_true_all = []
    y_pred_all = []

    for tfrecord_file in tqdm(tfrecord_files, desc="Evaluando los chunks"):
        dataset = create_dataset_from_tfrecord(tfrecord_file, batch_size)

        y_true_chunk = []
        y_pred_chunk = []
        
        for (X_seq_chunk, X_non_seq_chunk), y_chunk in tqdm(dataset, desc="Prediciendo"):
            y_pred = model.predict([X_seq_chunk, X_non_seq_chunk], verbose=0)
            y_pred = np.argmax(y_pred, axis=1)

            y_true_chunk.extend(y_chunk.numpy())
            y_pred_chunk.extend(y_pred)

        y_true_all.extend(y_true_chunk)
        y_pred_all.extend(y_pred_chunk)

    return np.array(y_true_all), np.array(y_pred_all)

def model_predict_wrapper(data):
    print("[INFO] in model_predict_wrapper")
    """
    Envoltorio de la función predict para manejar múltiples entradas del modelo.
    """
    # Asumiendo que data es una combinación de las características secuenciales y no secuenciales
    # Necesitamos dividir el array `data` de vuelta en las dos partes que espera el modelo.
    
    # Extraer el tamaño de las características secuenciales
    seq_features_len = sequence_length * num_seq_features

    # Dividir las características secuenciales y no secuenciales
    X_seq = data[:, :seq_features_len].reshape(-1, sequence_length, num_seq_features)
    X_non_seq = data[:, seq_features_len:]

    # Pasar ambas partes al modelo como entradas
    return model.predict([X_seq, X_non_seq])

import shap

def calculate_shap_for_batch(X_seq, X_non_seq, model, background_sample_size=1):
    """
    Calcular valores SHAP para un lote de datos utilizando KernelExplainer con un subconjunto de fondo.
    """
    print("[INFO] X_seq.shape: ", X_seq.shape)
    print("[INFO] X_non_seq.shape: ", X_non_seq.shape)
    # Concatenar las secuencias y las características no secuenciales
    X_combined = np.concatenate([X_seq.reshape(X_seq.shape[0], -1), X_non_seq], axis=1)
    
    # Seleccionar un subconjunto de datos de fondo (ej. 100 muestras)
    X_background = shap.sample(X_combined, background_sample_size)

    print("[INFO] tamaño de X_combined: ", X_combined.shape)
    print("[INFO] tamaño de X_background: ", X_background.shape)
    
    # Crear un explainer con un fondo reducido
    explainer = shap.KernelExplainer(model_predict_wrapper, X_background)
    
    # Calcular los valores SHAP
    shap_values = explainer.shap_values(X_combined)
    
    return shap_values

def model_predict(data):
    """
    Función personalizada para predecir con múltiples entradas.
    SHAP solo puede manejar una entrada, así que dividimos la entrada en dos.
    """
    # Asumimos que 'data' viene en un solo array concatenado, así que se divide en dos partes
    X_seq = data[:, :sequence_length * num_seq_features]
    X_seq = X_seq.reshape((-1, sequence_length, num_seq_features))
    
    X_non_seq = data[:, sequence_length * num_seq_features:]
    
    return model.predict([X_seq, X_non_seq])

def shap_feature_importance(tfrecord_file, model, batch_size, num_samples=10):
    """
    Calcular la importancia de las características usando SHAP con un subconjunto pequeño.
    """
    dataset = create_dataset_from_tfrecord(tfrecord_file, batch_size)
    dataset = dataset.take(num_samples)  # Tomar solo un subconjunto pequeño

    # Convertir el dataset a NumPy arrays
    X_seq_all = []
    X_non_seq_all = []
    for (X_seq_chunk, X_non_seq_chunk), _ in dataset:
        X_seq_all.append(X_seq_chunk.numpy())
        X_non_seq_all.append(X_non_seq_chunk.numpy())

    X_seq_all = np.concatenate(X_seq_all, axis=0)
    X_non_seq_all = np.concatenate(X_non_seq_all, axis=0)

    # Aplanar secuencias para concatenar con características no secuenciales
    X_seq_all_flat = X_seq_all.reshape((X_seq_all.shape[0], -1))

    # Concatenar características secuenciales y no secuenciales
    X_combined_all = np.concatenate([X_seq_all_flat, X_non_seq_all], axis=1)

    # Crear el explicador SHAP con la función de predicción personalizada
    explainer = shap.KernelExplainer(model_predict, X_combined_all)

    # Calcular los valores SHAP
    shap_values = explainer.shap_values(X_combined_all)
    
    return shap_values, X_combined_all, X_seq_all, X_non_seq_all

if __name__ == '__main__':
    # Obtener lista de archivos TFRecord de validación
    validation_tfrecord_files = [f'out/lstm/data/{f}' for f in os.listdir('out/lstm/data') if f.startswith("validation_chunk") and f.endswith('.tfrecord')]
    validation_tfrecord_files = sorted(validation_tfrecord_files)
    validation_tfrecord_files = validation_tfrecord_files[:3]

    model.summary()

    # Evaluar el modelo chunk por chunk
    y_true, y_pred = evaluate_in_chunks(validation_tfrecord_files, model, batch_size)

    print("Classification Report:\n", classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

    # Calcular la importancia de características con SHAP
    print("Calculando la importancia de características usando SHAP...")
    shap_values, X_combined_all, X_seq_all, X_non_seq_all = shap_feature_importance(validation_tfrecord_files[0], model, 10)

    # Mostrar el gráfico de resumen de SHAP
    shap.summary_plot(shap_values, X_combined_all)
