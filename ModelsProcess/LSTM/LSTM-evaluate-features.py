import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import os
from tqdm import tqdm
import logging
import concurrent.futures

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Parámetros
sequence_length = 24
batch_size = 500
num_non_seq_features = 9
num_seq_features = 12

# Cargar el modelo entrenado
model = tf.keras.models.load_model('out/lstm/best_lstm_model')

def create_dataset_from_tfrecord(tfrecord_path, batch_size):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_tfrecord_fn)
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.cache()
    #dataset = dataset.take(10) # solo para pruebas
    return dataset

def parse_tfrecord_fn(example):
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

@tf.function
def model_predict(model, X_seq, X_non_seq):
    return model([X_seq, X_non_seq], training=False)

def evaluate_chunk(model, dataset):
    """
    Función para evaluar un solo chunk de datos.
    """
    y_true_chunk = []
    y_pred_chunk = []

    dataset_length = sum(1 for _ in dataset)
    
    for (X_seq_chunk, X_non_seq_chunk), y_chunk in tqdm(dataset, "Evaluando chunk", total=dataset_length):
        y_pred = model_predict(model, X_seq_chunk, X_non_seq_chunk)
        y_pred = np.argmax(y_pred, axis=1)
        y_true_chunk.extend(y_chunk.numpy())
        y_pred_chunk.extend(y_pred)

    return y_true_chunk, y_pred_chunk

def process_tfrecord(tfrecord_file, model, batch_size):
    """
    Función para procesar un archivo TFRecord en paralelo.
    """
    dataset = create_dataset_from_tfrecord(tfrecord_file, batch_size)
    return evaluate_chunk(model, dataset)

def evaluate_in_chunks(tfrecord_files, model, batch_size):
    y_true_all = []
    y_pred_all = []

    # Usamos ThreadPoolExecutor para paralelizar la evaluación por archivo TFRecord
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(lambda tfrecord_file: process_tfrecord(tfrecord_file, model, batch_size), tfrecord_files),
                            total=len(tfrecord_files), desc="Evaluando los chunks"))

    # Unimos los resultados de cada chunk
    for y_true_chunk, y_pred_chunk in results:
        y_true_all.extend(y_true_chunk)
        y_pred_all.extend(y_pred_chunk)

    return np.array(y_true_all), np.array(y_pred_all)

def permute_inplace(arr, axis):
    indices = np.random.permutation(arr.shape[axis])
    return np.take(arr, indices, axis=axis)

def calculate_importance(index, model, tfrecord_files, batch_size, base_accuracy):
    # Determinar si es una característica secuencial o no secuencial
    if index < num_seq_features:
        feature_type = 'sequence'
        feature_index = index
    else:
        feature_type = 'non_seq'
        feature_index = index - num_seq_features

    logging.info(f"Permutando característica {index} ({feature_type})")

    # Recorremos los chunks de TFRecord
    y_true_permuted_all = []
    y_pred_permuted_all = []

    for tfrecord_file in tqdm(tfrecord_files, "Permutando"):
        dataset = create_dataset_from_tfrecord(tfrecord_file, batch_size)
        dataset_len = sum(1 for _ in dataset)

        for (X_seq_chunk, X_non_seq_chunk), y_chunk in tqdm(dataset, "permutando chunk", total=dataset_len):
            # Crear una copia de las características para permutar
            if feature_type == 'sequence':
                X_seq_permuted = X_seq_chunk.numpy()
                np.random.shuffle(X_seq_permuted[:, :, feature_index])  # Permutar la columna seleccionada
                y_pred_permuted = model.predict([X_seq_permuted, X_non_seq_chunk], verbose=0)
            else:
                X_non_seq_permuted = X_non_seq_chunk.numpy()
                np.random.shuffle(X_non_seq_permuted[:, feature_index])
                y_pred_permuted = model.predict([X_seq_chunk, X_non_seq_permuted], verbose=0)

            y_pred_permuted = np.argmax(y_pred_permuted, axis=1)
            y_true_permuted_all.extend(y_chunk.numpy())
            y_pred_permuted_all.extend(y_pred_permuted)

    print(f"permutado listo - index {index}")

    accuracy_permuted = np.mean(np.array(y_true_permuted_all) == np.array(y_pred_permuted_all))
    importance = base_accuracy - accuracy_permuted

    logging.info(f"Importancia de la característica {index}: {importance}")
    return importance

def calculate_importance_wrapper(index, model, tfrecord_files, batch_size, base_accuracy):
    """
    Función wrapper que se pasa a ProcessPoolExecutor para calcular la importancia.
    """
    return calculate_importance(index, model, tfrecord_files, batch_size, base_accuracy)

def permutation_importance(tfrecord_files, model, batch_size, base_accuracy):
    """
    Calcula la importancia de las características mediante permutación.
    """
    feature_importances = np.zeros(num_seq_features + num_non_seq_features)
    
    # Usar paralelización para calcular la importancia de cada característica
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(calculate_importance_wrapper, 
                                    range(num_seq_features + num_non_seq_features),
                                    [model] * (num_seq_features + num_non_seq_features),
                                    [tfrecord_files] * (num_seq_features + num_non_seq_features),
                                    [batch_size] * (num_seq_features + num_non_seq_features),
                                    [base_accuracy] * (num_seq_features + num_non_seq_features)))

    print("termino thread concurrent para permutation importance")
    feature_importances = np.array(results)
    return feature_importances

if __name__ == '__main__':
    validation_tfrecord_files = [f'out/lstm/data/{f}' for f in os.listdir('out/lstm/data') if f.startswith("validation_chunk") and f.endswith('.tfrecord')]
    #validation_tfrecord_files = sorted(validation_tfrecord_files)[:8] # solo para pruebas
    validation_tfrecord_files = sorted(validation_tfrecord_files)

    model.summary()

    # Evaluar el modelo en el dataset sin permutación
    y_true, y_pred = evaluate_in_chunks(validation_tfrecord_files, model, batch_size)
    base_accuracy = np.mean(y_true == y_pred)

    # Evaluar importancia por permutación
    logging.info(f"Precisión base: {base_accuracy}")
    feature_importances = permutation_importance(validation_tfrecord_files, model, batch_size, base_accuracy)

    # Evaluar el rendimiento final
    print("Classification Report:\n", classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

    # Mostrar las importancias
    logging.info(f"Importancia de las características: {feature_importances}")

    # Lista de nombres de las características secuenciales y no secuenciales
    seq_feature_names = [
        "Interval", "Time", "BPM", "BPM_Avg", "BPM_Var", "HeartRateDiff", 
        "BPM_Avg_Diff", "Time_SleepStage", "SleepStage_Changes", "BPM_Trend", 
        "BPM_Range", "BPM_Std"
    ]

    non_seq_feature_names = [
        "height", "fatigue", "sleep_duration", "sleep_quality", "stress", 
        "Age_Binned_<25", "Age_Binned_25_40", "Age_Binned_40_60", "Age_Binned_>60"
    ]

    # Combina las listas de características secuenciales y no secuenciales
    all_feature_names = seq_feature_names + non_seq_feature_names

    # Imprimir las importancias con los nombres de características
    logging.info("Importancia de las características:")
    for name, importance in zip(all_feature_names, feature_importances):
        logging.info(f"{name}: {importance}")
