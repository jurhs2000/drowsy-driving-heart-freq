import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tqdm import tqdm
import joblib
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pyarrow.parquet as pq
import gc

print("Dispositivos disponibles:", tf.config.list_physical_devices('GPU'))

# Parámetros
sequence_length = 24
chunk_size = 500000

# Variables continuas para el StandardScaler
continuous_features = ['height', 'sleep_duration', 'Interval', 'Time', 'BPM', 'BPM_Mean_Acc', 'BPM_Var_Acc', 'BPM_Std_Acc', 'BPM_Diff', 'BPM_Acceleration', 'BPM_Mean_Diff', 'Time_SleepStage', 'SleepStage_Changes', 'BPM_Trend']
# Variables de rango fijo para el MinMaxScaler
fixed_range_features = ['fatigue', 'stress', 'sleep_quality']

# Crear carpeta de salida si no existe
if not os.path.exists('out/lstm/data'):
    os.makedirs('out/lstm/data')

# Función para confirmar si el usuario quiere continuar
def ask_for_confirmation(dataset):
    user_input = input(f"¿Deseas continuar generando las secuencias de {dataset}? (s/n): ")
    if user_input.lower() != 's':
        return False
    return True

def create_sequences_for_session(sessionId, group, sequence_length):
    sequences = []
    non_seq_features_list = []  # Lista para las características no secuenciales
    labels = []

    # Tomar las características no secuenciales de la primera fila del grupo (no cambian en una sesión)
    non_seq_features = group.iloc[0][['height', 'fatigue', 'sleep_duration', 'sleep_quality', 'stress', 
                                    'Age_Binned_<25', 'Age_Binned_25_40', 'Age_Binned_40_60', 'Age_Binned_>60']].values.astype(np.float32)

    for i in range(len(group) - sequence_length):
        seq = group.iloc[i:i + sequence_length].drop(columns=['SleepStage', 'sessionId', 'height', 'fatigue', 'sleep_duration', 'sleep_quality', 'stress', 
                                                    'Age_Binned_<25', 'Age_Binned_25_40', 'Age_Binned_40_60', 'Age_Binned_>60']).values.astype(np.float32)
        if seq.shape == (sequence_length, 12):  # Asegúrate de ajustar el número de características secuenciales correctamente
            label = group.iloc[i + sequence_length - 1]['SleepStage']
            sequences.append(seq)
            labels.append(label)
            non_seq_features_list.append(non_seq_features)
        else:
            print(f"Advertencia: Secuencia con tamaño incorrecto en sessionId {sessionId}. Tamaño: {seq.shape}")
    
    return sequences, non_seq_features_list, labels

def create_sequences(data, sequence_length):
    bpm_index = data.columns.get_loc('BPM')
    print(f"Índice de la columna 'BPM': {bpm_index}")
    grouped = data.groupby('sessionId')

    # Procesar en paralelo las secuencias por cada sessionId
    results = Parallel(n_jobs=-1)(delayed(create_sequences_for_session)(sessionId, group, sequence_length) 
                                    for sessionId, group in tqdm(grouped, desc="Creando secuencias"))
    
    # Verificar que todas las secuencias tengan la misma dimensión y no estén vacías
    valid_sequences = [np.array(res[0]) for res in results if len(res[0]) > 0 and np.array(res[0]).ndim == 3]
    valid_non_seq_features = [res[1] for res in results if len(res[0]) > 0 and np.array(res[0]).ndim == 3]
    valid_labels = [res[2] for res in results if len(res[0]) > 0 and np.array(res[0]).ndim == 3]

    # Solo concatenar si hay secuencias válidas
    if valid_sequences:
        sequences = np.concatenate(valid_sequences)
        non_seq_features = np.concatenate(valid_non_seq_features)
        labels = np.concatenate(valid_labels)
    else:
        # Si no hay secuencias válidas, retornar arrays vacíos
        sequences = np.array([])
        non_seq_features = np.array([])
        labels = np.array([])

    return sequences, non_seq_features, labels  # Ahora también devolvemos las características no secuenciales

def process_chunk(start, df, buffer, chunk_size, sequence_length, sscaler, mmscaler, chunk_count, tfrecord_prefix):
    tfrecord_file = f"{tfrecord_prefix}_chunk_{chunk_count}.tfrecord"
    
    if os.path.exists(tfrecord_file):
        print(f"[INFO] {tfrecord_file} ya existe. Saltando generación de secuencias.")
        return

    print(f"\n[INFO] Procesando chunk {chunk_count}...")

    chunk = df.iloc[start:start + chunk_size]
    chunk = pd.concat([buffer, chunk])

    if chunk.empty:
        print(f"[WARNING] Chunk {chunk_count} está vacío, saltando procesamiento.")
        return buffer  # Salir temprano si el chunk está vacío

    last_sessionId = chunk.iloc[-1]['sessionId']
    buffer = chunk[chunk['sessionId'] == last_sessionId]
    chunk = chunk[chunk['sessionId'] != last_sessionId]

    # Ajustar y luego transformar los datos de entrenamiento
    if len(chunk) > 0:
        # Transformar las características de entrenamiento
        chunk[continuous_features] = sscaler.transform(chunk[continuous_features])
        chunk[fixed_range_features] = mmscaler.transform(chunk[fixed_range_features])
    else:
        print(f"[WARNING] Chunk {chunk_count} está vacío, saltando procesamiento.")

    # Crear secuencias para el chunk actual
    X_chunk, non_seq_chunk, y_chunk = create_sequences(chunk, sequence_length)

    # Guardar las secuencias en un archivo TFRecord si son válidas
    if len(X_chunk) > 0:
        save_to_tfrecord(X_chunk, non_seq_chunk, y_chunk, tfrecord_file)

    del chunk
    del X_chunk, non_seq_chunk, y_chunk

    gc.collect()  # Importar y utilizar el recolector de basura para liberar memoria
    return buffer

def create_sequences_in_chunks(parquet_file, sequence_length, chunk_size, tfrecord_prefix, sscaler, mmscaler):
    """
    Función para procesar secuencias en chunks en paralelo y guardarlas en TFRecord.
    """
    df = parquet_file.to_pandas()

    # Ajustar el StandardScaler y MinMaxScaler con el conjunto de entrenamiento completo
    if "train" in tfrecord_prefix:
        print("[INFO] Ajustando los scalers con el conjunto de entrenamiento completo...")
        sscaler.fit(df[continuous_features])
        mmscaler.fit(df[fixed_range_features])

    buffer = pd.DataFrame()  # Para almacenar las filas que no completan un chunk

    n_jobs = -1 if os.cpu_count() > 1 else 1  # Si solo hay un núcleo disponible, evitar la paralelización

    # Definir una función paralelizada usando joblib
    results = Parallel(n_jobs)(
        delayed(process_chunk)(start, df, buffer, chunk_size, sequence_length, sscaler, mmscaler, chunk_count + 1, tfrecord_prefix)
        for chunk_count, start in enumerate(range(0, len(df), chunk_size))
    )

    # Procesar el buffer final si no está vacío
    if not buffer.empty:
        print(f"\n[INFO] Procesando el buffer final...")
        X_chunk, non_seq_chunk, y_chunk = create_sequences(buffer, sequence_length)  # Debes obtener las non_seq_features correctamente aquí
        if len(X_chunk) > 0:
            tfrecord_file = f"{tfrecord_prefix}_buffer.tfrecord"
            save_to_tfrecord(X_chunk, non_seq_chunk, y_chunk, tfrecord_file)  # Asegúrate de guardar las características no secuenciales

        # Liberar memoria del buffer una vez procesado
        del buffer
        del X_chunk, non_seq_chunk, y_chunk  # Libera las secuencias generadas correctamente

    print(f"[INFO] Todos los chunks han sido procesados y guardados en TFRecord.")

# Función para serializar los ejemplos y guardarlos en TFRecord
def serialize_example(sequence_features, non_seq_features, label):
    feature = {
        'sequence': tf.train.Feature(float_list=tf.train.FloatList(value=sequence_features.flatten())),  # Convierte las secuencias en un array plano
        'non_seq_features': tf.train.Feature(float_list=tf.train.FloatList(value=non_seq_features)),  # Guarda las características no secuenciales
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))  # Guarda la etiqueta
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))  # Crea el ejemplo serializado
    return example_proto.SerializeToString()  # Devuelve la secuencia serializada lista para ser escrita en el TFRecord

# Función para guardar las secuencias en TFRecord
def save_to_tfrecord(sequences, non_seq_features, labels, file_name):
    with tf.io.TFRecordWriter(file_name) as writer:
        for seq, non_seq, lbl in zip(sequences, non_seq_features, labels):
            example = serialize_example(seq, non_seq, lbl)  # Pasar también las características no secuenciales
            writer.write(example)
    print(f"[INFO] Guardado en {file_name}")

# Proceso para cada conjunto de datos (entrenamiento, validación, prueba)
def process_dataset(parquet_file_path, dataset_type, sscaler, mmscaler):
    parquet_file = pq.read_table(parquet_file_path, use_threads=True)
    tfrecord_prefix = f'out/lstm/data/{dataset_type}'
    create_sequences_in_chunks(parquet_file, sequence_length, chunk_size, tfrecord_prefix, sscaler, mmscaler)

if __name__ == '__main__':
    # Inicializar el scaler
    sscaler = StandardScaler()
    mmscaler = MinMaxScaler()

    # Procesar el conjunto de entrenamiento
    if ask_for_confirmation('entrenamiento'):
        print("[INFO] Procesando conjunto de entrenamiento...")
        process_dataset('data/LSTM_train_80.parquet', 'train', sscaler, mmscaler)
    else:
        print("[INFO] Procesamiento del conjunto de entrenamiento omitido.")
        sscaler = joblib.load('out/lstm/LSTM_standard_scaler.pkl')
        mmscaler = joblib.load('out/lstm/LSTM_minmax_scaler.pkl')

    # Procesar el conjunto de prueba
    if ask_for_confirmation('prueba'):
        print("[INFO] Procesando conjunto de prueba...")
        process_dataset('data/LSTM_test_10.parquet', 'test', sscaler, mmscaler)

    # Procesar el conjunto de validación
    if ask_for_confirmation('validación'):
        print("[INFO] Procesando conjunto de validación...")
        process_dataset('data/LSTM_validation_10.parquet', 'validation', sscaler, mmscaler)

    print("[INFO] Secuencias generadas y guardadas exitosamente.")

    joblib.dump(sscaler, 'out/lstm/LSTM_standard_scaler.pkl')
    joblib.dump(mmscaler, 'out/lstm/LSTM_minmax_scaler.pkl')
    print("[INFO] Scalers guardados exitosamente.")
