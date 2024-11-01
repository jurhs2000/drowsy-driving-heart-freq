import os
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate, BatchNormalization, TimeDistributed, Attention, Bidirectional
from tensorflow.keras import mixed_precision
from tensorflow.keras.regularizers import l2
import re
import gc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Solo errores importantes

# Imprimir la versión de TensorFlow
print(tf.__version__)

print("Precision policy:", mixed_precision.global_policy())
# Habilitar la política de mixed precision para acelerar en GPUs
mixed_precision.set_global_policy('mixed_float16')

#tf.debugging.set_log_device_placement(True)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Estrategia distribuida para múltiples GPUs
strategy = tf.distribute.MirroredStrategy()

# Parámetros del modelo
sequence_length = 24
num_non_seq_features = 9 # height, fatigue, sleep_duration, sleep_quality, stress, Age_Binned_<25, Age_Binned_25_40, Age_Binned_40_60 y Age_Binned_>60
num_seq_features = 12 # Interval, Time, BPM, BPM_Mean_Acc, BPM_Var_Acc, BPM_Std_Acc, BPM_Diff, BPM_Acceleration, BPM_Mean_Diff, Time_SleepStage, SleepStage_Changes y BPM_Trend
bpm_index = 7
batch_size = 500
epochs = 12

# Función para crear datasets desde archivos TFRecord
def create_dataset_from_tfrecord(tfrecord_path, batch_size):
    print(f"[INFO] Cargando datos desde {tfrecord_path}")
    dataset = tf.data.TFRecordDataset(f'out/lstm/data/{tfrecord_path}')
    dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    print(f"[INFO] Dataset de {tfrecord_path} cargado correctamente")
    return dataset

# Función para parsear un TFRecord
def parse_tfrecord_fn(example):
    feature_description = {
        'sequence': tf.io.FixedLenFeature([sequence_length * num_seq_features], tf.float32),  # 12 características secuenciales
        'non_seq_features': tf.io.FixedLenFeature([num_non_seq_features], tf.float32),  # 9 características no secuenciales
        'label': tf.io.FixedLenFeature([], tf.int64)
    }
    
    example = tf.io.parse_single_example(example, feature_description)
    
    # Las características secuenciales deben ser reformateadas en una secuencia (sequence_length, num_seq_features)
    sequence = tf.reshape(example['sequence'], (sequence_length, num_seq_features))
    
    # Las características no secuenciales se mantienen como un vector
    non_seq_features = example['non_seq_features']
    
    label = example['label']
    
    return (sequence, non_seq_features), label

# Función para combinar múltiples archivos TFRecord
def create_dataset_from_multiple_tfrecords(tfrecord_files, batch_size):
    datasets = [create_dataset_from_tfrecord(file, batch_size) for file in tfrecord_files]
    combined_dataset = datasets[0]
    for dataset in datasets[1:]:
        combined_dataset = combined_dataset.concatenate(dataset)
    return combined_dataset

# Extraer el número del chunk de los nombres de archivos
def extract_chunk_number(filename):
    match = re.search(r'chunk_(\d+)', filename)
    if match:
        return int(match.group(1))
    return -1

# Preguntar si se desea continuar con el entrenamiento
create_model = True
if os.path.exists('out/lstm/last_lstm_model.keras'):
    print("[INFO] Se ha encontrado un modelo previamente entrenado.")
    print("[INFO] ¿Desea continuar el entrenamiento del modelo? (s/n)")
    answer = input()
    if answer.lower() == 's':
        create_model = False

# Definir el modelo bajo estrategia distribuida
if create_model:
    with strategy.scope():
        # Crear y compilar el modelo LSTM
        # Definir las características de entrada de la secuencia (sin incluir sessionId)
        # Entrada de las variables secuenciales (21 características)
        input_features = Input(shape=(sequence_length, num_seq_features), name='seq_input')

        # Asignar mayor peso a BPM
        weights = tf.constant([2 if i == bpm_index else 1.0 for i in range(num_seq_features)], dtype=tf.float32)
        weights_matrix = tf.linalg.diag(weights)
        x_weighted = tf.matmul(input_features, weights_matrix)

        # Procesamiento de las variables secuenciales con LSTM
        x_seq = TimeDistributed(Dense(32, activation='relu'))(x_weighted)
        x_seq = Bidirectional(LSTM(128, activation='tanh', return_sequences=True))(x_seq)  # Mantener las secuencias
        x_seq = BatchNormalization()(x_seq)
        x_seq = LSTM(65, activation='tanh', return_sequences=False)(x_seq)  # Segunda capa LSTM
        x_seq = BatchNormalization()(x_seq)
        x_seq = Dropout(0.2)(x_seq)

        # Entrada de las variables no secuenciales (como Age_Binned, etc.)
        input_non_seq = Input(shape=(num_non_seq_features,), name='non_seq_input')

        # Procesamiento de las variables no secuenciales con capas densas
        x_non_seq = Dense(16, activation='relu', kernel_regularizer=l2(0.001))(input_non_seq)

        # Combinar las dos ramas de procesamiento (sec y no sec)
        attention_layer = Attention()([x_seq, x_seq])
        x_combined = Concatenate(axis=-1)([attention_layer, x_non_seq])
        x_combined = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(x_combined)
        x = Dropout(0.2)(x_combined)
        # Capa final y salida
        output = Dense(3, activation='softmax')(x)

        # Crear el modelo
        model = tf.keras.models.Model(inputs=[input_features, input_non_seq], outputs=output)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Resumen del modelo
        model.summary()
else:
    # Cargar el modelo previamente entrenado
    model = tf.keras.models.load_model('out/lstm/last_lstm_model.keras')

# Procesar validación
test_tfrecord_files = [f for f in os.listdir('out/lstm/data') if f.startswith("test_chunk") and f.endswith('.tfrecord')]
test_tfrecord_files = sorted(test_tfrecord_files, key=extract_chunk_number)

# Procesar entrenamiento
train_tfrecord_files = [f for f in os.listdir('out/lstm/data') if f.startswith("train_chunk") and f.endswith('.tfrecord')]
train_tfrecord_files = sorted(train_tfrecord_files, key=extract_chunk_number)

# Asegurarse de procesar el buffer al final
if 'train_chunk_buffer.tfrecord' in train_tfrecord_files:
    train_tfrecord_files.remove('train_chunk_buffer.tfrecord')
    train_tfrecord_files.append('train_chunk_buffer.tfrecord')
if 'test_chunk_buffer.tfrecord' in test_tfrecord_files:
    test_tfrecord_files.remove('test_chunk_buffer.tfrecord')
    test_tfrecord_files.append('test_chunk_buffer.tfrecord')

# Crear datasets con múltiples chunks
test_dataset = create_dataset_from_multiple_tfrecords(test_tfrecord_files, batch_size)

# Callbacks para detener el entrenamiento temprano y guardar el mejor modelo
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=0.0001, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(filepath='out/lstm/best_lstm_model', save_weights_only=False, save_best_only=True, monitor='val_loss'),
    tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: model.evaluate(test_dataset) if epoch % 3 == 0 else None),
    tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: model.save('out/lstm/last_lstm_model.keras'))
    #tf.keras.callbacks.TensorBoard(log_dir='out/lstm/logs/profile', profile_batch=0) # El perfilado empieza en el primer batch
]

chunk_number = 1
if not create_model:
    print("[INFO] Ingrese el chunk a partir del cual desea continuar el entrenamiento:")
    chunk_number = int(input())

train_tfrecord_files = train_tfrecord_files[chunk_number - 1:]

# Entrenamiento por chunks
for chunk_count, tfrecord_file in enumerate(train_tfrecord_files, 1):
    print(f"[INFO] Entrenando modelo con el chunk {chunk_number + chunk_count - 1} de {len(train_tfrecord_files) + chunk_number - 1}")
    train_dataset = create_dataset_from_tfrecord(tfrecord_file, batch_size)

    # Determinar steps_per_epoch dinámicamente para evitar cargar todos los datos
    print("sum de elements de train dataset: ", sum(1 for _ in train_dataset))
    print("batch_size: ", batch_size)
    steps_per_epoch = sum(1 for _ in train_dataset)
    validation_steps = sum(1 for _ in test_dataset)

    print(f"[INFO] steps_per_epoch: {steps_per_epoch}, validation_steps: {validation_steps}")

    # Entrenamiento del modelo
    model.fit(
        train_dataset.repeat(),
        epochs=epochs,
        validation_data=test_dataset,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=1
    )

    # Liberar memoria del chunk procesado
    del train_dataset
    gc.collect()  # Llamar al recolector de basura para liberar memoria

# Guardar el modelo entrenado
model.save('out/lstm/last_lstm_model.keras', save_format='keras')
print("[INFO] Entrenamiento completado y modelo guardado.")
