import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from tqdm import tqdm
import os

# Directorio de salida para los archivos resampleados
output_dir = 'out/autogluonforecast/resampled_sequences/'
os.makedirs(output_dir, exist_ok=True)

# Parámetros para la escritura por lotes
batch_size = 500000  # Número de secuencias que se acumulan antes de escribir al disco
batch_counter = 0  # Contador para controlar los lotes
batch_data = []  # Lista temporal para acumular los datos
parquet_index = 0  # Índice para los nombres de archivo

# Verificar si ya existen archivos parquet resampleados
if len(os.listdir(output_dir)) > 0:
    print("Archivos resampleados ya existen. Saltando el resampling.")
else:
    # Cargar el dataset original
    df_sequences = pd.read_parquet('data/Autogluon_sequences_train_80.parquet')

    # Optimización de tipos de datos
    df_sequences['Time'] = pd.to_numeric(df_sequences['Time'], downcast='integer')
    df_sequences['SleepStage'] = df_sequences['SleepStage'].astype('int8')

    # Convertir 'Time' a datetime
    df_sequences['timestamp'] = pd.to_datetime(df_sequences['Time'], unit='s')

    # Chequear duplicados y eliminarlos
    df_sequences = df_sequences.drop_duplicates(subset=['sequenceId', 'timestamp'])

    # Agrupar por sequenceId
    grouped = df_sequences.groupby('sequenceId')

    # Guardar los grupos resampleados en lotes en lugar de escribir uno por uno
    for idx, (seq_id, group) in tqdm(enumerate(grouped), "Resampling sequences", total=len(grouped)):
        if len(group) >= 50:  # Asegurarse de que el grupo no esté vacío
            # Resamplear por 'timestamp' con un intervalo de 5 segundos
            group_resampled = group.set_index('timestamp').resample('5S').ffill().reset_index()
            group_resampled['sequenceId'] = seq_id  # Reagregar sequenceId

            # Acumular las secuencias en la lista temporal
            batch_data.append(group_resampled)
            batch_counter += 1

        # Cuando alcanzamos el tamaño de lote, escribimos a un archivo Parquet
        if batch_counter >= batch_size:
            df_batch = pd.concat(batch_data)
            output_file = os.path.join(output_dir, f"resampled_sequences_batch_{parquet_index}.parquet")
            df_batch.to_parquet(output_file, engine='pyarrow')
            print(f"Escrito el archivo {output_file} con {batch_counter} secuencias.")

            # Resetear los acumuladores
            batch_data = []
            batch_counter = 0
            parquet_index += 1

    # Si quedan secuencias no escritas después del loop, escribirlas
    if batch_data:
        df_batch = pd.concat(batch_data)
        output_file = os.path.join(output_dir, f"resampled_sequences_batch_{parquet_index}.parquet")
        df_batch.to_parquet(output_file, engine='pyarrow')
        print(f"Escrito el archivo final {output_file} con {batch_counter} secuencias.")

# Cargar los archivos resampleados existentes para el siguiente paso
resampled_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.parquet')]
df_sequences_resampled = pd.concat([pd.read_parquet(f) for f in resampled_files])

# If we have valid resampled data, proceed
if not df_sequences_resampled.empty:
    # Ensure the TimeSeriesDataFrame is created with consistent timestamps
    time_series_data = TimeSeriesDataFrame(df_sequences_resampled, id_column="sequenceId", timestamp_column="timestamp")

    # Define and fit the predictor with a reduced prediction length if necessary
    predictor = TimeSeriesPredictor(target='SleepStage', path='out/autogluonforecast/model/', prediction_length=12)

    # Fit the predictor
    predictor.fit(time_series_data, presets='best_quality', time_limit=28800)

    # Print leaderboard
    leaderboard = predictor.leaderboard()
    print(leaderboard)
else:
    print("No data available for model training.")
