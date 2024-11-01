from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm
import warnings
import logging
import os
import gc
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report

# Configuración y carga de datos (sin cambios)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

resampled_file = 'out/autogluonforecast/Autogluon_sequences_validation_resampled.parquet'

if os.path.isfile(resampled_file):
    print("El archivo resampleado para validación ya existe. Cargando...")
    df_validation_sequences_resampled = pd.read_parquet(resampled_file)
else:
    df_validation_sequences = pd.read_parquet('data/Autogluon_sequences_validation_20.parquet')
    pruebas = False
    if pruebas:
        # usar el 10% de los datos para pruebas
        session_ids = df_validation_sequences['sequenceId'].unique()
        sampled_session_ids = pd.Series(session_ids).sample(frac=0.1)
        df_validation_sequences = df_validation_sequences[df_validation_sequences['sequenceId'].isin(sampled_session_ids)]
    df_validation_sequences['Time'] = pd.to_numeric(df_validation_sequences['Time'], downcast='integer')
    df_validation_sequences['SleepStage'] = df_validation_sequences['SleepStage'].astype('int8')
    df_validation_sequences['timestamp'] = pd.to_datetime(df_validation_sequences['Time'], unit='s')
    df_validation_sequences = df_validation_sequences.drop_duplicates(subset=['sequenceId', 'timestamp'])

    resampled_list = []
    grouped = df_validation_sequences.groupby('sequenceId')

    for seq_id, group in tqdm(grouped, "Resampling validation sequences", total=len(grouped)):
        if len(group) >= 50:
            group_resampled = group.set_index('timestamp').resample('5S').ffill().reset_index()
            group_resampled['sequenceId'] = seq_id
            resampled_list.append(group_resampled)

    if resampled_list:
        df_validation_sequences_resampled = pd.concat(resampled_list)
        df_validation_sequences_resampled.to_parquet(resampled_file)
    else:
        print("No valid resampled data. Check sequenceId groups.")
        df_validation_sequences_resampled = pd.DataFrame()

df_validation_sequences_resampled = df_validation_sequences_resampled.groupby('sequenceId').filter(lambda x: len(x) >= 50)

if not df_validation_sequences_resampled.empty:
    validation_data = TimeSeriesDataFrame(df_validation_sequences_resampled, id_column="sequenceId", timestamp_column="timestamp")
    print("Validation data shape:", validation_data.shape)
    predictor = TimeSeriesPredictor.load('out/autogluonforecast/model/')
    print("Predictor loaded successfully.")
    evaluation_score = predictor.evaluate(validation_data, model='TemporalFusionTransformer')
    print("Evaluation score:", evaluation_score)

    feature_importance = predictor.feature_importance(data=validation_data, model='TemporalFusionTransformer')
    print("Feature importance:")
    print(feature_importance)

    unique_sequence_ids = df_validation_sequences_resampled['sequenceId'].nunique()
    grouped = list(df_validation_sequences_resampled.groupby('sequenceId'))

    def process_batch(batch):
        batch_results = []
        for seq_id, group in tqdm(batch, desc="Processing batch", leave=False):
            if len(group) > 0:
                group_time_series = validation_data.loc[seq_id]

                # Verificar que la columna 'timestamp' está presente
                if 'timestamp' not in group_time_series.columns:
                    group_time_series = group_time_series.reset_index()  # Convertir el índice a columnas
                    group_time_series['sequenceId'] = seq_id  # Re-agregar 'sequenceId' si es necesario

                # Crear MultiIndex si 'timestamp' no está en el índice
                group_time_series = group_time_series.set_index(['sequenceId', 'timestamp'])
                group_time_series.index.names = ['item_id', 'timestamp']  # Asegurar nombres correctos

                # Predicción
                # Usar la secuencia sin los últimos 12 valores para predecir los siguientes 12
                to_predict = group_time_series.iloc[:-12]
                forecast = predictor.predict(to_predict, model='TemporalFusionTransformer')
                
                # Obtener predicciones
                prediction = forecast['mean'].values[-12:].round()
                true_value = group_time_series['SleepStage'].values[-12:]

                batch_results.append((prediction, true_value))

        return batch_results

    # Procesar por lotes
    batch_size = 100
    results = []
    grouped = list(grouped)
    for i in tqdm(range(0, len(grouped), batch_size), "Processing batches", total=unique_sequence_ids // batch_size):
        batch = grouped[i:i + batch_size]
        results.extend(process_batch(batch))
        gc.collect()  # Recolección de basura para liberar memoria

    # Desempaquetar resultados y calcular métricas (sin cambios)
    predictions, true_values = [], []
    for pred, true in results:
        if pred is not None:
            predictions.extend(pred)
            true_values.extend(true)

    print("Number of predictions:", len(predictions))
    print("head predictions:", predictions[:10])
    print("Number of true values:", len(true_values))
    print("head true values:", true_values[:10])

    accuracy = accuracy_score(true_values, predictions)
    mae = mean_absolute_error(true_values, predictions)
    mse = mean_squared_error(true_values, predictions)

    print("Classification Report:")
    print(classification_report(true_values, predictions))

    print("Accuracy:", accuracy)
    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)
    
    cm = confusion_matrix(true_values, predictions)
    print("Confusion Matrix:")
    print(cm)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized Confusion Matrix:")
    print(cm)

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=['awake', 'drowsy', 'sleeping'], yticklabels=['awake', 'drowsy', 'sleeping'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

else:
    print("No valid resampled validation data available for evaluation.")
