from autogluon.tabular import TabularPredictor
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np
import pyarrow.parquet as pq
import logging
import gc
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

# Configurar el logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mostrar todas las filas y columnas
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)  # Evita dividir las columnas en varias líneas
pd.set_option('display.max_colwidth', None)  # Muestra el contenido completo de cada columna

logger.info('Cargando el TabularPredictor...')
# Cargar el predictor desde el directorio
predictor = TabularPredictor.load(f'out/autogluon/model/')

# Inicializar listas para recolectar predicciones y etiquetas verdaderas
all_Y_true = []
all_Y_pred = []

label = 'SleepStage'

# Identificar las columnas categóricas, numéricas y las de rango fijo
categorical_columns = ['Age_Binned_<25', 'Age_Binned_25_40', 'Age_Binned_40_60', 'Age_Binned_>60']
# Variables continuas para el StandardScaler
numerical_columns = [
    'height', 'sleep_duration',
    'Interval', 'Time', 'BPM', 'BPM_Mean_Acc', 'BPM_Var_Acc',
    'BPM_Std_Acc', 'BPM_Diff', 'BPM_Acceleration', 'BPM_Mean_Diff',
    'Time_SleepStage', 'SleepStage_Changes', 'BPM_Trend',
    'BPM_Lag_1', 'BPM_Diff_Lag_1', 'BPM_Acceleration_Lag_1',
    'BPM_Mean_Diff_Lag_1', 'BPM_Mean_Acc_Lag_1', 'BPM_Trend_Lag_1',
    'Time_SleepStage_Lag_1', 'BPM_Lag_2', 'BPM_Diff_Lag_2',
    'BPM_Acceleration_Lag_2', 'BPM_Mean_Diff_Lag_2', 'BPM_Mean_Acc_Lag_2',
    'BPM_Trend_Lag_2', 'Time_SleepStage_Lag_2', 'BPM_Lag_3',
    'BPM_Diff_Lag_3', 'BPM_Acceleration_Lag_3', 'BPM_Mean_Diff_Lag_3',
    'BPM_Mean_Acc_Lag_3', 'BPM_Trend_Lag_3', 'Time_SleepStage_Lag_3',
    'BPM_Lag_4', 'BPM_Diff_Lag_4', 'BPM_Acceleration_Lag_4',
    'BPM_Mean_Diff_Lag_4', 'BPM_Mean_Acc_Lag_4', 'BPM_Trend_Lag_4',
    'Time_SleepStage_Lag_4', 'BPM_Lag_5', 'BPM_Diff_Lag_5',
    'BPM_Acceleration_Lag_5', 'BPM_Mean_Diff_Lag_5', 'BPM_Mean_Acc_Lag_5',
    'BPM_Trend_Lag_5', 'Time_SleepStage_Lag_5', 'BPM_rolling_mean_5',
    'BPM_rolling_max_5', 'BPM_rolling_min_5', 'BPM_rolling_std_5',
    'BPM_rolling_range_5'
]
# Variables de rango fijo para el MinMaxScaler
fixed_range_features = ['fatigue', 'stress', 'sleep_quality']


logger.info('Cargando el scaler...')
# Cargar el scaler
sscaler = joblib.load('out/autogluon/Autogluon_standard_scaler.pkl')
mmscaler = joblib.load('out/autogluon/Autogluon_minmax_scaler.pkl')
pca = joblib.load('out/autogluon/Autogluon_PCA.pkl')

# Leer el archivo Parquet en batches
logger.info('Leyendo el archivo Parquet en batches...')
parquet_file = pq.ParquetFile('data/Autogluon_validation_20.parquet')

batch_size = 100000  # Ajusta el tamaño del batch según tu memoria disponible
total_rows = parquet_file.metadata.num_rows
logger.info(f'Número total de filas en el archivo Parquet: {total_rows}')
num_batches = (total_rows + batch_size - 1) // batch_size
logger.info(f'Número de batches: {num_batches}')

batch_number = 0

# Variables para muestrear datos para feature importance y leaderboard
sample_size = 500000  # Ajusta el tamaño de la muestra según tus necesidades
sampled_rows = 0
sampled_X = []
sampled_Y = []

for batch in parquet_file.iter_batches(batch_size=batch_size):
    batch_number += 1
    logger.info(f'Procesando batch {batch_number}/{num_batches}...')
    df_batch = batch.to_pandas()

    #print(f"Tamano del batch: {df_batch.shape}")
    
    # Limpiar los datos
    df_clean = df_batch.replace([np.inf, -np.inf], np.nan).dropna()

    #print(f"Tamano del batch limpio: {df_clean.shape}")

    # Verificar si hay valores infinitos en la etiqueta y características
    infinite_labels = df_clean[np.isinf(df_clean[label])]
    if not infinite_labels.empty:
        logger.warning("Hay valores infinitos en la columna de etiquetas:")
        logger.warning(infinite_labels)

    #print(f"Tamano del batch limpio de infinitos: {df_clean.shape}")

    # Separar características y etiquetas
    X_batch = df_clean.drop(columns=[label])
    Y_batch = df_clean[label]

    #print(f"Tamano de X_batch: {X_batch.shape} y Y_batch: {Y_batch.shape}")
    
    # Asegurar consistencia
    if len(X_batch) != len(Y_batch):
        logger.warning(f'Numero inconsistente de muestras en X y Y en el batch {batch_number}')
        continue  # Saltar este batch
    
    # Normalizar las variables (usar transform en lugar de fit_transform)
    scaled_X_batch = X_batch.copy()
    scaled_X_batch[numerical_columns] = sscaler.transform(X_batch[numerical_columns])
    scaled_X_batch[fixed_range_features] = mmscaler.transform(X_batch[fixed_range_features])
    scaled_X_batch[categorical_columns] = X_batch[categorical_columns].astype('category')
    scaled_df_batch = pd.DataFrame(scaled_X_batch, columns=X_batch.columns)
    #print(f"Tamano de scaled_df_batch: {scaled_df_batch.shape}")
    num_columns_pca = [
        'BPM', 'BPM_Mean_Acc', 'BPM_Var_Acc',
        'BPM_Std_Acc', 'BPM_Diff', 'BPM_Acceleration', 'BPM_Mean_Diff',
        'Time_SleepStage', 'SleepStage_Changes', 'BPM_Trend',
        'BPM_Lag_1', 'BPM_Diff_Lag_1', 'BPM_Acceleration_Lag_1',
        'BPM_Mean_Diff_Lag_1', 'BPM_Mean_Acc_Lag_1', 'BPM_Trend_Lag_1',
        'Time_SleepStage_Lag_1', 'BPM_Lag_2', 'BPM_Diff_Lag_2',
        'BPM_Acceleration_Lag_2', 'BPM_Mean_Diff_Lag_2', 'BPM_Mean_Acc_Lag_2',
        'BPM_Trend_Lag_2', 'Time_SleepStage_Lag_2', 'BPM_Lag_3',
        'BPM_Diff_Lag_3', 'BPM_Acceleration_Lag_3', 'BPM_Mean_Diff_Lag_3',
        'BPM_Mean_Acc_Lag_3', 'BPM_Trend_Lag_3', 'Time_SleepStage_Lag_3',
        'BPM_Lag_4', 'BPM_Diff_Lag_4', 'BPM_Acceleration_Lag_4',
        'BPM_Mean_Diff_Lag_4', 'BPM_Mean_Acc_Lag_4', 'BPM_Trend_Lag_4',
        'Time_SleepStage_Lag_4', 'BPM_Lag_5', 'BPM_Diff_Lag_5',
        'BPM_Acceleration_Lag_5', 'BPM_Mean_Diff_Lag_5', 'BPM_Mean_Acc_Lag_5',
        'BPM_Trend_Lag_5', 'Time_SleepStage_Lag_5', 'BPM_rolling_mean_5',
        'BPM_rolling_max_5', 'BPM_rolling_min_5', 'BPM_rolling_std_5',
        'BPM_rolling_range_5'
    ]
    pca_X_batch = scaled_df_batch[num_columns_pca]
    pca_X_batch = pca.transform(pca_X_batch)
    assert not np.isnan(pca_X_batch).any(), "Hay valores NaN después de aplicar PCA"
    assert not np.isinf(pca_X_batch).any(), "Hay valores infinitos después de aplicar PCA"
    assert len(pca_X_batch) == len(Y_batch), f"Inconsistencia en el tamaño después de PCA. Características: {len(pca_X_batch)}, Etiquetas: {len(Y_batch)}"

    #print(f"Tamano de pca_X_batch: {pca_X_batch.shape}")
    pca_df = pd.DataFrame(pca_X_batch, columns=[f'PCA_{i+1}' for i in range(pca.n_components_)])
    pca_df = pca_df.reset_index(drop=True)
    #print(f"Tamano de pca_df: {pca_df.shape}")

    # Eliminar las columnas que fueron transformadas por PCA
    # scaled_df_batch tiene 61 columnas al principio, pero estamos eliminando las 51 transformadas por PCA
    scaled_df_batch_no_pca = scaled_df_batch.drop(columns=num_columns_pca)
    scaled_df_batch_no_pca = scaled_df_batch_no_pca.reset_index(drop=True)
    #print(f"Tamano de scaled_df_batch_no_pca: {scaled_df_batch_no_pca.shape}")

    # Verificar los índices antes de concatenar
    #print("Índices de scaled_df_batch_no_pca (primeros 5):", scaled_df_batch_no_pca.index[:5])
    #print("Índices de pca_df (primeros 5):", pca_df.index[:5])

    #print(f"Head de scaled_df_batch_no_pca: {scaled_df_batch_no_pca.head()}")
    #print(f"Head de pca_df: {pca_df.head()}")

    # Ahora concatenamos las 10 componentes principales de PCA (pca_df)
    # scaled_df_batch_no_pca debería tener las columnas no transformadas por PCA
    transformed_df_batch = pd.concat([scaled_df_batch_no_pca, pca_df], axis=1)
    #print(f"Tamano de transformed_df_batch: {transformed_df_batch.shape}")

    # Verificar que el número de columnas sea el correcto (debería ser 61 nuevamente)
    assert transformed_df_batch.shape == (len(Y_batch), len(X_batch.columns) - len(num_columns_pca) + pca.n_components_), \
        f"El numero de filas o columnas no coincide. Esperado: ({len(Y_batch)}, {len(X_batch.columns) - len(num_columns_pca) + pca.n_components_}), Obtenido: {transformed_df_batch.shape}"

    # Verificar que el número de filas en el DataFrame coincide con el número de etiquetas antes de predecir
    assert len(transformed_df_batch) == len(Y_batch), f"El numero de filas en las caracteristicas ({len(transformed_df_batch)}) no coincide con las etiquetas ({len(Y_batch)})"

    # Realizar predicciones
    y_pred_batch = predictor.predict(transformed_df_batch)

    # Verificar que las dimensiones sean consistentes
    if len(Y_batch) != len(y_pred_batch):
        logger.warning(f'Numero inconsistente de muestras en el batch {batch_number}. Y_true: {len(Y_batch)}, Y_pred: {len(y_pred_batch)}')
        logger.warning(f'Numero inconsistente de muestras en el batch {batch_number}.')
        logger.warning(f'Y_true: {len(Y_batch)}, Y_pred: {len(y_pred_batch)}')
        logger.warning(f'Primeras filas del DataFrame de caracteristicas:\n{transformed_df_batch.head()}')
        continue  # Saltar este batch si no coinciden

    # Recolectar predicciones y etiquetas verdaderas
    all_Y_true.extend(Y_batch.tolist())
    all_Y_pred.extend(y_pred_batch.tolist())
    
    # Muestrear datos para feature importance y leaderboard
    if sampled_rows < sample_size:
        remaining_sample = sample_size - sampled_rows
        if len(X_batch) <= remaining_sample:
            sampled_X.append(transformed_df_batch)
            sampled_Y.append(Y_batch)
            sampled_rows += len(X_batch)
        else:
            sampled_X.append(transformed_df_batch.iloc[:remaining_sample])
            sampled_Y.append(Y_batch.iloc[:remaining_sample])
            sampled_rows += remaining_sample
    
    logger.info(f'Terminado el procesamiento del batch {batch_number}/{num_batches}')

    del df_clean, X_batch, Y_batch, scaled_X_batch, pca_X_batch, transformed_df_batch
    gc.collect()

# Convertir las listas de predicciones y etiquetas a Series de pandas
Y_true = pd.Series(all_Y_true, name='TrueLabel')
Y_pred = pd.Series(all_Y_pred, name='PredictedLabel')

logger.info('Evaluando el desempeño...')
# Evaluar el desempeño
perf = predictor.evaluate_predictions(y_true=Y_true, y_pred=Y_pred, auxiliary_metrics=True)

print("\nDesempeño:")
print(perf)

print("\nReporte de clasificación:")
print(classification_report(Y_true, Y_pred))

# Preparar datos para feature importance y leaderboard si se muestrearon datos
if sampled_rows > 0:
    X_sampled = pd.concat(sampled_X, ignore_index=True)
    Y_sampled = pd.concat(sampled_Y, ignore_index=True)
    
    total_df = X_sampled.copy()
    total_df[label] = Y_sampled.reset_index(drop=True)
    
    logger.info('Calculando la importancia de las características...')
    # Importancia de características
    fi = predictor.feature_importance(data=total_df, feature_stage='transformed')
    print("\nImportancia de características:")
    print(fi)
    
    logger.info('Generando el leaderboard...')
    # Leaderboard
    print("\nLeaderboard:")
    predictor.leaderboard(extra_info=True)
    print("\nLeaderboard (datos muestreados):")
    predictor.leaderboard(total_df)
else:
    logger.warning('No se muestrearon datos para feature importance y leaderboard.')

# Calcular métricas
accuracy = accuracy_score(Y_true, Y_pred)
mae = mean_absolute_error(Y_true, Y_pred)
mse = mean_squared_error(Y_true, Y_pred)

print("Accuracy:", accuracy)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)

# Matriz de confusión
logger.info('Generando la matriz de confusión...')
conf_matrix = confusion_matrix(Y_true, Y_pred)
print("\nMatriz de confusión:")
print(conf_matrix)

# Visualizar la matriz de confusión
plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix, annot=True, cmap="YlGnBu", fmt='g')
plt.ylabel('Etiqueta verdadera')
plt.xlabel('Etiqueta predicha')
plt.title('Matriz de confusión')
plt.show()
