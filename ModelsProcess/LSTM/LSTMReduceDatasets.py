import pyarrow.parquet as pq
import pandas as pd
import numpy as np

# validation dataset
validation_file = pq.ParquetFile('data/LSTM_validation_15.parquet')
df = validation_file.read().to_pandas()

# Obtener una lista única de sessionId
unique_sessionIds = df['sessionId'].unique()

# Elegir un subconjunto de sessionIds. Por ejemplo, selecciona el % aleatoriamente.
selected_sessionIds = np.random.choice(unique_sessionIds, size=int(0.02 * len(unique_sessionIds)), replace=False)

# Filtrar el dataset para mantener solo las filas con los sessionIds seleccionados
filtered_df = df[df['sessionId'].isin(selected_sessionIds)]

# Comprobar el resultado
print(f"Tamaño original del dataset: {len(df)}")
print(f"Tamaño reducido del dataset: {len(filtered_df)}")

# Guardar o utilizar el dataframe reducido
filtered_df.to_parquet('data/LSTM_validation_reduced.parquet')  # Guardar si es necesario

# Train dataset
train_file = pq.ParquetFile('data/LSTM_train_70.parquet')
df = train_file.read().to_pandas()

# Obtener una lista única de sessionId
unique_sessionIds = df['sessionId'].unique()

# Elegir un subconjunto de sessionIds. Por ejemplo, selecciona el % aleatoriamente.
selected_sessionIds = np.random.choice(unique_sessionIds, size=int(0.02 * len(unique_sessionIds)), replace=False)

# Filtrar el dataset para mantener solo las filas con los sessionIds seleccionados
filtered_df = df[df['sessionId'].isin(selected_sessionIds)]

# Comprobar el resultado
print(f"Tamaño original del dataset: {len(df)}")
print(f"Tamaño reducido del dataset: {len(filtered_df)}")

# Guardar o utilizar el dataframe reducido
filtered_df.to_parquet('data/LSTM_train_reduced.parquet')  # Guardar si es necesario

# test dataset
test_file = pq.ParquetFile('data/LSTM_test_15.parquet')
df = test_file.read().to_pandas()

# Obtener una lista única de sessionId
unique_sessionIds = df['sessionId'].unique()

# Elegir un subconjunto de sessionIds. Por ejemplo, selecciona el % aleatoriamente.
selected_sessionIds = np.random.choice(unique_sessionIds, size=int(0.02 * len(unique_sessionIds)), replace=False)

# Filtrar el dataset para mantener solo las filas con los sessionIds seleccionados
filtered_df = df[df['sessionId'].isin(selected_sessionIds)]

# Comprobar el resultado
print(f"Tamaño original del dataset: {len(df)}")
print(f"Tamaño reducido del dataset: {len(filtered_df)}")

# Guardar o utilizar el dataframe reducido
filtered_df.to_parquet('data/LSTM_test_reduced.parquet')  # Guardar si es necesario
