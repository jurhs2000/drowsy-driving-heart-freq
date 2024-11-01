import pandas as pd
from sklearn.model_selection import train_test_split

# Supongamos que ya tienes el DataFrame cargado con las secuencias agrupadas por 'sessionId'
df = pd.read_parquet('data/LSTM.parquet')

# Obtener los sessionIds únicos
session_ids = df['sessionId'].unique()

# Dividir los sessionIds en 70% entrenamiento y 30% conjunto temporal (prueba + validación)
train_ids, temp_ids = train_test_split(session_ids, test_size=0.20, random_state=42)

# Dividir el conjunto temporal en 50% prueba y 50% validación (lo que da 15% para cada uno)
test_ids, validation_ids = train_test_split(temp_ids, test_size=0.50, random_state=42)

# Crear los DataFrames basados en los sessionId correspondientes
train_df = df[df['sessionId'].isin(train_ids)]
test_df = df[df['sessionId'].isin(test_ids)]
validation_df = df[df['sessionId'].isin(validation_ids)]

# Guardar los DataFrames resultantes como archivos Parquet
train_df.to_parquet('data/LSTM_train_80.parquet')
test_df.to_parquet('data/LSTM_test_10.parquet')
validation_df.to_parquet('data/LSTM_validation_10.parquet')

# Imprimir la cantidad de sesiones en cada conjunto para verificar
print(f"Train set session count: {len(train_ids)}")
print(f"Test set session count: {len(test_ids)}")
print(f"Validation set session count: {len(validation_ids)}")
