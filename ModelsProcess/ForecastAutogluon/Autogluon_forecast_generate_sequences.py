import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

df = pd.read_parquet('data/LSTM.parquet')

prubeas = False
if prubeas:
    # reducir el dataset para pruebas
    # Muestrear un porcentaje de los grupos por 'sessionId' para mantener secuencias completas

    # Paso 1: Seleccionar un porcentaje de los sessionId
    sampled_session_ids = df['sessionId'].drop_duplicates().sample(frac=0.1)  # Aquí 0.1 es el 10% de los sessionId

    # Filtrar el DataFrame para solo incluir los sessionId seleccionados
    df_sampled_sessions = df[df['sessionId'].isin(sampled_session_ids)]

    # Paso 2: Para cada sessionId seleccionado, seleccionar un porcentaje de las primeras filas
    # En este caso tomamos el 30% de las primeras filas de cada sessionId (cambiar el valor de frac según lo necesites)
    frac_filas_por_sesion = 0.8  # Seleccionar el 30% de las primeras filas en cada sesión

    # Usamos groupby y apply para tomar el porcentaje deseado de las primeras filas de cada grupo
    df_final = df_sampled_sessions.groupby('sessionId').apply(lambda x: x.head(int(len(x) * frac_filas_por_sesion)))

    # Resetear el índice ya que groupby y apply pueden desorganizar el índice
    df_final = df_final.reset_index(drop=True)

    # Mostrar el DataFrame resultante
    df = df_final

# cantidad de filas en el dataframe
print(f"La cantidad de filas en el dataframe es: {df.shape[0]}")

print(f"Cantidad de sessionId unicos: {df['sessionId'].nunique()}")

# columnas del dataframe
print(f"Las columnas del dataframe son: {df.columns}")

# Obtener los sessionIds únicos
session_ids = df['sessionId'].unique()

# Dividir los sessionIds en 70% entrenamiento y 30% conjunto temporal (prueba + validación)
train_ids, validation_ids = train_test_split(session_ids, test_size=0.2, random_state=42)

# Crear los DataFrames basados en los sessionId correspondientes
train_df = df[df['sessionId'].isin(train_ids)]
validation_df = df[df['sessionId'].isin(validation_ids)]

# ---- Aplica estandarización y normalización a las variables numéricas del dataset de entrenamiento ----

# Identificar las columnas categóricas, numéricas y las de rango fijo
categorical_columns = ['Age_Binned_<25', 'Age_Binned_25_40', 'Age_Binned_40_60', 'Age_Binned_>60']
# Variables continuas para el StandardScaler
numerical_columns = [
    'height', 'sleep_duration',
    'Interval', 'BPM', 'BPM_Mean_Acc', 'BPM_Var_Acc',
    'BPM_Std_Acc', 'BPM_Diff', 'BPM_Acceleration', 'BPM_Mean_Diff',
    'Time_SleepStage', 'SleepStage_Changes', 'BPM_Trend'
]
# Variables de rango fijo para el MinMaxScaler
fixed_range_features = ['fatigue', 'stress', 'sleep_quality']

# Normalizar y estandarizar las variables
sscaler = StandardScaler()
mmscaler = MinMaxScaler()

# Aplicar las transformaciones
train_df[numerical_columns] = sscaler.fit_transform(train_df[numerical_columns])
train_df[fixed_range_features] = mmscaler.fit_transform(train_df[fixed_range_features])
train_df[categorical_columns] = train_df[categorical_columns].astype('category')

# Guardar el scaler
joblib.dump(sscaler, 'out/autogluonforecast/Autogluon_standard_scaler.pkl')
joblib.dump(mmscaler, 'out/autogluonforecast/Autogluon_minmax_scaler.pkl')

validation_df[numerical_columns] = sscaler.transform(validation_df[numerical_columns])
validation_df[fixed_range_features] = mmscaler.transform(validation_df[fixed_range_features])
validation_df[categorical_columns] = validation_df[categorical_columns].astype('category')

sequence_length = 50

# Función optimizada para generar secuencias con sequenceId y ordenar por 'Time'
def create_sequences(df, sequence_length):
    sequences = []  # Lista para almacenar las secuencias
    sequence_id = 0  # Inicializar el contador para sequenceId
    
    # Iterar por cada grupo de sessionId
    for session_id, group in tqdm(df.groupby('sessionId'), desc='Creating sequences'):
        # Ordenar el grupo por la columna 'Time'
        group = group.sort_values(by='Time').to_numpy()  # Convertir el grupo en un array de NumPy
        
        # Crear secuencias de longitud 'sequence_length' para cada sessionId
        for i in range(0, len(group) - sequence_length + 1, 50):  # Incrementar el paso en 8
            sequence = group[i:i + sequence_length]  # Extraer la secuencia
            sequences.append(np.column_stack((sequence, np.full((sequence_length,), sequence_id))))  # Agregar sequenceId
            sequence_id += 1  # Incrementar el sequenceId para la siguiente secuencia
    
    # Concatenar todas las secuencias en un solo array
    all_sequences = np.vstack(sequences)
    
    # Convertir el array de vuelta a DataFrame
    columns = df.columns.tolist() + ['sequenceId']  # Nombres de las columnas
    return pd.DataFrame(all_sequences, columns=columns)

train_df_sequences = create_sequences(train_df, sequence_length)
validation_df_sequences = create_sequences(validation_df, sequence_length)

# drop the sessionId column
train_df_sequences = train_df_sequences.drop(columns=['sessionId'])
validation_df_sequences = validation_df_sequences.drop(columns=['sessionId'])

print(train_df_sequences[['Interval', 'Time', 'BPM', 'SleepStage', 'sequenceId', 'Age_Binned_40_60', 'BPM_Mean_Acc']].head(n=50).to_markdown())

# Guardar los datasets
train_df_sequences.to_parquet('data/Autogluon_sequences_train_80.parquet')
validation_df_sequences.to_parquet('data/Autogluon_sequences_validation_20.parquet')
