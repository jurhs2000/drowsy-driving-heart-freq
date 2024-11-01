from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_parquet('data/Autogluon.parquet')

#df = df.sample(frac=0.1, random_state=42) # solo para pruebas

label = 'SleepStage'

# Eliminar filas con NaN o valores infinitos en las etiquetas o en las características
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# Verificar si hay valores no finitos en la columna de etiquetas
if df[label].isin([np.inf, -np.inf]).any():
    print("Hay valores infinitos en la columna de etiquetas.")

# Obtener los sessionIds únicos
session_ids = df['sessionId'].unique()

# Dividir los sessionIds en 70% entrenamiento y 30% conjunto temporal (prueba + validación)
train_ids, validation_ids = train_test_split(session_ids, test_size=0.2, random_state=42)

# Crear los DataFrames basados en los sessionId correspondientes
train_df = df[df['sessionId'].isin(train_ids)]
validation_df = df[df['sessionId'].isin(validation_ids)]

# Eliminar la columna 'sessionId' de los DataFrames
train_df = train_df.drop(columns=['sessionId'])
validation_df = validation_df.drop(columns=['sessionId'])

# ---- Aplica estandarización y normalización a las variables numéricas del dataset de entrenamiento ----

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

# Normalizar y estandarizar las variables
sscaler = StandardScaler()
mmscaler = MinMaxScaler()

# Aplicar las transformaciones
train_df[numerical_columns] = sscaler.fit_transform(train_df[numerical_columns])
train_df[fixed_range_features] = mmscaler.fit_transform(train_df[fixed_range_features])
train_df[categorical_columns] = train_df[categorical_columns].astype('category')

# Guardar el scaler
joblib.dump(sscaler, 'out/autogluon/Autogluon_standard_scaler.pkl')
joblib.dump(mmscaler, 'out/autogluon/Autogluon_minmax_scaler.pkl')

# ------ Realiza el analisis de componentes principales (PCA) al dataset de entrenamiento ------

# Mostrar todas las columnas y filas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Si quieres ver todo sin truncamientos en una sola fila
pd.set_option('display.max_colwidth', None)

print("revisando si existen valores infinitos en el dataset de entrenamiento")
print(train_df.isin([np.inf, -np.inf]).sum())

print("revisando si existen valores NaN en el dataset de entrenamiento")
print(train_df.isnull().sum())

print("revisando si las features fueron correctamente normalizadas y estandarizadas")
print(train_df.describe())
print("comparacion con el describe del dataset original")
print(df.describe())

# Aplicar PCA solo a las columnas numéricas
pca = PCA(n_components=0.9)  # Elegir el número de componentes principales

'''num_columns_pca = [
    'BPM', 'BPM_Mean_Acc', 'BPM_Var_Acc',
    'BPM_Std_Acc', 'BPM_Diff', 'BPM_Acceleration', 'BPM_Mean_Diff', 'BPM_Trend'
]'''
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

# Ajustar y transformar los datos

print("revisando tipo de datos antes de PCA")
print(train_df[num_columns_pca].dtypes)
print(train_df.shape)
print(train_df[num_columns_pca].var())
# Aplicar el PCA
pca_result = pca.fit_transform(train_df[num_columns_pca])

print("revisando tipo de datos despues de PCA")
print(pca_result.dtype)
print(pca_result.shape)
print(pca_result[:5])

# Crear un DataFrame con las nuevas columnas del PCA y reiniciar los índices
pca_columns = [f'PCA_{i+1}' for i in range(pca.n_components_)]
df_pca = pd.DataFrame(pca_result, columns=pca_columns).reset_index(drop=True)

# Asegurarse de que el DataFrame original también tenga índices alineados
train_df = train_df.reset_index(drop=True)

# Concatenar el resultado del PCA con el DataFrame original (sin las columnas numéricas originales)
df_clean_pca = pd.concat([train_df.drop(columns=num_columns_pca), df_pca], axis=1)

# Verificar si hay NaN
print(df_clean_pca.isnull().sum())

print(f"El número de componentes principales seleccionados es: {pca.n_components_}")
print(f"El porcentaje de varianza explicada es: {pca.explained_variance_ratio_.sum()}")
print(f"El porcentaje de varianza explicada por cada componente es: {pca.explained_variance_ratio_}")
print(f"Las columnas del DataFrame resultante son: {df_clean_pca.columns}")
print(f"Head del DataFrame resultante:\n{df_clean_pca.head(n=15)}")

# Obtener las cargas de los componentes (relación entre componentes y variables originales)
pca_loadings = pd.DataFrame(pca.components_.T, index=num_columns_pca, columns=pca_columns)

print("Cargas de los componentes:\n", pca_loadings)
# convertir en absolutos los valores de las cargas
pca_loadings = pca_loadings.abs()
# valores maximos de cada feature original todos los componentes
max_values = pca_loadings.max(axis=1)
# valores maximos de cada componente
max_values_component = pca_loadings.max(axis=0)
print("max_values", max_values)
print("max_values_component", max_values_component)


# Guardar el modelo PCA
joblib.dump(pca, 'out/autogluon/Autogluon_PCA.pkl')

# Guardar los DataFrames resultantes como archivos Parquet
df_clean_pca.to_parquet('data/Autogluon_train_80.parquet')
validation_df.to_parquet('data/Autogluon_validation_20.parquet')

# Imprimir la cantidad de sesiones en cada conjunto para verificar
print(f"Train set session count: {len(train_ids)}")
print(f"Validation set session count: {len(validation_ids)}")
