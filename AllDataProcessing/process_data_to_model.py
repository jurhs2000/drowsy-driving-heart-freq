import numpy as np
import pandas as pd
from tqdm import tqdm

# Cargar los datos
df = pd.read_csv('data/df.csv')

# Eliminar la columna 'gender'
df = df.drop(columns=['gender'])

# map de los valores de 'SleepStage' 9 a 0
df['SleepStage'] = df['SleepStage'].replace({9: 0})

# Map de los valores de 'SleepStage' 3, 4 y 5 a 2
df['SleepStage'] = df['SleepStage'].replace({3: 2, 4: 2, 5: 2})

# Eliminar las filas con valores 'SleepStage' 6
df = df[df['SleepStage'] != 6]

# Crear un identificador único para cada sesión combinando 'id' y 'logId'
df['sessionId'] = df['id'].astype(str) + '_' + df['logId'].astype(str)

# Eliminar las columnas 'id' y 'logId'
df = df.drop(columns=['id', 'logId'])

# Codificación categórica en lugar de get_dummies
df['sessionId'] = df['sessionId'].astype('category').cat.codes

# Asegurar que 'sessionId' sea numérico
df['sessionId'] = pd.to_numeric(df['sessionId'], errors='coerce').fillna(0).astype(np.int32)

# Resetear la variable 'Interval' cada 15 minutos (900 segundos).
df['Interval_Reset'] = df.groupby('sessionId')['Time'].transform(lambda x: (x // 900).astype(np.int32))

# Reiniciar el contador de 'Interval' dentro de cada bloque de 15 minutos
df['Interval'] = df.groupby(['sessionId', 'Interval_Reset']).cumcount().astype(np.int32)

# Eliminar la columna 'Interval_Reset'
df.drop(columns='Interval_Reset', inplace=True)

# Agregar un promedio de ritmo cardiaco (BPM) acumulativo por sesión
df['BPM_Mean_Acc'] = df.groupby('sessionId')['BPM'].transform(lambda x: x.expanding().mean()).astype(np.float32)

# Agregar la varianza de ritmo cardiaco (BPM) acumulativa por sesión
df['BPM_Var_Acc'] = df.groupby('sessionId')['BPM'].transform(lambda x: x.expanding().var()).astype(np.float32)

# Agregar la desviación estándar de ritmo cardiaco (BPM) acumulativa por sesión
df['BPM_Std_Acc'] = df.groupby('sessionId')['BPM'].transform(lambda x: x.expanding().std()).astype(np.float32)

# Calcular la diferencia entre los valores de BPM entre intervalos consecutivos
df['BPM_Diff'] = df['BPM'].diff().astype(np.float32)

# Calcular la aceleración de BPM
df['BPM_Acceleration'] = df['BPM_Diff'].diff().astype(np.float32)

# Calcular la diferencia entre el ritmo cardíaco actual y el promedio acumulativo de BPM
df['BPM_Mean_Diff'] = (df['BPM'] - df['BPM_Mean_Acc']).astype(np.float32)

# Convertir la edad en intervalos por rangos y generar dummies
df['Age_Binned'] = pd.cut(df['age'], bins=[0, 25, 40, 60, 100], labels=['<25', '25_40', '40_60', '>60'])
df = pd.get_dummies(df, columns=['Age_Binned'])
df.drop(columns='age', inplace=True)

# Ajuste en las variables Time_SleepStage y SleepStage_Changes
df['SleepStage'] = df['SleepStage'].astype(np.int32)

# Detectar cambios en SleepStage por sesión
df['SleepStage_Change_Flag'] = df.groupby('sessionId')['SleepStage'].transform(lambda x: x.diff().ne(0).astype(np.int32))

# Calcular la diferencia de tiempo y reiniciar 'Time_SleepStage'
df['Time_Diff'] = df.groupby('sessionId')['Time'].diff().fillna(0).astype(np.float32)
df['Time_SleepStage'] = df.groupby(['sessionId', df['SleepStage_Change_Flag'].cumsum()])['Time_Diff'].cumsum().reset_index(drop=True).astype(np.float32)
df['Time_SleepStage'] = (df['Time_SleepStage'] + df.groupby('sessionId')['Time'].transform('first')).astype(np.float32)

# Contador de cambios de SleepStage
def reset_sleepstage_changes(group):
    group['SleepStage_Changes'] = group['SleepStage'].diff().ne(0).cumsum().fillna(0).astype(np.int32)
    group['SleepStage_Changes'] = group['SleepStage_Changes'] - group['SleepStage_Changes'].min()  # Reiniciar a 0
    return group

# Guardar la columna 'sessionId'
session_ids = df['sessionId']
# Realizar el groupby y aplicar la función de reseteo
df = df.groupby('sessionId').apply(reset_sleepstage_changes, include_groups=False).reset_index(drop=True)
# Restaurar la columna 'sessionId'
df['sessionId'] = session_ids.reset_index(drop=True)

# Eliminar las columnas auxiliares
df.drop(columns=['SleepStage_Change_Flag', 'Time_Diff'], inplace=True)

# Configuración de tqdm para trabajar con pandas
tqdm.pandas()
window_size = 5

# Cálculo de la tendencia de BPM
print("Calculando la tendencia de BPM...")
# Precalcular los valores necesarios para la fórmula de la pendiente
x = np.arange(window_size)
sum_x = np.sum(x)
sum_x_squared = np.sum(x**2)
def calculate_slope(y):
    # Calcular la pendiente de la regresión lineal de la ventana
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    numerator = window_size * sum_xy - sum_x * sum_y
    denominator = window_size * sum_x_squared - sum_x**2
    return numerator / denominator
df['BPM_Trend'] = df['BPM'].rolling(window=window_size).progress_apply(calculate_slope, raw=True).astype(np.float32)

# Conversión de tipos de variables
# Convertir a int las variables que deben ser enteras
for col in ['height', 'fatigue', 'sleep_duration', 'sleep_quality', 'stress', 'Interval', 'Time', 'BPM', 'SleepStage', 'sessionId', 'Time_SleepStage', 'SleepStage_Changes']:
    df[col] = df[col].ffill().bfill().astype(np.int32)
# Interpolar valores faltantes en las variables continuas
for col in ['BPM_Diff', 'BPM_Acceleration']:
    df[col] = df[col].interpolate(method='linear').astype(np.float32)
# Convertir a float con 2 decimales y asegurarse de que todo esté bien ajustado
for col in ['BPM_Mean_Acc', 'BPM_Var_Acc', 'BPM_Std_Acc', 'BPM_Mean_Diff', 'BPM_Trend']:
    df[col] = df[col].ffill().bfill().round(2).astype(np.float32)

# revisar si hay filas con valores nan
print("Revisando valores NaN...")
print(df.isnull().sum())

# Guardar en formato parquet
df.to_parquet('data/LSTM.parquet')

# Agregar variables lag para 'BPM' y 'BPM_Diff'
print("Agregando variables lag...")
lag_columns = ['BPM', 'BPM_Diff', 'BPM_Acceleration', 'BPM_Mean_Diff', 'BPM_Mean_Acc', 'BPM_Trend', 'Time_SleepStage']
for lag in range(1, 6):
    for col in lag_columns:
        df[f'{col}_Lag_{lag}'] = df.groupby('sessionId')[col].shift(lag).astype(np.float32).ffill().bfill()

print("Agregando promedios, máximos y mínimos móviles...")
rolling_stats = df['BPM'].rolling(window=window_size).agg(['mean', 'max', 'min', 'var', 'std'])
df[f'BPM_rolling_mean_{window_size}'] = rolling_stats['mean'].astype(np.float32).ffill().bfill()
df[f'BPM_rolling_max_{window_size}'] = rolling_stats['max'].astype(np.float32).ffill().bfill()
df[f'BPM_rolling_min_{window_size}'] = rolling_stats['min'].astype(np.float32).ffill().bfill()
df[f'BPM_rolling_std_{window_size}'] = rolling_stats['std'].astype(np.float32).ffill().bfill()

# Calcular el rango basado en las columnas de máximo y mínimo ya existentes
df[f'BPM_rolling_range_{window_size}'] = (df[f'BPM_rolling_max_{window_size}'] - df[f'BPM_rolling_min_{window_size}']).astype(np.float32).ffill().bfill()

# revisar si hay filas con valores nan
print("Revisando valores NaN...")
print(df.isnull().sum())

# Guardar el dataframe final
df.to_parquet('data/Autogluon.parquet')
