import numpy as np
import random
import pandas as pd
import json
import joblib
from autogluon.tabular import TabularDataset, TabularPredictor
from collections import deque
import asyncio
import websockets

# Definir una cola de tamaño fijo para almacenar los últimos 5 BPM
window_size = 5
bpm_window = deque(maxlen=window_size)

def is_valid_json(mystring):
    try:
        json.loads(mystring)
        return True
    except ValueError:
        return False

def calculate_trend(bpm_window):
    # Si no tenemos suficientes datos, devolvemos None
    if len(bpm_window) < window_size:
        return np.nan
    # Calcular la tendencia usando regresión lineal (pendiente)
    trend = np.polyfit(range(len(bpm_window)), bpm_window, 1)[0]
    return trend


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

def make_predictions(hr_sleep_df):
    # Predecir el SleepStage con el modelo de Autogluon
    # agarrar solo la ultima fila
    scaled_X_batch = hr_sleep_df.iloc[-1].copy()
    scaled_X_batch = scaled_X_batch.to_frame().T
    scaled_X_batch[numerical_columns] = sscaler.transform(scaled_X_batch[numerical_columns])
    scaled_X_batch[fixed_range_features] = mmscaler.transform(scaled_X_batch[fixed_range_features])
    scaled_X_batch[categorical_columns] = scaled_X_batch[categorical_columns].astype('category')
    scaled_df_batch = pd.DataFrame(scaled_X_batch, columns=scaled_X_batch.columns)
    pca_X_batch = scaled_df_batch[num_columns_pca]
    pca_X_batch = pca.transform(pca_X_batch)
    pca_df = pd.DataFrame(pca_X_batch, columns=[f'PCA_{i+1}' for i in range(pca.n_components_)])
    pca_df = pca_df.reset_index(drop=True)
    scaled_df_batch_no_pca = scaled_df_batch.drop(columns=num_columns_pca)
    scaled_df_batch_no_pca = scaled_df_batch_no_pca.reset_index(drop=True)
    transformed_df_batch = pd.concat([scaled_df_batch_no_pca, pca_df], axis=1)
    predictions_proba = predictor.predict_proba(transformed_df_batch)
    print(f"Probabilidades de SleepStage:\n{predictions_proba}")
    sleepStage = 0
    if predictions_proba.at[0, 1] > 0.7:
        sleepStage = 1
        print("[INFO] Se detectó que el usuario está somnoliento")
    elif predictions_proba.at[0, 2] > 0.7:
        sleepStage = 2
        print("[INFO] Se detectó que el usuario está dormido")
    return sleepStage

# Load the model
predictor = TabularPredictor.load(f'out/autogluon/model/')
sscaler = joblib.load('out/autogluon/Autogluon_standard_scaler.pkl')
mmscaler = joblib.load('out/autogluon/Autogluon_minmax_scaler.pkl')
pca = joblib.load('out/autogluon/Autogluon_PCA.pkl')

# ask for the user input
age = int(input("Ingrese la edad del participante: "))
height = int(input("Ingrese la altura del participante en centímetros: "))
sleep_duration_h = int(input("Ingrese cuántas horas durmió: "))
fatigue = int(input("Ingrese el nivel de fatiga del participante (1 - 5, siendo 3 medio, 1-2 bajo lo normal y 4-5 más alto lo normal): "))
sleep_quality = int(input("Ingrese la calidad del sueño del participante (1 - 5, siendo 3 medio, 1-2 bajo lo normal y 4-5 más alto lo normal): "))
stress = int(input("Ingrese el nivel estrés del participante (1 - 5, siendo 3 medio, 1-2 bajo lo normal y 4-5 más alto lo normal): "))

file_name = "my_test.csv"

# dataframe to store the heart rate data
hr_sleep_df = pd.DataFrame(columns=['height', 'fatigue', 'sleep_duration', 'sleep_quality', 'stress',
        'Interval', 'Time', 'BPM', 'BPM_Mean_Acc', 'BPM_Var_Acc',
        'BPM_Std_Acc', 'BPM_Diff', 'BPM_Acceleration', 'BPM_Mean_Diff',
        'Age_Binned_<25', 'Age_Binned_25_40', 'Age_Binned_40_60',
        'Age_Binned_>60', 'Time_SleepStage', 'SleepStage_Changes', 'BPM_Trend',
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
        'BPM_rolling_range_5'])
df_columns = set(hr_sleep_df.columns)

person_data = {
    "Age_Binned_<25": 1 if age < 25 else 0,
    "Age_Binned_25_40": 1 if 25 <= age < 40 else 0,
    "Age_Binned_40_60": 1 if 40 <= age < 60 else 0,
    "Age_Binned_>60": 1 if age >= 60 else 0,
    "height": height,
    "fatigue": fatigue,
    "sleep_duration": sleep_duration_h,
    "sleep_quality": sleep_quality,
    "stress": stress
}

startTime = 0
time_sleepStage_offset = 0
last_sleepStage = 0
sleepStage_changes = 0
interval = 0

async def handle_connection(websocket, path):
    global startTime, time_sleepStage_offset, last_sleepStage, sleepStage_changes, interval, hr_sleep_df  # Declaramos las variables globales
    print("Esperando mensaje...")
    try:
        async for message in websocket:
            # Verificar si el mensaje es un ping
            if message == '{"type":"ping"}':
                print("Se recibió un ping del cliente.")
                await websocket.pong()
                continue  # Saltar el procesamiento de datos si es un ping

            # Procesar solo los mensajes que no sean ping
            try:
                received_json = json.loads(message)
                if "dateTime" not in received_json or "bpm" not in received_json:
                    raise ValueError("Mensaje inválido, faltan campos necesarios")

                dateTime = pd.to_datetime(received_json["dateTime"])
                if startTime == 0:
                    startTime = dateTime
                secondsPassed = int((dateTime - startTime).total_seconds())
                print(f"Hora: {dateTime}, Time: {secondsPassed}, Interval: {interval}, BPM: {received_json['bpm']}")

                if secondsPassed > 900:
                    # quitar los registros mas antiguos, dejar siempre 15 minutos
                    # la diferencia de Time entre el registro mas nuevo y el ultimo que se deja debe ser 900 segundos
                    hr_sleep_df = hr_sleep_df[hr_sleep_df["Time"] >= hr_sleep_df["Time"].iloc[-1] - 900]

                    # Verificar si han pasado 15 minutos
                    if (secondsPassed % 900) < (hr_sleep_df["Time"].iloc[-1] % 900):
                        # Reiniciar el intervalo
                        interval = 0

                bpm_window.append(received_json["bpm"])
                bpm_trend = calculate_trend(bpm_window)

                new_row = {
                    "Interval": interval,
                    "Time": secondsPassed,
                    "BPM": received_json["bpm"],
                    "Age_Binned_<25": person_data["Age_Binned_<25"],
                    "Age_Binned_25_40": person_data["Age_Binned_25_40"],
                    "Age_Binned_40_60": person_data["Age_Binned_40_60"],
                    "Age_Binned_>60": person_data["Age_Binned_>60"],
                    "height": person_data["height"],
                    "fatigue": person_data["fatigue"],
                    "sleep_duration": person_data["sleep_duration"],
                    "sleep_quality": person_data["sleep_quality"],
                    "stress": person_data["stress"],
                    "BPM_Mean_Acc": hr_sleep_df["BPM"].mean() if len(hr_sleep_df) > 0 else np.nan,
                    "BPM_Var_Acc": hr_sleep_df["BPM"].var() if len(hr_sleep_df) > 0 else np.nan,
                    "BPM_Std_Acc": hr_sleep_df["BPM"].std() if len(hr_sleep_df) > 0 else np.nan,
                    "BPM_Diff": received_json["bpm"] - hr_sleep_df["BPM"].iloc[-1] if len(hr_sleep_df) > 0 else np.nan,
                    "BPM_Acceleration": received_json["bpm"] - 2 * hr_sleep_df["BPM"].iloc[-1] + hr_sleep_df["BPM"].iloc[-2] if len(hr_sleep_df) > 1 else np.nan,
                    "BPM_Mean_Diff": received_json["bpm"] - hr_sleep_df["BPM"].mean() if len(hr_sleep_df) > 0 else np.nan,
                    "BPM_Trend": bpm_trend,
                    "Time_SleepStage": secondsPassed - time_sleepStage_offset,
                    "SleepStage_Changes": sleepStage_changes
                }

                new_row_df = pd.DataFrame([new_row], columns=hr_sleep_df.columns)

                lag_columns = ['BPM', 'BPM_Diff', 'BPM_Acceleration', 'BPM_Mean_Diff', 'BPM_Mean_Acc', 'BPM_Trend', 'Time_SleepStage']
                for i in range(1, 6):
                    for col in lag_columns:
                        new_row_df[f'{col}_Lag_{i}'] = hr_sleep_df[col].shift(i).ffill().bfill()

                # Agregando promedios, máximos y mínimos móviles
                rolling_stats = hr_sleep_df['BPM'].rolling(window=window_size).agg(['mean', 'max', 'min', 'var', 'std'])
                new_row_df[f'BPM_rolling_mean_{window_size}'] = rolling_stats['mean'].ffill().bfill()
                new_row_df[f'BPM_rolling_max_{window_size}'] = rolling_stats['max'].ffill().bfill()
                new_row_df[f'BPM_rolling_min_{window_size}'] = rolling_stats['min'].ffill().bfill()
                new_row_df[f'BPM_rolling_std_{window_size}'] = rolling_stats['std'].ffill().bfill()
                new_row_df[f'BPM_rolling_range_{window_size}'] = (new_row_df[f'BPM_rolling_max_{window_size}'] - new_row_df[f'BPM_rolling_min_{window_size}']).ffill().bfill()

                for col in ['BPM_Mean_Acc', 'BPM_Var_Acc', 'BPM_Std_Acc', 'BPM_Diff', 'BPM_Acceleration', 'BPM_Mean_Diff', 'BPM_Trend', 'BPM_rolling_std_5']:
                    new_row_df[col] = new_row_df[col].round(2)

                if len(hr_sleep_df) >= 24:
                    sleepStage = make_predictions(hr_sleep_df)
                    if last_sleepStage != sleepStage:
                        time_sleepStage_offset = secondsPassed
                        last_sleepStage = sleepStage
                        sleepStage_changes += 1
                        new_row_df["Time_SleepStage"] = secondsPassed - time_sleepStage_offset
                        new_row_df["SleepStage_Changes"] = sleepStage_changes
                        # enviar alerta al cliente
                        await websocket.send(json.dumps({"type": "alert", "sleepStage": sleepStage}))

                hr_sleep_df = pd.concat([hr_sleep_df, new_row_df], ignore_index=True)
                hr_sleep_df = hr_sleep_df.infer_objects(copy=False)
                hr_sleep_df = hr_sleep_df.ffill()
                hr_sleep_df = hr_sleep_df.bfill()
                await asyncio.to_thread(hr_sleep_df.to_csv, file_name, index=False)
                interval += 1

            except (ValueError, KeyError) as e:
                print(f"Error en datos recibidos: {e}")

    except websockets.ConnectionClosed:
        print("Conexión cerrada")
    except websockets.ConnectionClosedError as e:
        print(f"Conexión cerrada con error: {e}")
    except Exception as e:
        print(f"Error inesperado: {e}")
    finally:
        print("Conexión finalizada o cerrada.")

# Bind the socket to a specific address and port
host = '0.0.0.0'
port = 2222

# Ejecutar el servidor
async def main():
    # Crear el servidor websocket
    server = await websockets.serve(handle_connection, '0.0.0.0', 2222)
    print(f"Servidor websocket escuchando en 0.0.0.0:2222")

    # Mantener el servidor activo indefinidamente
    await server.wait_closed()

# Ejecutar el servidor usando asyncio.run()
if __name__ == "__main__":
    asyncio.run(main())