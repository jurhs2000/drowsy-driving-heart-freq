import numpy as np
import random
import pandas as pd
import json
import joblib
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from collections import deque
import asyncio
import websockets
import traceback
import sys

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

def weighted_prediction(predictions, decay_factor=0.8):
    weighted_avg = np.zeros(predictions[0].shape)
    for i, pred in enumerate(predictions):
        weight = decay_factor ** (len(predictions) - i - 1)
        weighted_avg += weight * pred
    weighted_avg /= sum([decay_factor ** i for i in range(len(predictions))])
    return weighted_avg

def make_predictions(hr_sleep_df):
    # Escalar los datos
    scaled_features = hr_sleep_df.copy()
    scaled_features[numerical_columns] = sscaler.transform(scaled_features[numerical_columns])
    scaled_features[fixed_range_features] = mmscaler.transform(scaled_features[fixed_range_features])
    scaled_features[categorical_columns] = scaled_features[categorical_columns].astype('category')

    # agregar una columna 'sequenceId' con valor 0
    scaled_features['sequenceId'] = 0

    # Convertir 'Time' a numérico para optimización
    scaled_features['Time'] = pd.to_numeric(scaled_features['Time'], downcast='integer')

    # Crear la columna 'timestamp' a partir de 'Time'
    scaled_features['timestamp'] = pd.to_datetime(scaled_features['Time'], unit='s')

    # Resample based on timestamp and fill missing values with ffill()
    #print(scaled_features.iloc[0].to_markdown())
    #print("columns: ", scaled_features.columns)
    #duplicados = scaled_features[scaled_features.duplicated(subset=['timestamp'], keep=False)]
    #if not duplicados.empty:
        #print("Se encontraron duplicados en 'timestamp':")
        #print(duplicados)

    scaled_features = scaled_features.drop_duplicates(subset=['timestamp'], keep='last')
    scaled_features = scaled_features.set_index('timestamp').resample('5S').ffill().reset_index()

    # revisar si existen valores nan o inf
    if scaled_features.isnull().values.any():
        #print("Existen valores nulos en el DataFrame")
        scaled_features = scaled_features.dropna()
    if scaled_features.isin([np.inf, -np.inf]).values.any():
        #print("Existen valores infinitos en el DataFrame")
        scaled_features = scaled_features.replace([np.inf, -np.inf], np.nan).dropna()

    # Convertir a entero si no lo es
    scaled_features['sequenceId'] = scaled_features['sequenceId'].astype(int)

    # Eliminar filas con valores nulos en 'sequenceId'
    scaled_features = scaled_features.dropna(subset=['sequenceId'])

    # Si 'sequenceId' es parte del índice, restablece el índice
    if 'sequenceId' not in scaled_features.columns:
        scaled_features = scaled_features.reset_index()

    timeSeries_df = TimeSeriesDataFrame(scaled_features, id_column="sequenceId", timestamp_column="timestamp")

    # Verificar que la columna 'timestamp' está presente
    if 'timestamp' not in timeSeries_df.columns:
        timeSeries_df = timeSeries_df.reset_index()  # Convertir el índice a columnas
        timeSeries_df['sequenceId'] = 0  # Reagregar 'sequenceId' si es necesario

    # Crear MultiIndex si 'timestamp' no está en el índice
    timeSeries_df = timeSeries_df.set_index(['sequenceId', 'timestamp'])
    timeSeries_df.index.names = ['item_id', 'timestamp']  # Asegurar nombres correctos
    #print(timeSeries_df.index.names)


    # quitar la columna item_id sin quitar el índice
    timeSeries_df = timeSeries_df.drop(columns=['item_id'])

    #print("ejemplo completo de una fila de timeSeries_df")
    #print(timeSeries_df.iloc[0].to_markdown())
    # Lista del orden de columnas deseado
    columnas_ordenadas = [
        'height', 'fatigue', 'sleep_duration', 'sleep_quality', 'stress', 'Interval', 
        'Time', 'BPM', 'SleepStage', 'BPM_Mean_Acc', 'BPM_Var_Acc', 'BPM_Std_Acc', 
        'BPM_Diff', 'BPM_Acceleration', 'BPM_Mean_Diff', 'Age_Binned_<25', 
        'Age_Binned_25_40', 'Age_Binned_40_60', 'Age_Binned_>60', 
        'Time_SleepStage', 'SleepStage_Changes', 'BPM_Trend'
    ]

    # Reordenar las columnas del TimeSeriesDataFrame
    timeSeries_df = timeSeries_df[columnas_ordenadas]

    # Verificar el reordenamiento
    #print(timeSeries_df.iloc[0].to_markdown())


    # Predicción
    print("timeSeries_df:")
    print(timeSeries_df[['Interval', 'Time', 'BPM', 'SleepStage']])
    forecast = predictor.predict(timeSeries_df, use_cache=False)
    print(f"Predicciones:\n{forecast}")
    #print(f"Predicciones: {forecast}")
    # obtener los valores de mean como lista
    predictions = forecast['mean'].values
    #print(f"Predicciones mean: {predictions}")
    weighted_pred = weighted_prediction(predictions, decay_factor=0.95)
    # redondear al valor mas cercano y hacer absoluto para solo positivos (o 0)
    print(f"Predicciones ponderadas: {weighted_pred}")
    sleepStage = np.abs(np.round(weighted_pred))
    print(f"weighted sleepstage: {sleepStage}")
    mean_predictions = np.round(forecast['mean'])
    mean_predictions = np.clip(mean_predictions, 0, 2)
    print(f"Predicciones mean: {mean_predictions}")
    return sleepStage

# Load the model
predictor = TimeSeriesPredictor.load('out/autogluonforecast/model/')
sscaler = joblib.load('out/autogluonforecast/Autogluon_standard_scaler.pkl')
mmscaler = joblib.load('out/autogluonforecast/Autogluon_minmax_scaler.pkl')

# ask for the user input
age = int(input("Ingrese la edad del participante: "))
height = int(input("Ingrese la altura del participante en centímetros: "))
sleep_duration_h = int(input("Ingrese cuántas horas durmió: "))
fatigue = int(input("Ingrese el nivel de fatiga del participante (1 - 5, siendo 3 medio, 1-2 bajo lo normal y 4-5 más alto lo normal): "))
sleep_quality = int(input("Ingrese la calidad del sueño del participante (1 - 5, siendo 3 medio, 1-2 bajo lo normal y 4-5 más alto lo normal): "))
stress = int(input("Ingrese el nivel estrés del participante (1 - 5, siendo 3 medio, 1-2 bajo lo normal y 4-5 más alto lo normal): "))

file_name = "my_test.csv"

# DataFrame para almacenar datos de ritmo cardíaco
hr_sleep_df = pd.DataFrame(columns=['height', 'fatigue', 'sleep_duration', 'sleep_quality', 'stress',
                                    'Interval', 'Time', 'BPM', 'BPM_Mean_Acc', 'BPM_Var_Acc',
                                    'BPM_Std_Acc', 'BPM_Diff', 'BPM_Acceleration', 'BPM_Mean_Diff',
                                    'Age_Binned_<25', 'Age_Binned_25_40', 'Age_Binned_40_60',
                                    'Age_Binned_>60', 'Time_SleepStage', 'SleepStage_Changes', 'BPM_Trend', 'SleepStage'])


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
            print(f"Mensaje recibido: {message}")
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
                for col in ['BPM_Mean_Acc', 'BPM_Var_Acc', 'BPM_Std_Acc', 'BPM_Diff', 'BPM_Acceleration', 'BPM_Mean_Diff', 'BPM_Trend']:
                    new_row_df[col] = new_row_df[col].round(2)

                sleepStage = 0
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

                # Agregar SleepStage a la fila
                new_row_df["SleepStage"] = sleepStage

                hr_sleep_df = pd.concat([hr_sleep_df, new_row_df], ignore_index=True)
                hr_sleep_df = hr_sleep_df.infer_objects(copy=False)
                hr_sleep_df = hr_sleep_df.ffill()
                hr_sleep_df = hr_sleep_df.bfill()
                await asyncio.to_thread(hr_sleep_df.to_csv, file_name, index=False)
                interval += 1

            except (ValueError, KeyError) as e:
                print(f"Error durante el proceso: {e}")
                # Captura y muestra el traceback completo del error
                traceback.print_exc()
                # Obtén más información del error con sys.exc_info()
                exc_type, exc_value, exc_tb = sys.exc_info()
                linea_error = exc_tb.tb_lineno  # Línea donde ocurrió el error
                print(f"Tipo de excepción: {exc_type}")
                print(f"Ocurrió en la línea: {linea_error}")

    except websockets.ConnectionClosed:
        print("Conexión cerrada")
    except websockets.ConnectionClosedError as e:
        print(f"Conexión cerrada con error: {e}")
    except Exception as e:
        print(f"Error inesperado: {e}")
        traceback.print_exc()  # Muestra el traceback para excepciones inesperadas
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