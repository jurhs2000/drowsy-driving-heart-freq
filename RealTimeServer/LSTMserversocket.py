import numpy as np
import pandas as pd
import json
import joblib
import tensorflow as tf
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

x = np.arange(window_size)
sum_x = np.sum(x)
sum_x_squared = np.sum(x**2)

def calculate_trend(bpm_window):
    if len(bpm_window) < window_size:
        return np.nan
    sum_y = np.sum(bpm_window)
    sum_xy = np.sum(x * bpm_window)
    numerator = window_size * sum_xy - sum_x * sum_y
    denominator = window_size * sum_x_squared - sum_x**2
    return numerator / denominator

def weighted_prediction(predictions, decay_factor=0.65):
    weighted_avg = np.zeros(predictions[0].shape)
    for i, pred in enumerate(predictions):
        weight = decay_factor ** (len(predictions) - i - 1)
        weighted_avg += weight * pred
    weighted_avg /= sum([decay_factor ** i for i in range(len(predictions))])
    return weighted_avg

def make_predictions_for_large_sequence(hr_sleep_df):
    model_seq_length = 24
    scaled_features = hr_sleep_df.copy()
    scaled_features[continuous_features] = sscaler.transform(scaled_features[continuous_features])
    scaled_features[fixed_range_features] = mmscaler.transform(scaled_features[fixed_range_features])

    X_seq_features_all = scaled_features[seq_features].values.astype(np.float32)
    if len(X_seq_features_all) > model_seq_length:
        sub_sequences = [X_seq_features_all[i:i + model_seq_length] for i in range(len(X_seq_features_all) - model_seq_length + 1)]
        X_seq_features_all = np.array(sub_sequences)
    else:
        X_seq_features_all = X_seq_features_all.reshape(1, model_seq_length, len(seq_features))

    X_non_seq_features_all = scaled_features[non_seq_features].values[-1].astype(np.float32).reshape(1, len(non_seq_features))
    X_non_seq_features_all = np.repeat(X_non_seq_features_all, X_seq_features_all.shape[0], axis=0)

    predictions = model.predict([X_seq_features_all, X_non_seq_features_all], verbose=0)
    weighted_pred = weighted_prediction(predictions)
    # mostrar predicciones solo con 5 decimales
    weighted_pred = [round(p, 5) for p in weighted_pred]
    print(f"Predicciones: {weighted_pred}")
    sleepStage = 0
    if weighted_pred[1] > 0.7:
        sleepStage = 1
        print("[INFO] Se detectó que el usuario está somnoliento")
    elif weighted_pred[2] > 0.7:
        sleepStage = 2
        print("[INFO] Se detectó que el usuario está dormido")
    return sleepStage

# Cargar el modelo entrenado
model = tf.keras.models.load_model('out/lstm/best_lstm_model')
sscaler = joblib.load(f'out/lstm/LSTM_standard_scaler.pkl')
mmscaler = joblib.load(f'out/lstm/LSTM_minmax_scaler.pkl')

# Pedir datos al usuario
age = int(input("Ingrese la edad del participante: "))
height = int(input("Ingrese la altura del participante en centímetros: "))
sleep_duration_h = int(input("Ingrese cuántas horas durmió: "))
fatigue = int(input("Ingrese el nivel de fatiga del participante (1 - 5): "))
sleep_quality = int(input("Ingrese la calidad del sueño del participante (1 - 5): "))
stress = int(input("Ingrese el nivel de estrés del participante (1 - 5): "))

file_name = "my_test.csv"

# DataFrame para almacenar datos de ritmo cardíaco
hr_sleep_df = pd.DataFrame(columns=['height', 'fatigue', 'sleep_duration', 'sleep_quality', 'stress',
                                    'Interval', 'Time', 'BPM', 'BPM_Mean_Acc', 'BPM_Var_Acc',
                                    'BPM_Std_Acc', 'BPM_Diff', 'BPM_Acceleration', 'BPM_Mean_Diff',
                                    'Age_Binned_<25', 'Age_Binned_25_40', 'Age_Binned_40_60',
                                    'Age_Binned_>60', 'Time_SleepStage', 'SleepStage_Changes', 'BPM_Trend'])

# Variables continuas y secuenciales
continuous_features = ['height', 'sleep_duration', 'Interval', 'Time', 'BPM', 'BPM_Mean_Acc', 'BPM_Var_Acc', 'BPM_Std_Acc', 'BPM_Diff', 'BPM_Acceleration', 'BPM_Mean_Diff', 'Time_SleepStage', 'SleepStage_Changes', 'BPM_Trend']
fixed_range_features = ['fatigue', 'stress', 'sleep_quality']
seq_features = ['Interval', 'Time', 'BPM', 'BPM_Mean_Acc', 'BPM_Var_Acc', 'BPM_Std_Acc', 'BPM_Diff', 'BPM_Acceleration', 'BPM_Mean_Diff', 'Time_SleepStage', 'SleepStage_Changes', 'BPM_Trend']
non_seq_features = ['height', 'fatigue', 'sleep_duration', 'sleep_quality', 'stress', 'Age_Binned_<25', 'Age_Binned_25_40', 'Age_Binned_40_60', 'Age_Binned_>60']

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
    global startTime, time_sleepStage_offset, last_sleepStage, sleepStage_changes, interval, hr_sleep_df
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
                for col in ['BPM_Mean_Acc', 'BPM_Var_Acc', 'BPM_Std_Acc', 'BPM_Diff', 'BPM_Acceleration', 'BPM_Mean_Diff', 'BPM_Trend']:
                    new_row_df[col] = new_row_df[col].round(2)

                if len(hr_sleep_df) >= 24:
                    sleepStage = make_predictions_for_large_sequence(hr_sleep_df)
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