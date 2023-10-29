import tensorflow as tf
import numpy as np
import socket
import random
import pandas as pd
import json
import joblib
from autogluon.tabular import TabularDataset, TabularPredictor

def is_valid_json(mystring):
    try:
        json.loads(mystring)
        return True
    except ValueError:
        return False
    
processed_no = int(input("Ingrese el número del dataset a usar: "))

# Load the model
predictor = TabularPredictor.load(f'data/processed{processed_no}/AutogluonModels')
loaded_scaler = joblib.load(f'data/processed{processed_no}/scaler.pkl')

# Create a socket object
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

file_name = f"data/processed{processed_no}/all_hr_sleep_pred.csv"

# Bind the socket to a specific address and port
host = '0.0.0.0'
port = 2222
server_socket.bind((host, port))

# Listen for incoming connections
server_socket.listen(1)

# Accept a connection from a client
client_socket, addr = server_socket.accept()
print(f"Received connection from {addr}")

# dataframe to store the heart rate data
hr_sleep_df = pd.DataFrame(columns=['participant','logId','phaseNo','hrNo','secondsPassed','daySeconds','bpm','resting_hr','age','sex','person','fatigue','mood','sleep_duration_h','sleep_quality','stress'])
hr_sleep_pred_df = pd.DataFrame(columns=['participant','logId','phaseNo','hrNo','secondsPassed','daySeconds','bpm','resting_hr','age','sex','person','fatigue','mood','sleep_duration_h','sleep_quality','stress', 'level', '0', '1', '2'])

age = int(input("Ingrese la edad del participante: "))
sex = int(input("Ingrese el sexo del participante (0 para hombre, 1 para mujer): "))
height = int(input("Ingrese la altura del participante en centímetros: "))
person = int(input("Ingrese el número del participante (0 para A, 1 para B): "))
resting_hr = int(input("Ingrese la frecuencia cardíaca en reposo del participante: "))
fatigue = int(input("Ingrese el nivel de fatiga del participante (1 para baja, 2 para media, 3 para alta): "))
mood = int(input("Ingrese el nivel ánimo del participante (1 para bajo, 2 para medio, 3 para alto): "))
sleep_duration_h = int(input("Ingrese cuántas horas durmió: "))
sleep_quality = int(input("Ingrese la calidad del sueño del participante (1 para baja, 2 para media, 3 para alta): "))
stress = int(input("Ingrese el nivel estrés del participante (1 para bajo, 2 para medio, 3 para alto): "))

person_data = {
    "id": 17,
    "logId": random.randint(1, 10000000000),
    "age": age,
    "height": height,
    "sex": sex,
    "person": person,
    "resting_hr": resting_hr,
    "fatigue": fatigue,
    "mood": mood,
    "sleep_duration_h": sleep_duration_h,
    "sleep_quality": sleep_quality,
    "stress": stress
}

phaseNo = 0
startTime = 0
while True:
    try:
        # Receive JSON data from the client
        data = client_socket.recv(1024).decode('utf-8')
        if not data:
            break
        # Deserialize the JSON data
        if is_valid_json(data):
            received_json = json.loads(data)
            dateTime = pd.to_datetime(received_json["dateTime"])
            if startTime == 0:
                startTime = dateTime
            secondsPassed = (dateTime - startTime).total_seconds()
            new_row = {
                "participant": person_data["id"],
                "logId": person_data["logId"],
                "phaseNo": phaseNo,
                "hrNo": len(hr_sleep_df),
                "secondsPassed": int(secondsPassed),
                "daySeconds": int(pd.Timedelta(hours=dateTime.hour, minutes=dateTime.minute, seconds=dateTime.second).total_seconds()),
                "bpm": received_json["bpm"],
                "age": person_data["age"],
                "height": person_data["height"],
                "sex": person_data["sex"],
                "person": person_data["person"]
            }
            # Append the received dataframe to the main dataframe
            hr_sleep_df.loc[len(hr_sleep_df)] = new_row
            scaled_new_data = loaded_scaler.transform(hr_sleep_df)
            scaled_df = pd.DataFrame(scaled_new_data, columns=hr_sleep_df.columns)
            to_predict = TabularDataset(scaled_df)
            predictions = predictor.predict(to_predict).tolist()
            predictions_proba = predictor.predict_proba(to_predict).values.tolist()
            new_row["level"] = predictions[-1]
            new_row["0"] = predictions_proba[-1][0]
            new_row["1"] = predictions_proba[-1][1]
            if len(predictions_proba[-1]) == 3:
                new_row["2"] = predictions_proba[-1][2]
            else:
                new_row["2"] = 0
            # Append the received dataframe to the main dataframe
            hr_sleep_pred_df.loc[len(hr_sleep_pred_df)] = new_row
            # Save the dataframe to a CSV file
            hr_sleep_pred_df.to_csv(file_name, index=False)
            if predictions[-1] != phaseNo:
                phaseNo += 1
    except ConnectionResetError:
        print("Client disconnected abruptly")
        break



