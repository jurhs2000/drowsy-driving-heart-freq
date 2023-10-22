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

# Load the model
model = tf.keras.models.load_model('model.h5', compile=False)
predictor = TabularPredictor.load('AutogluonModels/ag-20231022_143722/')
loaded_scaler = joblib.load('scaler.pkl')

# Create a socket object
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

file_name = "my_test.csv"

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
hr_sleep_df = pd.DataFrame(columns=['participant', 'logId', 'phaseNo', 'hrNo', 'dateTime', 'bpm', 'age', 'sex', 'person'])
hr_sleep_pred_df = pd.DataFrame(columns=['participant', 'logId', 'phaseNo', 'hrNo', 'dateTime', 'bpm', 'age', 'sex', 'person', 'level', '0', '1', '2'])

person_data = {
    "id": 17,
    "logId": random.randint(1, 10000000000),
    "age": 23,
    "height": 164, # in cm
    "sex": 0, # 0 for Male and 1 for Female
    "person": 0, # 0 for A Person and 1 for B Person
}

phaseNo = 0
while True:
    try:
        # Receive JSON data from the client
        data = client_socket.recv(1024).decode('utf-8')
        if not data:
            break
        # Deserialize the JSON data
        if is_valid_json(data):
            received_json = json.loads(data)
            print(received_json)
            new_row = {
                "participant": person_data["id"],
                "logId": person_data["logId"],
                "phaseNo": phaseNo,
                "hrNo": len(hr_sleep_df),
                "dateTime": pd.to_datetime(received_json["dateTime"]).timestamp(),
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
            print(predictions[-1], " ----- ", predictions_proba[-1])
            new_row["level"] = predictions[-1]
            new_row["0"] = predictions_proba[-1][0]
            new_row["1"] = predictions_proba[-1][1]
            if len(predictions_proba[-1]) == 3:
                new_row["2"] = predictions_proba[-1][2]
            else:
                new_row["2"] = 0
            # return 'dateTime' to the original format
            new_row["dateTime"] = pd.to_datetime(new_row["dateTime"], unit='s').strftime('%Y-%m-%d %H:%M:%S')
            # Append the received dataframe to the main dataframe
            hr_sleep_pred_df.loc[len(hr_sleep_pred_df)] = new_row
            # Save the dataframe to a CSV file
            hr_sleep_pred_df.to_csv(file_name, index=False)
            if predictions[-1] != phaseNo:
                phaseNo += 1
    except ConnectionResetError:
        print("Client disconnected abruptly")
        break



