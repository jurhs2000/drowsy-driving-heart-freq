import tensorflow as tf
import numpy as np
import socket
import random
import pandas as pd
import json

# Load the model
model = tf.keras.models.load_model('model.h5', compile=False)

# Create a socket object
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

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
hr_sleep_df = pd.DataFrame(columns=['participant', 'logId', 'dateTime', 'bpm', 'age', 'sex', 'person'])

person_data = {
    "id": 17,
    "logId": random.randint(1, 10000000000),
    "age": 23,
    "height": 164, # in cm
    "sex": 0, # 0 for Male and 1 for Female
    "person": 0, # 0 for A Person and 1 for B Person
}

while True:
    try:
        # Receive JSON data from the client
        data = client_socket.recv(1024).decode('utf-8')
        if not data:
            break
        # Deserialize the JSON data
        received_json = json.loads(data)
        new_row = {
            "participant": person_data["id"],
            "logId": person_data["logId"],
            "dateTime": pd.to_datetime(received_json["dateTime"]).timestamp(),
            "bpm": received_json["bpm"],
            "age": person_data["age"],
            "height": person_data["height"],
            "sex": person_data["sex"],
            "person": person_data["person"]
        }
        # Append the received dataframe to the main dataframe
        hr_sleep_df.loc[len(hr_sleep_df)] = new_row
        x_array = np.asarray(hr_sleep_df).astype(np.float32)
        predictions = model.predict(x_array.reshape(-1, 1, 7))
        #print(predictions)
        predicted_class = tf.argmax(predictions, axis=-1).numpy()
        print(predicted_class)
    except ConnectionResetError:
        print("Client disconnected abruptly")
        break



