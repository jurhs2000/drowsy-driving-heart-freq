import tensorflow as tf
import numpy as np
import socket
import json

# Load the model
model = tf.keras.models.load_model('model.h5')

# Create a socket object
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to a specific address and port
host = '0.0.0.0'
port = 2222
server_socket.bind((host, port))

# Listen for incoming connections
server_socket.listen(1)

while True:
    # Accept a connection from a client
    client_socket, addr = server_socket.accept()
    print(f"Received connection from {addr}")

    # Receive JSON data from the client
    data = client_socket.recv(1024).decode('utf-8')

    # Deserialize the JSON data
    received_json = json.loads(data)

    print(f"Received JSON object: {received_json}")

    # Close the client socket
    client_socket.close()

X_test = np.array([[[1, 2, 3, 4, 5, 6, 7]]])

predictions = model.predict(X_test)

print(predictions)

predicted_class = tf.argmax(predictions, axis=-1).numpy()

print(predicted_class)