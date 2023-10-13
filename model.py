import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, LSTM

class HRModel():
    def __init__(self):
        # pandas dataframe to store the data
        self.df = pd.DataFrame(columns=['timestamp', 'heart_rate'])
        self.df.set_index('timestamp', inplace=True)
        self.sequence_length = 10
        self.create_model()
        #self.train_model()
        
    # add a new row to the dataframe
    def add_row(self, timestamp, heart_rate):
        self.df.loc[timestamp] = heart_rate
        self.save_to_csv('heart_rate.csv')
        #self.predict()

    # save the dataframe to a csv file or update the csv file
    # don't delete the old data in the csv file
    def save_to_csv(self, filename):
        self.df.to_csv(filename, mode='a', header=False)

    # create the RNN model
    def create_model(self):
        self.model = Sequential()
        self.model.add(LSTM(128, input_shape=(5, 10)))
        self.model.add(Dense(1))
        self.model.build()
        self.model.summary()
        #self.model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError()])
    
    # prepare sequences from data
    def prepare_sequences(self, data, sequence_length):
        sequences = []
        labels = []
        for i in range(len(data) - sequence_length):
            seq = data[i:i + sequence_length, :]
            label = data[i + sequence_length, 1]  # Assuming drowsiness is determined by the last entry in the sequence (heartrate)
            sequences.append(seq)
            labels.append(label)
        return np.array(sequences), np.array(labels)

    # train the model
    def train_model(self):
        # Create a sample DataFrame (replace this with your actual data)
        data = {
            'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='1H'),
            'heartrate': np.random.randint(60, 100, size=100),
            'drowsy': np.random.randint(0, 2, size=100)
        }
        print(data)
        df = pd.DataFrame(data)
        # Split the DataFrame into features (X) and labels (y)
        X = df[['timestamp', 'heartrate']].values
        y = df['drowsy'].values
        # Normalize the features (heart rate)
        self.scaler = StandardScaler()
        X[:, 1] = self.scaler.fit_transform(X[:, 1].reshape(-1, 1)).flatten()
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_seq, y_train_seq = self.prepare_sequences(X_train, self.sequence_length)
        X_test_seq, y_test_seq = self.prepare_sequences(X_test, self.sequence_length)
        print(X_train_seq.shape)
        print(y_train_seq.shape)
        #self.model.fit(X_train_seq, y_train_seq, epochs=10, batch_size=32, validation_split=0.1, use_multiprocessing=False)
        #self.model.fit(X_train_seq, y_train_seq, epochs=10, batch_size=32, validation_split=0.1)
        #test_loss, test_accuracy = self.model.evaluate(X_test_seq, y_test_seq)
        #print(f"Test loss: {test_loss}, test accuracy: {test_accuracy}")

    def predict(self):
        # predict the drowsiness
        # if the predicted drowsiness is 1, then the driver is drowsy
        # if the predicted drowsiness is 0, then the driver is not drowsy
        # Prepare sequences from a new DataFrame for evaluation (e.g., df_eval)
        # df_eval should have columns 'timestamp' and 'heartrate'
        X_eval = self.df[['timestamp', 'heart_rate']].values
        X_eval[:, 1] = self.scaler.transform(X_eval[:, 1].reshape(-1, 1)).flatten()
        X_eval_seq, _ = self.prepare_sequences(X_eval, self.sequence_length)
        predictions = self.model.predict(X_eval_seq)
        print('----------------------------------')
        print(predictions)