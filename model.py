import pandas as pd

class HRModel():
    def __init__(self):
        # pandas dataframe to store the data
        self.df = pd.DataFrame(columns=['timestamp', 'heart_rate'])
        self.df.set_index('timestamp', inplace=True)
        
    # add a new row to the dataframe
    def add_row(self, timestamp, heart_rate):
        self.df.loc[timestamp] = heart_rate
        self.save_to_csv('heart_rate.csv')

    # save the dataframe to a csv file or update the csv file
    # don't delete the old data in the csv file
    def save_to_csv(self, filename):
        self.df.to_csv(filename, mode='a', header=False)
