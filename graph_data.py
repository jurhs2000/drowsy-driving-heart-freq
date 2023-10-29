import matplotlib.pyplot as plt
import pandas as pd

processed_no = 1

participant = 4

# load df from data/pmdata/p01/fitbit/hr_sleep.csv
df = pd.read_csv(f'data/processed{processed_no}/data/all_hr_sleep_p{participant:02d}.csv')

# logId to search for
logId = 25723406286

# get the data for the logId
logId_df = df[df['logId'] == logId]

# dataframe to store the heart rate data, with the columns of the original dataframe
hr_sleep_df = pd.DataFrame(columns=df.columns)

last_level = None
for hr in logId_df.iterrows():
    # get the level
    level = hr[1]['level']
    # if the level is different from the last level
    if level != last_level:
        # if the last level is not None
        if last_level is not None:
            # add the part of the plot for this phase, use green for level 0, red for level 1, and blue for level 2
            plt.plot(hr_sleep_df['secondsPassed'], hr_sleep_df['bpm'], color='green' if last_level == 0 else 'red' if last_level == 1 else 'blue')
            # reset the dataframe
            hr_sleep_df = pd.DataFrame(columns=df.columns)
    # add the row to the dataframe
    hr_sleep_df.loc[len(hr_sleep_df)] = hr[1]
    # update the last level
    last_level = level

# add the part of the plot for this phase, use green for level 0, red for level 1, and blue for level 2
plt.plot(hr_sleep_df['secondsPassed'], hr_sleep_df['bpm'], color='green' if last_level == 0 else 'red' if last_level == 1 else 'blue')
plt.title(f'Heart Rate vs Time for logId {logId}')
plt.xlabel('Time')
plt.ylabel('Heart Rate')
plt.show()
