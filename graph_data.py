import matplotlib.pyplot as plt
import pandas as pd

# load df from data/pmdata/p01/fitbit/hr_sleep.csv
df = pd.read_csv('data/pmdata/count_all_phases_only_first_cycle_both_types_both_sleeps_random_early_time.csv')

# logId to search for
logId = 25384935788

# get the data for the logId
logId_df = df[df['logId'] == logId]

# dataframe to store the heart rate data
hr_sleep_df = pd.DataFrame(columns=['participant', 'logId', 'phaseNo', 'hrNo', 'dateTime', 'bpm', 'level', 'age', 'sex', 'person'])

last_level = None
for hr in logId_df.iterrows():
    # get the level
    level = hr[1]['level']
    # if the level is different from the last level
    if level != last_level:
        # if the last level is not None
        if last_level is not None:
            # add the part of the plot for this phase, use green for level 0, red for level 1, and blue for level 2
            plt.plot(hr_sleep_df['dateTime'], hr_sleep_df['bpm'], color='green' if last_level == 0 else 'red' if last_level == 1 else 'blue')
            # reset the dataframe
            hr_sleep_df = pd.DataFrame(columns=['participant', 'logId', 'dateTime', 'bpm', 'level', 'age', 'sex', 'person'])
    # add the row to the dataframe
    hr_sleep_df.loc[len(hr_sleep_df)] = hr[1]
    # update the last level
    last_level = level

# add the part of the plot for this phase, use green for level 0, red for level 1, and blue for level 2
plt.plot(hr_sleep_df['dateTime'], hr_sleep_df['bpm'], color='green' if last_level == 0 else 'red' if last_level == 1 else 'blue')
plt.title(f'Heart Rate vs Time for logId {logId}')
plt.xlabel('Time')
plt.ylabel('Heart Rate')
plt.show()
