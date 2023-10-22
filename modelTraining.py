import pandas as pd
import random
from ydata_profiling import ProfileReport

participants_data = [
    {
        "id": 1,
        "participant": 'p01',
        "age": 48,
        "height": 195, # in cm
        "sex": 0, # 0 for Male and 1 for Female
        "person": 0, # 0 for A Person and 1 for B Person
    },
    {
        "id": 2,
        "participant": 'p02',
        "age": 60,
        "height": 180, # in cm
        "sex": 0, # 0 for Male and 1 for Female
        "person": 0, # 0 for A Person and 1 for B Person
    },
    {
        "id": 3,
        "participant": 'p03',
        "age": 25,
        "height": 184, # in cm
        "sex": 0, # 0 for Male and 1 for Female
        "person": 0, # 0 for A Person and 1 for B Person
    },
    {
        "id": 4,
        "participant": 'p04',
        "age": 26,
        "height": 163, # in cm
        "sex": 1, # 0 for Male and 1 for Female
        "person": 0, # 0 for A Person and 1 for B Person
    },
    {
        "id": 5,
        "participant": 'p05',
        "age": 35,
        "height": 176, # in cm
        "sex": 0, # 0 for Male and 1 for Female
        "person": 0, # 0 for A Person and 1 for B Person
    },
    {
        "id": 6,
        "participant": 'p06',
        "age": 42,
        "height": 179, # in cm
        "sex": 0, # 0 for Male and 1 for Female
        "person": 1, # 0 for A Person and 1 for B Person
    },
    {
        "id": 7,
        "participant": 'p07',
        "age": 26,
        "height": 177, # in cm
        "sex": 0, # 0 for Male and 1 for Female
        "person": 1, # 0 for A Person and 1 for B Person
    },
    {
        "id": 8,
        "participant": 'p08',
        "age": 27,
        "height": 186, # in cm
        "sex": 0, # 0 for Male and 1 for Female
        "person": 1, # 0 for A Person and 1 for B Person
    },
    {
        "id": 9,
        "participant": 'p09',
        "age": 26,
        "height": 180, # in cm
        "sex": 0, # 0 for Male and 1 for Female
        "person": 1, # 0 for A Person and 1 for B Person
    },
    {
        "id": 10,
        "participant": 'p10',
        "age": 38,
        "height": 179, # in cm
        "sex": 1, # 0 for Male and 1 for Female
        "person": 1, # 0 for A Person and 1 for B Person
    },
    {
        "id": 11,
        "participant": 'p11',
        "age": 25,
        "height": 171, # in cm
        "sex": 1, # 0 for Male and 1 for Female
        "person": 0, # 0 for A Person and 1 for B Person
    },
    {
        "id": 12,
        "participant": 'p12',
        "age": 27,
        "height": 178, # in cm
        "sex": 0, # 0 for Male and 1 for Female
        "person": 0, # 0 for A Person and 1 for B Person
    },
    {
        "id": 13,
        "participant": 'p13',
        "age": 31,
        "height": 183, # in cm
        "sex": 0, # 0 for Male and 1 for Female
        "person": 0, # 0 for A Person and 1 for B Person
    },
    {
        "id": 14,
        "participant": 'p14',
        "age": 45,
        "height": 181, # in cm
        "sex": 0, # 0 for Male and 1 for Female
        "person": 0, # 0 for A Person and 1 for B Person
    },
    {
        "id": 15,
        "participant": 'p15',
        "age": 54,
        "height": 180, # in cm
        "sex": 0, # 0 for Male and 1 for Female
        "person": 0, # 0 for A Person and 1 for B Person
    },
    {
        "id": 16,
        "participant": 'p16',
        "age": 23,
        "height": 182, # in cm
        "sex": 0, # 0 for Male and 1 for Female
        "person": 1, # 0 for A Person and 1 for B Person
    }
]

# dataframe to store the new heart rate + sleep data
hr_sleep_df = pd.DataFrame(columns=['participant', 'logId', 'phaseNo', 'hrNo', 'dateTime', 'bpm', 'level', 'age', 'sex', 'person']) # height, max heart rate

for person_data in participants_data:
    print(f'Starting {person_data["participant"]}')
    # read the JSON file
    with open(f'data/pmdata/{person_data["participant"]}/fitbit/heart_rate.json', 'r') as f:
        data = f.read()

    # convert JSON to pandas dataframe
    hr_df = pd.read_json(data)

    # read the JSON file
    with open(f'data/pmdata/{person_data["participant"]}/fitbit/sleep.json', 'r') as f:
        data = f.read()

    # convert JSON to pandas dataframe\
    sleep_df = pd.read_json(data)

    # the value column is a dictionary, keep only the 'bpm' attribute on int type
    hr_df['value'] = hr_df['value'].apply(lambda x: x['bpm'])
    # change the column name from value to bpm
    hr_df.rename(columns={'value': 'bpm'}, inplace=True)

    # generate sleep_df profile report
    #profile = ProfileReport(sleep_df, title='Sleep Profile Report', explorative=True)
    #profile.to_file(f'data/pmdata/{person_data["participant"]}/fitbit/sleep_profile_report.html')

    # delete from sleep_df the columns dateOfSleep, duration, minutesToFallAsleep, minutesAsleep, minutesAwake, minutesAfterWakeup, timeInBed, efficiency
    sleep_df = sleep_df.drop(columns=['dateOfSleep', 'duration', 'minutesToFallAsleep', 'minutesAsleep', 'minutesAwake', 'minutesAfterWakeup', 'timeInBed', 'efficiency'])

    # remaining: logId, startTime, endTime, type, infoCode, levels, mainSleep

    def get_level_value(level):
        if level == 'wake':
            return 0
        elif level == 'awake':
            return 0
        elif level == 'light':
            return 1
        elif level == 'restless':
            return 1
        elif level == 'asleep':
            return 2
        elif level == 'deep':
            return 2
        elif level == 'rem':
            return 2

    # will consider the "sleep" phases of the cycles
    check_sleep_level = True
    # will consider only the first cycle, until the first "asleep" or "deep" or "rem" phase
    check_only_first_cycle = True
    # specify which type of sleep phase to consider, "classic" or "stages" or "classic, stages"
    check_sleep_type = "stages, classic"
    # will use only phase with infoCode 0, that means that are sufficient data to determine sleep stage
    check_only_infoCode_0 = True
    # specify if will check only main sleeps, those that are not main sleeps or both
    check_main_sleep = "both" # "main", "not main", "both"
    # time to add before start of sleep (in seconds)
    time_before_sleep = random.randint(1800, 10800)

    # iterate over the rows of sleep_df
    for index, row in sleep_df.iterrows():
        # compare the phase type with the check_sleep_type
        if row['type'] not in check_sleep_type.split(', '):
            continue
        # if check_only_infoCode_0 is True, check if the infoCode is 0
        if check_only_infoCode_0 == True:
            if row['infoCode'] != 0:
                continue
        # if check_main_sleep is not "both", check what type of sleep want to consider
        if check_main_sleep != "both":
            if check_main_sleep == "main" and row['mainSleep'] == False:
                continue
            elif check_main_sleep == "not main" and row['mainSleep'] == True:
                continue
        first_cycle_complete = False
        is_first_phase = True
        phaseNo = 0
        hrNoOffset = 0
        for phase in row['levels']['data']:
            # if phase level is unknown, pass
            if phase['level'] == 'unknown':
                continue
            # only if check_only_first_cycle is True, check if the first cycle is complete
            if check_only_first_cycle == True:
                # if the first cycle is complete, end the for loop
                if first_cycle_complete == True:
                    break
                # if the first cycle is not complete, check if the current phase is "asleep" or "deep" or "rem"
                if get_level_value(phase['level']) == 2:
                    # only set to true, the loop will end at the next iteration
                    first_cycle_complete = True
            # if check_sleep_level is True, check if the current phase is "asleep" or "deep" or "rem"
            if check_sleep_level == False:
                if get_level_value(phase['level']) == 2:
                    # if first_cycle_complete is true, ends the for loop because this will be the last phase
                    if first_cycle_complete == True:
                        break
                    # if first_cycle_complete is false, just avoid this phase
                    else:
                        continue
            # convert to datetime
            if is_first_phase == True:
                is_first_phase = False
                start_phase = pd.to_datetime(phase['dateTime']) - pd.Timedelta(seconds=time_before_sleep)
            else:
                start_phase = pd.to_datetime(phase['dateTime'])
            end_phase = pd.to_datetime(phase['dateTime']) + pd.Timedelta(seconds=phase['seconds'])
            # create a new row for each hr_df row between the phae['dateTime'] and phase['dateTime'] + phase['seconds']
            # if no values are found, skip this phase
            between_hr = hr_df[(hr_df['dateTime'] >= start_phase) & (hr_df['dateTime'] <= end_phase)]
            if len(between_hr) == 0:
                continue
            phase_hr = pd.DataFrame(columns=['participant', 'logId', 'phaseNo', 'hrNo', 'dateTime', 'bpm', 'level', 'age', 'sex', 'person'])
            # add the number of row on between_hr
            phase_hr.loc[:, 'dateTime'] = between_hr['dateTime']
            phase_hr.loc[:, 'bpm'] = between_hr['bpm']
            # add the new rows to the hr_sleep_df dataframe (use iloc)
            phase_hr.loc[:, 'participant'] = person_data["id"]
            phase_hr.loc[:, 'logId'] = row['logId']
            phase_hr.loc[:, 'level'] = get_level_value(phase['level'])
            phase_hr.loc[:, 'age'] = person_data["age"]
            phase_hr.loc[:, 'sex'] = person_data["sex"]
            phase_hr.loc[:, 'person'] = person_data["person"]
            phase_hr.loc[:, 'phaseNo'] = phaseNo
            phase_hr.loc[:, 'hrNo'] = range(hrNoOffset, hrNoOffset + len(between_hr))
            hrNoOffset += len(between_hr)
            # concat the new rows to the hr_sleep_df dataframe
            hr_sleep_df = pd.concat([hr_sleep_df, phase_hr])
            phaseNo += 1

# write the hr_sleep_df dataframe to a csv file
hr_sleep_df.to_csv(f'data/pmdata/all_hr_sleep.csv', index=False)

# generate sleep_df profile report
profile = ProfileReport(hr_sleep_df, title='Heart Rate + Sleep Profile Report', explorative=True)
profile.to_file(f'data/pmdata/hr_sleep_profile_report.html')