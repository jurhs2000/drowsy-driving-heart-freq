import pandas as pd
import random
from ydata_profiling import ProfileReport
import threading

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

# will consider the "sleep" phases of the cycles
print('Seleccione si se tomarán en cuenta las fases de sueño profundo:')
print('1: Incluir fases de sueño profundo')
print('2: No incluir fases de sueño profundo')
check_sleep_level = int(input('Selecciona: '))
if check_sleep_level == 1:
    check_sleep_level = True
elif check_sleep_level == 2:
    check_sleep_level = False

# will consider only the first cycle, until the first "asleep" or "deep" or "rem" phase
print('Seleccione si se tomará en cuenta solo el primer ciclo de sueño, es decir, cuando termine la primera fase de sueño profundo:')
print('1: Solo el primer ciclo')
print('2: Todos los ciclos')
check_only_first_cycle = int(input('Selecciona: '))
if check_only_first_cycle == 1:
    check_only_first_cycle = True
elif check_only_first_cycle == 2:
    check_only_first_cycle = False

# specify which type of sleep phase to consider, "classic" or "stages" or "classic, stages"
print('Seleccione qué tipo de sueño se utilizará:')
print('1: Clasico')
print('2: Por etapas')
print('3: Clasico y por etapas')
check_sleep_type = int(input('Selecciona: '))
if check_sleep_type == 1:
    check_sleep_type = "classic"
elif check_sleep_type == 2:
    check_sleep_type = "stages"
elif check_sleep_type == 3:
    check_sleep_type = "stages, classic"

# will use only phase with infoCode 0, that means that are sufficient data to determine sleep stage
print('Seleccione si se usarán solamente los logs de sueño con suficientes datos (infoCide = 0):')
print('1: Solo logs con suficientes datos')
print('2: También logs con pocos datos y períodos de sueño cortos')
check_only_infoCode_0 = int(input('Selecciona: '))
if check_only_infoCode_0 == 1:
    check_only_infoCode_0 = True
elif check_only_infoCode_0 == 2:
    check_only_infoCode_0 = False

# what level of confidence of heart rate will use 1: low, 2: medium, 3: high
print('Seleccione el nivel de confianza mínimo de la frecuencia cardíaca:')
print('1: Bajo')
print('2: Medio')
print('3: Alto')
check_hr_confidence = int(input('Selecciona: '))
check_hr_confidence = 3

# specify if will check only main sleeps, those that are not main sleeps or both
print('Seleccione si se tomarán en cuenta solamente los sueños principales:')
print('1: Solo principales')
print('2: Solo no principales')
print('3: Ambos')
check_main_sleep = int(input('Selecciona: '))
if check_main_sleep == 1:
    check_main_sleep = "main"
elif check_main_sleep == 2:
    check_main_sleep = "not main"
elif check_main_sleep == 3:
    check_main_sleep = "both"

# time to add before start of sleep (in seconds) 30 minutes to 3 hours
print('Seleccione el rango de tiempo a agregar de forma aleatoria antes del inicio del sueño (en minutos):')
min_time_before_sleep = int(input('Mínimo: '))
max_time_before_sleep = int(input('Máximo: '))
time_before_sleep = random.randint(min_time_before_sleep * 60, max_time_before_sleep * 60)

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

def process_participant(person_data):
    print(f'\nStarting {person_data["participant"]}')
    # read the JSON file
    with open(f'data/pmdata/{person_data["participant"]}/fitbit/heart_rate.json', 'r') as f:
        data = f.read()

    # convert JSON to pandas dataframe
    hr_df = pd.read_json(data)

    # the value column is a dictionary, keep only the 'bpm' attribute on int type
    hr_df['bpm'] = hr_df['value'].apply(lambda x: x['bpm'])
    hr_df['confidence'] = hr_df['value'].apply(lambda x: x['confidence'])
    # delete the value column
    hr_df = hr_df.drop(columns=['value'])

    # read the JSON file
    with open(f'data/pmdata/{person_data["participant"]}/fitbit/sleep.json', 'r') as f:
        data = f.read()

    # convert JSON to pandas dataframe\
    sleep_df = pd.read_json(data)

    # delete from sleep_df the columns dateOfSleep, duration, minutesToFallAsleep, minutesAsleep, minutesAwake, minutesAfterWakeup, timeInBed, efficiency
    sleep_df = sleep_df.drop(columns=['dateOfSleep', 'duration', 'minutesToFallAsleep', 'minutesAsleep', 'minutesAwake', 'minutesAfterWakeup', 'timeInBed', 'efficiency'])

    # remaining sleep_df columns: logId, startTime, endTime, type, infoCode, levels, mainSleep

    # read the CSV file
    sleep_score_df = pd.read_csv(f'data/pmdata/{person_data["participant"]}/fitbit/sleep_score.csv')

    # read the CSV file
    wellness_df = pd.read_csv(f'data/pmdata/{person_data["participant"]}/pmsys/wellness.csv')

    # generate sleep_df profile report
    profile = ProfileReport(sleep_df, title='Sleep Profile Report', explorative=True)
    profile.to_file(f'data/pmdata/{person_data["participant"]}/fitbit/sleep_profile_report.html')

    # dataframe to store the new heart rate + sleep data
    hr_sleep_df = pd.DataFrame(columns=['participant', 'logId', 'phaseNo', 'hrNo', 'secondsPassed', 'daySeconds', 'bpm', 'level', 'resting_hr', 'age', 'height', 'sex', 'person', 'fatigue', 'mood', 'sleep_duration_h', 'sleep_quality', 'stress']) # height, max heart rate

    # iterate over the rows of sleep_df
    for index, row in sleep_df.iterrows():
        # compare the phase type with the check_sleep_type
        if row['type'] not in check_sleep_type.split(', '):
            continue
        # if check_only_infoCode_0 is True, check if the infoCode is 0
        if check_only_infoCode_0 == True:
            if row['infoCode'] != 0:
                continue
            if row['infoCode'] == 3:
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
        resting_hr = 0
        first_start_phase = None
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
            if first_start_phase is None:
                first_start_phase = start_phase
            end_phase = pd.to_datetime(phase['dateTime']) + pd.Timedelta(seconds=phase['seconds'])
            # create a new row for each hr_df row between the phae['dateTime'] and phase['dateTime'] + phase['seconds']
            # if no values are found, skip this phase
            between_hr = hr_df[(hr_df['dateTime'] >= start_phase) & (hr_df['dateTime'] <= end_phase) & (hr_df['confidence'] >= check_hr_confidence)]
            if len(between_hr) == 0:
                continue
            phase_hr = pd.DataFrame(columns=['participant', 'logId', 'phaseNo', 'hrNo', 'secondsPassed', 'daySeconds', 'bpm', 'level', 'resting_hr', 'age', 'height', 'sex', 'person', 'fatigue', 'mood', 'sleep_duration_h', 'sleep_quality', 'stress'])
            # add the number of row on between_hr
            for hr in between_hr.iterrows():
                timePassed = (hr[1]['dateTime'] - first_start_phase).total_seconds()
                phase_hr.loc[len(phase_hr)] = hr[1] # to set the bpm
                phase_hr.loc[len(phase_hr) - 1, 'secondsPassed'] = timePassed
                # secondsPassed to int
                phase_hr.loc[len(phase_hr) - 1, 'secondsPassed'] = int(phase_hr.loc[len(phase_hr) - 1, 'secondsPassed'])
                # seconds passed of the day based on the hour
                phase_hr.loc[len(phase_hr) - 1, 'daySeconds'] = pd.Timedelta(hours=hr[1]['dateTime'].hour, minutes=hr[1]['dateTime'].minute, seconds=hr[1]['dateTime'].second).total_seconds()
                # daySeconds to int
                phase_hr.loc[len(phase_hr) - 1, 'daySeconds'] = int(phase_hr.loc[len(phase_hr) - 1, 'daySeconds'])
            # add the new rows to the hr_sleep_df dataframe (use iloc)
            phase_hr.loc[:, 'participant'] = person_data["id"]
            phase_hr.loc[:, 'logId'] = row['logId']
            phase_hr.loc[:, 'level'] = get_level_value(phase['level'])
            sleep_score_log = sleep_score_df[sleep_score_df['sleep_log_entry_id'] == row['logId']]
            # si no hay sleep_score_log, usar el resting_hr anterior
            if len(sleep_score_log) == 0:
                if resting_hr == 0:
                    resting_hr = sleep_score_df['resting_heart_rate'].mean()
                phase_hr.loc[:, 'resting_hr'] = resting_hr
            else:
                resting_hr = sleep_score_log['resting_heart_rate'].values[0]
                if resting_hr:
                    phase_hr.loc[:, 'resting_hr'] = resting_hr
                else:
                    continue
            phase_hr.loc[:, 'age'] = person_data["age"]
            phase_hr.loc[:, 'height'] = person_data["height"]
            phase_hr.loc[:, 'sex'] = person_data["sex"]
            phase_hr.loc[:, 'person'] = person_data["person"]
            phase_hr.loc[:, 'phaseNo'] = phaseNo
            phase_hr.loc[:, 'hrNo'] = range(hrNoOffset, hrNoOffset + len(between_hr))
            hrNoOffset += len(between_hr)
            # obtener datos de wellness
            wellness_log = wellness_df[pd.to_datetime(wellness_df['effective_time_frame']).dt.date == pd.to_datetime(phase['dateTime']).date()]
            # si no hay wellness_log, usar la media de los datos
            if len(wellness_log) == 0:
                phase_hr.loc[:, 'fatigue'] = wellness_df['fatigue'].mean()
                phase_hr.loc[:, 'mood'] = wellness_df['mood'].mean()
                phase_hr.loc[:, 'sleep_duration_h'] = wellness_df['sleep_duration_h'].mean()
                phase_hr.loc[:, 'sleep_quality'] = wellness_df['sleep_quality'].mean()
                phase_hr.loc[:, 'stress'] = wellness_df['stress'].mean()
            else:
                phase_hr.loc[:, 'fatigue'] = wellness_log['fatigue'].values[0]
                phase_hr.loc[:, 'mood'] = wellness_log['mood'].values[0]
                phase_hr.loc[:, 'sleep_duration_h'] = wellness_log['sleep_duration_h'].values[0]
                phase_hr.loc[:, 'sleep_quality'] = wellness_log['sleep_quality'].values[0]
                phase_hr.loc[:, 'stress'] = wellness_log['stress'].values[0]
            # concat the new rows to the hr_sleep_df dataframed
            hr_sleep_df = pd.concat([hr_sleep_df, phase_hr])
            phaseNo += 1

    # write the hr_sleep_df dataframe to a csv file
    hr_sleep_df.to_csv(f'data/pmdata/all_hr_sleep_{person_data["participant"]}.csv', index=False)

    # generate sleep_df profile report
    profile = ProfileReport(hr_sleep_df, title='Heart Rate + Sleep Profile Report', explorative=True)
    profile.to_file(f'data/pmdata/hr_sleep_profile_report_{person_data["participant"]}.html')

threads = []

for person_data in participants_data:
    thread = threading.Thread(target=process_participant, args=(person_data,))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()

print('Done')