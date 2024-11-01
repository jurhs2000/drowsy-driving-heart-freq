import pandas as pd
import random
from ydata_profiling import ProfileReport
import threading
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

random.seed(42)

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
print('\nSeleccione si se tomarán en cuenta las fases de sueño profundo:')
print('1: Incluir fases de sueño profundo')
print('2: No incluir fases de sueño profundo')
check_sleep_level = int(input('Selecciona: '))
if check_sleep_level == 1:
    check_sleep_level = True
elif check_sleep_level == 2:
    check_sleep_level = False

# will consider only the first cycle, until the first "asleep" or "deep" or "rem" phase
print('\nSeleccione si se tomará en cuenta solo el primer ciclo de sueño, es decir, cuando termine la primera fase de sueño profundo:')
print('1: Solo el primer ciclo')
print('2: Todos los ciclos')
check_only_first_cycle = int(input('Selecciona: '))
if check_only_first_cycle == 1:
    check_only_first_cycle = True
elif check_only_first_cycle == 2:
    check_only_first_cycle = False

# will use only phase with infoCode 0, that means that are sufficient data to determine sleep stage
print('\nSeleccione si se usarán solamente los logs de siestas (menor a 3 horas):')
print('1: Solo siestas')
print('2: Siestas y sueños largos')
check_only_naps = int(input('Selecciona: '))
if check_only_naps == 1:
    check_only_naps = True
elif check_only_naps == 2:
    check_only_naps = False

print("\nSeleccione si desea solo crear el profile report o solo el csv:")
print("1: Solo crear csv")
print("2: Solo crear profile report")
print("3: Crear csv y profile report")
create_profile_report = int(input("Selecciona: "))

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

def get_composed_dataframe(between_hr, first_start_phase, level, hrNoOffset):
    phase_hr = pd.DataFrame(columns=['hrNo', 'secondsPassed', 'daySeconds', 'bpm', 'level'])
    # add the number of row on between_hr
    for hr in between_hr.iterrows():
        timePassed = (hr[1]['dateTime'] - first_start_phase).total_seconds()
        phase_hr.loc[len(phase_hr)] = hr[1] # to set the bpm
        phase_hr.loc[len(phase_hr) - 1, 'secondsPassed'] = int(timePassed)
        # seconds passed of the day based on the hour
        phase_hr.loc[len(phase_hr) - 1, 'daySeconds'] = int(pd.Timedelta(hours=hr[1]['dateTime'].hour, minutes=hr[1]['dateTime'].minute, seconds=hr[1]['dateTime'].second).total_seconds())
    # add the new rows to the sleep_log_df dataframe (use iloc)
    phase_hr.loc[:, 'level'] = level
    phase_hr.loc[:, 'hrNo'] = range(hrNoOffset, hrNoOffset + len(between_hr))
    return phase_hr

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

    # remaining hr_df columns: dateTime, bpm, confidence

    # read the JSON file
    with open(f'data/pmdata/{person_data["participant"]}/fitbit/sleep.json', 'r') as f:
        data = f.read()

    # convert JSON to pandas dataframe\
    sleep_df = pd.read_json(data)

    # delete from sleep_df the non useful columns
    sleep_df = sleep_df.drop(columns=['dateOfSleep', 'startTime', 'endTime', 'duration', 'minutesToFallAsleep', 'minutesAsleep', 'minutesAwake', 'minutesAfterWakeup', 'timeInBed', 'efficiency', 'mainSleep'])

    # remaining sleep_df columns: logId, type, infoCode, levels

    # read the CSV file
    sleep_score_df = pd.read_csv(f'data/pmdata/{person_data["participant"]}/fitbit/sleep_score.csv')

    sleep_score_df = sleep_score_df.drop(columns=['timestamp', 'overall_score', 'composition_score', 'revitalization_score', 'duration_score', 'deep_sleep_in_minutes', 'restlessness'])

    # remaining sleep_score_df columns: sleep_log_entry_id, resting_heart_rate

    # read the CSV file
    wellness_df = pd.read_csv(f'data/pmdata/{person_data["participant"]}/pmsys/wellness.csv')

    # delete from wellness_df the non useful columns
    wellness_df = wellness_df.drop(columns=['readiness', 'soreness', 'soreness_area'])

    # remaining wellness_df columns: effective_time_frame, fatigue, mood, sleep_duration_h, sleep_quality, stress

    # dataframe to store the new heart rate + sleep data + resting heart rate + wellness data
    hr_sleep_df = pd.DataFrame(columns=['participant', 'logId', 'hrNo', 'secondsPassed', 'daySeconds', 'bpm', 'level', 'resting_hr', 'age', 'height', 'sex', 'person', 'fatigue', 'mood', 'sleep_duration_h', 'sleep_quality', 'stress']) # height, max heart rate

    # iterate over the rows of sleep_df
    for _, row in sleep_df.iterrows():
        # if infoCode is 1 (no sufficient heart rate data) or 3 (server error), pass
        if row['infoCode'] == 1 or row['infoCode'] == 3:
            continue
        # if check_only_naps is True, check if the infoCode is 0 (no nap) to pass
        if check_only_naps == True:
            if row['infoCode'] == 0:
                continue
        # if already exists a row with the same logId, pass
        if len(hr_sleep_df[hr_sleep_df['logId'] == row['logId']]) > 0:
            continue
        # sleep_log_df dataframe to store the rows of the complete sleep log, all phases
        sleep_log_df = pd.DataFrame(columns=['participant', 'logId', 'hrNo', 'secondsPassed', 'daySeconds', 'bpm', 'level', 'resting_hr', 'age', 'height', 'sex', 'person', 'fatigue', 'mood', 'sleep_duration_h', 'sleep_quality', 'stress'])
        first_cycle_complete = False
        is_first_phase = True
        hrNoOffset = 0
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
            # si es la primera fase y es de sueño profundo, romper el ciclo
            if is_first_phase == True and get_level_value(phase['level']) == 2:
                break
            # get the datetime of the start and end of the phase
            start_phase = pd.to_datetime(phase['dateTime'])
            if is_first_phase == True:
                is_first_phase = False
                min_minutes = 8 # cantidad de minutos mínima a agregar antes de la primera fase
                max_minutes = 30 # cantidad de minutos máxima a agregar antes de la primera fase
                time_before_sleep = random.randint(min_minutes * 60, max_minutes * 60)
                start_phase_previous = pd.to_datetime(phase['dateTime']) - pd.Timedelta(seconds=time_before_sleep)
                first_start_phase = start_phase_previous
                between_hr = hr_df[(hr_df['dateTime'] >= first_start_phase) & (hr_df['dateTime'] <= start_phase) & (hr_df['confidence'] >= 3)]
                if len(between_hr) == 0:
                    break
                phase_hr = get_composed_dataframe(between_hr, first_start_phase, 0, hrNoOffset)
                hrNoOffset += len(phase_hr)
                # concat the previous rows to the sleep_log_df dataframe
                sleep_log_df = pd.concat([sleep_log_df, phase_hr])
            # continue with the current phase after add the previous rows
            end_phase = pd.to_datetime(phase['dateTime']) + pd.Timedelta(seconds=phase['seconds'])
            # create a new row for each hr_df row between the phae['dateTime'] and phase['dateTime'] + phase['seconds']
            # if no values are found, skip this phase
            # only hr logs with high confidence (3) are considered
            between_hr = hr_df[(hr_df['dateTime'] >= start_phase) & (hr_df['dateTime'] <= end_phase) & (hr_df['confidence'] >= 3)]
            if len(between_hr) == 0:
                # clear the sleep_log_df dataframe
                sleep_log_df = pd.DataFrame(columns=sleep_log_df.columns)
                break
            phase_hr = get_composed_dataframe(between_hr, first_start_phase, get_level_value(phase['level']), hrNoOffset)
            hrNoOffset += len(phase_hr)
            # concat the new rows to the sleep_log_df dataframe
            sleep_log_df = pd.concat([sleep_log_df, phase_hr])
        # end phases for
        # si no hay nungun resultado proveniente de las fases, pasar a la siguiente iteración
        if len(sleep_log_df) == 0:
            continue
        # si en la columna 'level' hay solo un valor, pasar a la siguiente iteración
        if len(sleep_log_df['level'].unique()) == 1:
            continue
        # sleep_score_log dataframe to store the rows of the sleep score log matching the current sleep log
        sleep_score_log = sleep_score_df[sleep_score_df['sleep_log_entry_id'] == row['logId']]
        # si no hay match con el logId, usar la media de los datos
        if len(sleep_score_log) == 0:
            sleep_log_df.loc[:, 'resting_hr'] = sleep_score_df['resting_heart_rate'].mean()
        else:
            sleep_log_df.loc[:, 'resting_hr'] = sleep_score_log['resting_heart_rate'].values[0]
        sleep_log_df.loc[:, 'logId'] = row['logId']
        sleep_log_df.loc[:, 'participant'] = person_data["id"]
        sleep_log_df.loc[:, 'age'] = person_data["age"]
        sleep_log_df.loc[:, 'height'] = person_data["height"]
        sleep_log_df.loc[:, 'sex'] = person_data["sex"]
        sleep_log_df.loc[:, 'person'] = person_data["person"]
        if first_start_phase is None:
            print('EEEEEEEEEEEEERRRRRRRRRRRRRRRRROOOOOOOOOOOOOOOOOOOOORRRRRRRRRRRRRRRRRR: first_start_phase is None')
        # obtener datos de wellness
        wellness_log = wellness_df[pd.to_datetime(wellness_df['effective_time_frame']).dt.date == first_start_phase.date()]
        # si no hay wellness_log, usar la media de los datos
        if len(wellness_log) == 0:
            sleep_log_df.loc[:, 'fatigue'] = wellness_df['fatigue'].mean()
            sleep_log_df.loc[:, 'mood'] = wellness_df['mood'].mean()
            sleep_log_df.loc[:, 'sleep_duration_h'] = wellness_df['sleep_duration_h'].mean()
            sleep_log_df.loc[:, 'sleep_quality'] = wellness_df['sleep_quality'].mean()
            sleep_log_df.loc[:, 'stress'] = wellness_df['stress'].mean()
        else:
            sleep_log_df.loc[:, 'fatigue'] = wellness_log['fatigue'].values[0]
            sleep_log_df.loc[:, 'mood'] = wellness_log['mood'].values[0]
            sleep_log_df.loc[:, 'sleep_duration_h'] = wellness_log['sleep_duration_h'].values[0]
            sleep_log_df.loc[:, 'sleep_quality'] = wellness_log['sleep_quality'].values[0]
            sleep_log_df.loc[:, 'stress'] = wellness_log['stress'].values[0]
        # add the new rows to the hr_sleep_df dataframe (use iloc)
        hr_sleep_df = pd.concat([hr_sleep_df, sleep_log_df])

    # write the hr_sleep_df dataframe to a csv file
    hr_sleep_df.to_csv(f'data/pmdata/all_hr_sleep_{person_data["participant"]}.csv', index=False)

    # generate sleep_df profile report
    if create_profile_report == 3:
        profile = ProfileReport(hr_sleep_df, title='Heart Rate + Sleep Profile Report', explorative=True)
        profile.to_file(f'data/pmdata/hr_sleep_profile_report_{person_data["participant"]}.html')
    print(f'Finished {person_data["participant"]}')

'''threads = []

for person_data in participants_data:
    thread = threading.Thread(target=process_participant, args=(person_data,))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()'''

participant_no = int(input('Seleccione el número del participante a procesar: '))
if create_profile_report == 1 or create_profile_report == 3:
    process_participant(participants_data[participant_no - 1])
elif create_profile_report == 2:
    hr_sleep_df = pd.read_csv(f'data/pmdata/all_hr_sleep_{participants_data[participant_no - 1]["participant"]}.csv')
    profile = ProfileReport(hr_sleep_df, title='Heart Rate + Sleep Profile Report', explorative=True)
    profile.to_file(f'data/pmdata/hr_sleep_profile_report_{participants_data[participant_no - 1]["participant"]}.html')

print('Done')