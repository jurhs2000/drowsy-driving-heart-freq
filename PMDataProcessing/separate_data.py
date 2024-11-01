# Este programa separara todos los datos estaticos de los 16 pacientes de PMData en un archivo .csv
# y deja una carpeta con archivos csv individuales con los datos de BPM para cada paciente

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Se crea un dataframe con los datos de los 16 pacientes
df = pd.DataFrame()

for i in range(1,17):
    # if patient is 12, skip
    if i == 12:
        continue
    # patient dataframe, use two digits
    patientdf = pd.read_csv(f'all_hr_sleep_p{i:02d}.csv')
    print(f'Patient {i} shape: {patientdf.shape}')

    # show the columns
    print(patientdf.columns)

    # count the number of differents logId in the bpm data
    print(patientdf['logId'].nunique())

    # Crear un diccionario de mapeo desde los valores únicos de logId a un rango comenzando desde 1
    unique_ids = {log_id: idx+1 for idx, log_id in enumerate(sorted(patientdf['logId'].unique()))}
    print(unique_ids)
    # Mapear los valores de logId a los nuevos valores únicos
    patientdf['logId'] = patientdf['logId'].map(unique_ids)
    print(patientdf['logId'])

    # save the bpm data into bpm folder
    bpm_df = patientdf[['participant', 'logId', 'hrNo', 'secondsPassed', 'bpm', 'level']]

    # rename the patient column to pptid
    bpm_df = bpm_df.rename(columns={'participant': 'pptid'})
    # rename the hrNo column to Interval
    bpm_df = bpm_df.rename(columns={'hrNo': 'Interval'})
    # rename the secondsPassed column to Time
    bpm_df = bpm_df.rename(columns={'secondsPassed': 'Time'})
    # rename the bpm column to BPM
    bpm_df = bpm_df.rename(columns={'bpm': 'BPM'})
    # rename the level column to SleepStage
    bpm_df = bpm_df.rename(columns={'level': 'SleepStage'})

    # save the bpm data into a csv file different for each logId
    for logId in bpm_df['logId'].unique():
        # get the bpm data for the logId
        bpm_logId = bpm_df[bpm_df['logId'] == logId]
        # delete the logId column
        bpm_logId = bpm_logId.drop(columns=['logId'])
        # delete the pptid column
        bpm_logId = bpm_logId.drop(columns=['pptid'])
        # save the bpm data into a csv file
        bpm_logId.to_csv(f'bpm/bpm_data_{i:02d}_{logId}.csv', index=False)

    # save the static data into the main dataframe
    static_df = patientdf[['participant', 'logId', 'age', 'height', 'sex', 'fatigue', 'sleep_duration_h', 'sleep_quality', 'stress']]
    # change sex column to gender
    static_df = static_df.rename(columns={'sex': 'gender'})
    static_df['gender'] = static_df['gender'].map({0: 1, 1: 2})
    static_df = static_df.rename(columns={'participant': 'pptid'})
    static_df = static_df.rename(columns={'sleep_duration_h': 'sleep_duration'})

    # remove all the duplicates (all rows are the same) and keep only one row
    static_df = static_df.drop_duplicates()

    # make int without decimal, all the age, height, fatigue, sleep_duration, sleep_quality and stress columns
    static_df['age'] = static_df['age'].astype(int)
    static_df['height'] = static_df['height'].astype(int)
    static_df['fatigue'] = static_df['fatigue'].astype(int)
    static_df['sleep_duration'] = static_df['sleep_duration'].astype(int)
    static_df['sleep_quality'] = static_df['sleep_quality'].astype(int)
    static_df['stress'] = static_df['stress'].astype(int)

    # add the static data to the main dataframe
    df = pd.concat([df, static_df])

# save the main dataframe
df.to_csv('participants_data.csv', index=False)

# show the main dataframe
print(df.shape)
print(df.columns)
print(df.head())
