import pandas as pd
import os

########### CARGANDO DATASETS ###########

# load the PMData participants data
PMpd = pd.read_csv('PMData/participants_data.csv')
PMpd['id'] = PMpd['pptid']
PMpd.drop(columns=['pptid'], inplace=True)

# load the PMData bpm data
PMbpm = None
# read all the files in the PMData/bpm folder
for file in os.listdir('PMData/bpm'):
    if file.endswith('.csv'):
        # load the file
        df = pd.read_csv('PMData/bpm/' + file)
        ids = int(file.split('_')[2])
        logId = int(file.split('_')[3].split('.')[0])
        # add the id and logId columns
        df['id'] = ids
        df['logId'] = logId
        # add the file to the PMbpm dataframe
        if PMbpm is None:
            PMbpm = df
        else:
            # concat the dataframes
            PMbpm = pd.concat([PMbpm, df])
# reset the index
PMbpm.reset_index(drop=True, inplace=True)

# load the SHHS participants data
SHHSpd = pd.read_csv('SHHS/participants_data.csv')
SHHSpd['id'] = SHHSpd['nsrrid']
SHHSpd.drop(columns=['nsrrid'], inplace=True)

# load the SHHS bpm data
SHHSbpm = None
# read all the files in the SHHS/bpm folder
files_len = len(os.listdir('SHHS/bpm'))
count = 0
for file in os.listdir('SHHS/bpm'):
    if file.endswith('.csv'):
        print(f'{count}/{files_len} - {file}')
        # load the file
        df = pd.read_csv('SHHS/bpm/' + file)
        ids = int(file.split('_')[2])
        logId = int(file.split('_')[3].split('.')[0])
        # add the id and logId columns
        df['id'] = ids
        df['logId'] = logId
        # add the file to the SHHSbpm dataframe
        if SHHSbpm is None:
            SHHSbpm = df
        elif not df.isna().all().all():
            # concat the dataframes
            SHHSbpm = pd.concat([SHHSbpm, df])
        count += 1
# reset the index
SHHSbpm.reset_index(drop=True, inplace=True)

############ UNIENDO DATAFRAMES ############
print('Uniendo dataframes')

print(PMpd.head())
print(PMbpm.head())
print(SHHSpd.head())
print(SHHSbpm.head())

# merge the dataframes

PMdf = pd.merge(PMpd, PMbpm, on=['id', 'logId'])
SHHSdf = pd.merge(SHHSpd, SHHSbpm, on=['id'])

print(PMdf.head())
print(SHHSdf.head())

# crear la carpeta data si no existe
if not os.path.exists('data'):
    os.makedirs('data')

# save the dataframes
PMdf.to_csv('data/PMData.csv', index=False)
SHHSdf.to_csv('data/SHHS.csv', index=False)

df = pd.concat([PMdf, SHHSdf])

print(df.head())

# save the dataframe
df.to_csv('data/df.csv', index=False)
