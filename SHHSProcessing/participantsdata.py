# load the merged_shhs_data CSV file and display the first few rows of the DataFrame.
# this is located on the parent directory of the current file.
# Este archivo utiliza el archivo merged_shhs_data.csv para extraer los datos de los participantes y guardarlos en un nuevo archivo CSV.
# El archivo merged_shhs_data.csv se generó previamente con el archivo readallarchives.py.
# Los datos extraídos incluyen la identificación del participante, la edad, la altura, la fatiga, la duración del sueño, la calidad del sueño y el estrés.

import os
import pandas as pd

file_dir = os.path.dirname(__file__)
base_path = os.path.abspath(os.path.join(file_dir, '..'))
merged_data_path = os.path.join(base_path, 'merged_shhs_data.csv')

if os.path.exists(merged_data_path):
    merged_data = pd.read_csv(merged_data_path)

    # level (sleep phase)

    # Resting heart rate - no existe
    #print(merged_data[[col for col in merged_data.columns if 'savbrbh' in col]].head()) # Average heart rate during REM sleep in supine position from type II polysomnography
    #print(merged_data['savbrbh'].count()) # 1870

    # Age
    # print all the columns that contains "age" in the name
    # join "age_category_s1" and "age_category_s2" columns prioritazing the not NaN values
    merged_data['age'] = merged_data['age_s1'].combine_first(merged_data['age_s2'])
    print(merged_data['age'].head())
    # Count the number of age values in the DataFrame
    print(merged_data['age'].count()) # 5804
    print(merged_data['nsrrid'].count()) # 5804

    # Height
    print(merged_data[[col for col in merged_data.columns if 'height' in col]].head()) # height
    print(merged_data['height'].count()) # 5762
    #print(merged_data[[col for col in merged_data.columns if 'pm207' in col]].head()) # height
    #print(merged_data['pm207'].count()) # dfdsdfs

    # Sex
    print(merged_data[[col for col in merged_data.columns if 'gender' in col]].head()) # gender
    print(merged_data['gender'].count()) # 4080

    # Person (type) - No existe

    # Fatigue
    # search for a column
    #print(merged_data[[col for col in merged_data.columns if 'sleepy02' in col]].head()) # Frequency of excessive daytime sleepiness
    #print(merged_data['sleepy02'].count()) # 5721
    #print(merged_data[[col for col in merged_data.columns if 'sh308e' in col]].head()) # Frequency of excessive daytime sleepiness
    #print(merged_data['sh308e'].count()) # 4029
    #merged_data['fatigue1'] = merged_data['sleepy02'].combine_first(merged_data['sh308e'])
    #print(merged_data['fatigue1'].count()) # 5754

    print(merged_data[[col for col in merged_data.columns if 'tired25' in col]].head()) # Feel tired SHHS1
    print(merged_data['tired25'].count()) # 5362
    print(merged_data[[col for col in merged_data.columns if 'ql209i' in col]].head()) # Feel tired SHHS2
    print(merged_data['ql209i'].count()) # 3532
    merged_data['fatigue'] = merged_data['tired25'].combine_first(merged_data['ql209i'])
    print(merged_data['fatigue'].count()) # 5591

    #print(merged_data[[col for col in merged_data.columns if 'drive02' in col]].head()) # Fall asleep while driving SHHS1
    #print(merged_data['drive02'].count()) # 5685
    #print(merged_data[[col for col in merged_data.columns if 'sh319j' in col]].head()) # Chance of dozing while driving SHHS2
    #print(merged_data['sh319j'].count()) # 4020

    # Mood - No existe

    # Sleep duration hours
    # search for a column
    print(merged_data[[col for col in merged_data.columns if 'hwlghr10' in col]].head())
    print(merged_data[[col for col in merged_data.columns if 'rptacttimslp' in col]].head()) # No existe
    # count the number of hwlghr10 values in the DataFrame
    print(merged_data['hwlghr10'].count()) # hay menos de este dato que de nsrrid, 5622 (182 menos)
    # rename hwlghr10 to sleep_duration
    merged_data['sleep_duration'] = merged_data['hwlghr10']
    print(merged_data['nsrrid'].count()) # 5804

    # Sleep quality
    # search for a column
    #print(merged_data[[col for col in merged_data.columns if 'ms204a' in col]].head())
    #print(merged_data[[col for col in merged_data.columns if 'ms204b' in col]].head())
    #print(merged_data[[col for col in merged_data.columns if 'ms204c' in col]].head())
    #print(merged_data[[col for col in merged_data.columns if 'shlg10' in col]].head())
    #print(merged_data[[col for col in merged_data.columns if 'rest10' in col]].head())
    #print(merged_data[[col for col in merged_data.columns if 'ltdp10' in col]].head())

    print(merged_data[[col for col in merged_data.columns if 'hwwell10' in col]].head()) # Quality of sleep compared to usual SHHS1
    print(merged_data[[col for col in merged_data.columns if 'ms205' in col]].head()) # Quality of sleep compared to usual SHHS2
    print(merged_data[[col for col in merged_data.columns if 'hi205' in col]].head()) # how well slept last night
    print(merged_data['hwwell10'].count()) # 5677
    print(merged_data['ms205'].count()) # 2678
    print(merged_data['hi205'].count()) # 3623

    merged_data['sleep_quality'] = merged_data['hwwell10'].combine_first(merged_data['ms205']).combine_first(merged_data['hi205'])
    print(merged_data['sleep_quality'].count()) # 5764

    # Stress
    # unify the two columns called 'stress15' and 'hi207' into a new column called 'stress'
    merged_data['stress'] = merged_data['stress15'].combine_first(merged_data['hi207'])
    print(merged_data['stress'].head())

    # left only the columns "nsrrid", "age", "height", "fatigue", "sleep_duration", "sleep_quality", "stress"
    merged_data = merged_data[['nsrrid', 'age', 'height', 'fatigue', 'sleep_duration', 'sleep_quality', 'stress', 'gender']]
    
    # remove rows with NaN values on the columns "age", "height", "fatigue", "sleep_duration", "sleep_quality", "stress"
    merged_data = merged_data.dropna(subset=['age', 'height', 'fatigue', 'sleep_duration', 'sleep_quality', 'stress', 'gender'])

    merged_data['age'] = merged_data['age'].astype(int)
    merged_data['height'] = merged_data['height'].astype(int)
    merged_data['fatigue'] = merged_data['fatigue'].astype(int)
    merged_data['sleep_duration'] = merged_data['sleep_duration'].astype(int)
    merged_data['sleep_quality'] = merged_data['sleep_quality'].astype(int)
    merged_data['stress'] = merged_data['stress'].astype(int)
    merged_data['gender'] = merged_data['gender'].astype(int)
    print(merged_data.head())
    print(merged_data.count())

    # remove rows with NaN values
    merged_data = merged_data.dropna()
    print(merged_data.count())

    # check the min and max values of the columns
    print('Min and Max values of the columns:')
    print('min:')
    print(merged_data[['age', 'height', 'fatigue', 'sleep_duration', 'sleep_quality', 'stress', 'gender']].min())
    print('max:')
    print(merged_data[['age', 'height', 'fatigue', 'sleep_duration', 'sleep_quality', 'stress', 'gender']].max())

    # Normalizar la escala de 1 a 6 a una escala de 1 a 5
    merged_data['fatigue'] = merged_data['fatigue'].apply(lambda x: round((x - 1) * (5 - 1) / (6 - 1) + 1))
    merged_data['sleep_quality'] = merged_data['sleep_quality'].apply(lambda x: round((x - 1) * (5 - 1) / (6 - 1) + 1))
    merged_data['stress'] = merged_data['stress'].apply(lambda x: round((x - 1) * (5 - 1) / (6 - 1) + 1))

    # save the new DataFrame into a CSV file
    new_data_path = os.path.join(base_path, 'participants_data.csv')
    merged_data.to_csv(new_data_path, index=False)
    print(f"El archivo {new_data_path} ha sido guardado.")
else:
    print(f"El archivo {merged_data_path} no existe.")
