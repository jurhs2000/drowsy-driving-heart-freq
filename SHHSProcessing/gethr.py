# Este archivo lee los datos de H.R. de SHHS y calcula la frecuencia cardíaca promedio (BPM) en intervalos de tiempo específicos.
# Guarda los datos en la carpeta bpm como archivos CSV, el identificador es pptid

from pyhealth.datasets import SHHSDataset
import os
import matplotlib.pyplot as plt
import numpy as np
import pyedflib
import xml.etree.ElementTree as ET
import pandas as pd

# Ajusta el camino hacia la carpeta de datos
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_path = os.path.join(base_path, 'shhs', 'polysomnography')

# Cargar el dataset
dataset = SHHSDataset(root=data_path)
dataset.stat()
dataset.info()

def get_annotation_profusion_path(signal_file):
    if 'shhs1' in signal_file:
        annotations_dir = os.path.join(data_path, 'annotations-events-profusion', 'shhs1')
    elif 'shhs2' in signal_file:
        annotations_dir = os.path.join(data_path, 'annotations-events-profusion', 'shhs2')
    else:
        return None
    return annotations_dir

for patient in dataset.patients:
    print(f'Patient: {patient}')
    recordings = dataset.patients[patient]

    if len(recordings) > 1:
        print(f'Number of recordings: {len(recordings)}')

    recording_count = 0
    for recording in recordings:
        signal_file = recording['signal_file']
        annotations_dir = get_annotation_profusion_path(signal_file)
        if annotations_dir:
            annotation_file = os.path.join(annotations_dir, os.path.basename(signal_file).replace('.edf', '-profusion.xml'))
            if os.path.exists(annotation_file):
                tree = ET.parse(annotation_file)
                root = tree.getroot()
                sleep_stages = []
                for child in root:
                    if child.tag == 'SleepStages':
                        for stage in child:
                            sleep_stages.append(int(stage.text))  # Convertir a entero

        edf_path = os.path.join(data_path, signal_file)
        f = pyedflib.EdfReader(edf_path)
        hr_channel_indices = [i for i, name in enumerate(f.getSignalLabels()) if 'H.R.' in name]

        hr_data = []
        for idx in hr_channel_indices:
            hr_data.append(f.readSignal(idx))
        hr_signal = hr_data[0]
        fs = f.getSampleFrequency(hr_channel_indices[0])
        f.close()

        t = np.arange(0, len(hr_signal) / fs, 1)  # Generar el tiempo en segundos

        # Filtrar cada 5 segundos
        indices_5s = (t % 5 == 0)
        hr_filtered = hr_signal[indices_5s]
        time_filtered = t[indices_5s]

        # Crear el DataFrame con BPM y SleepStage
        bpm_df = pd.DataFrame({
            'Interval': range(len(hr_filtered)),
            'Time': time_filtered.astype(int),   # Convertir a entero
            'BPM': hr_filtered.round().astype(int)  # Redondear y convertir a entero
        })

        # Asignar y transformar las fases de sueño
        if sleep_stages:
            sleep_stages_filtered = [sleep_stages[int(time // 30)] if int(time // 30) < len(sleep_stages) else None for time in time_filtered]
            bpm_df['SleepStage'] = sleep_stages_filtered
            bpm_df['SleepStage'].replace({3: 2, 4: 2, 5: 2, 9: 0}, inplace=True)
            bpm_df = bpm_df[bpm_df['SleepStage'] != 6]

        # Filtro de calidad: descartar archivos con BPMs anómalos
        if len(bpm_df[(bpm_df['BPM'] < 40) | (bpm_df['BPM'] > 150)]) > len(bpm_df) / 3:
            print(f'Archivo descartado: {signal_file}')
            continue

        # Guardar el DataFrame en un archivo CSV
        if not os.path.exists('bpm'):
            os.makedirs('bpm')
        
        output_csv_path = f'bpm/bpm_data_{patient}_{recording_count}.csv'
        recording_count += 1
        bpm_df.to_csv(output_csv_path, index=False)
        print(f'Datos guardados en {output_csv_path}')
