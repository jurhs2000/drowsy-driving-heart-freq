# Este archivo lee los datos de ECG de SHHS y calcula la frecuencia cardíaca promedio (BPM) en intervalos de tiempo específicos.
# Guarda los datos en la carpeta bpm como archivos CSV, el identificador es pptid

from pyhealth.datasets import SHHSDataset
import os
import matplotlib.pyplot as plt
import numpy as np
import pyedflib
import xml.etree.ElementTree as ET
import pyedflib
import pandas as pd
import scipy.signal as signal
from scipy.signal import find_peaks, butter, filtfilt
import math

plt.rcParams.update({'font.size': 14})  # Aumenta el tamaño de fuente general de las gráficas

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

'''def get_annotation_nsrr_path(signal_file):
  if 'shhs1' in signal_file:
    annotations_dir = os.path.join(data_path, 'annotations-events-nsrr', 'shhs1')
  elif 'shhs2' in signal_file:
    annotations_dir = os.path.join(data_path, 'annotations-events-nsrr', 'shhs2')
  else:
    return None
  return annotations_dir'''

for patient in dataset.patients:
  print(f'Patient: {patient}')

  # Obtener las grabaciones del paciente
  recordings = dataset.patients[patient]

  if len(recordings) > 1:
    print(f'Number of recordings: {len(recordings)}')

  # Leer la información del paciente de los archivos de anotaciones
  recording_count = 0
  for recording in recordings:
    signal_file = recording['signal_file']
    annotations_dir = get_annotation_profusion_path(signal_file)
    if annotations_dir:
      annotation_file = os.path.join(annotations_dir, os.path.basename(signal_file).replace('.edf', '-profusion.xml'))
      
      if os.path.exists(annotation_file):
        # Parsear el archivo XML de anotaciones
        tree = ET.parse(annotation_file)
        root = tree.getroot()
        # Extraer eventos anotados y fases del sueño
        sleep_stages = []
        for child in root:
          if child.tag == 'SleepStages':
            for stage in child:
              sleep_stages.append(stage.text)

    '''nsrr_annotations_dir = get_annotation_nsrr_path(signal_file)
    if nsrr_annotations_dir:
      nsrr_annotation_file = os.path.join(nsrr_annotations_dir, os.path.basename(signal_file).replace('.edf', '-nsrr.xml'))
      if os.path.exists(nsrr_annotation_file):
        # Parsear el archivo XML de anotaciones NSRR
        tree = ET.parse(nsrr_annotation_file)
        root = tree.getroot()
        # Extraer eventos anotados y fases del sueño
        sleep_stages_nsrr = []
        for child in root:
          if child.tag == 'ScoredEvents':
            for stage in child:
              for event in stage:
                if event.tag == 'EventType' and event.text == 'Stages|Stages':
                  sleep_stages_nsrr.append({
                    'Stage': stage[1].text,
                    'Start': float(stage[2].text),
                    'Duration': float(stage[3].text)
                  })'''

    # Leer y visualizar la señal de ECG
    edf_path = os.path.join(data_path, signal_file)
    
    # Leer el archivo EDF
    f = pyedflib.EdfReader(edf_path)
    n = f.signals_in_file
    print(f'Número de señales en el archivo EDF: {n}')
    signal_labels = f.getSignalLabels()
    print(f'Etiquetas de las señales: {signal_labels}')

    ecg_channel_indices = [i for i, name in enumerate(signal_labels) if 'ECG' in name]
    print(f'Índices de los canales de ECG: {ecg_channel_indices}')

    # Leer los datos del canal de ECG
    ecg_data = []
    for idx in ecg_channel_indices:
        ecg_data.append(f.readSignal(idx))
    print(f'ecg_data: {ecg_data}')
    
    # Asumir un canal de ECG (puedes ajustar esto según tus necesidades)
    ecg_signal = ecg_data[0]

    # Obtener la frecuencia de muestreo
    fs = f.getSampleFrequency(ecg_channel_indices[0])

    # Cerrar el archivo EDF
    f.close()

    # Crear la variable de tiempo (t)
    t = np.linspace(0, len(ecg_signal) / fs, len(ecg_signal))

    # preguntar si se desea graficar la señal de ECG
    plot_ecg = input('¿Desea graficar la señal de ECG? (s/n): ')
    
    if plot_ecg.lower() == 's':
      # graficar la señal de ECG antes de filtrar
      plt.figure(figsize=(12, 4))
      plt.plot(t, ecg_signal, label='ECG')
      plt.title('Señal de ECG sin filtrar', fontsize=18)
      plt.xlabel('Tiempo (s)', fontsize=16)
      plt.ylabel('Amplitud', fontsize=16)
      plt.legend(fontsize=16)
      plt.show()

    # Filtro de paso alto para eliminar la deriva de línea base
    highpass_cutoff = 0.5  # Frecuencia de corte en Hz
    b_high, a_high = butter(1, highpass_cutoff / (fs / 2), 'high')
    ecg_highpassed = filtfilt(b_high, a_high, ecg_signal)

    if plot_ecg.lower() == 's':
      # graficar la señal de ECG después de filtrar
      plt.figure(figsize=(12, 4))
      plt.plot(t, ecg_highpassed, label='ECG')
      plt.title('Señal de ECG después de filtro de paso alto')
      plt.xlabel('Tiempo (s)')
      plt.ylabel('Amplitud')
      plt.legend()
      plt.show()

    # Filtro de paso bajo para eliminar el ruido de alta frecuencia
    lowpass_cutoff = 45  # Frecuencia de corte en Hz
    lowpass_cutoff = min(lowpass_cutoff, fs / 2 - 1)  # Frecuencia de corte en Hz
    if lowpass_cutoff >= fs / 2:
      raise ValueError("La frecuencia de corte del filtro de paso bajo debe ser menor que la mitad de la frecuencia de muestreo.")

    b_low, a_low = butter(1, lowpass_cutoff / (fs / 2), 'low')
    ecg_filtered = filtfilt(b_low, a_low, ecg_highpassed)

    if plot_ecg.lower() == 's':
      # graficar la señal de ECG después de filtrar
      plt.figure(figsize=(12, 4))
      plt.plot(t, ecg_filtered, label='ECG')
      plt.title('Señal de ECG después de filtro de paso bajo')
      plt.xlabel('Tiempo (s)')
      plt.ylabel('Amplitud')
      plt.legend()
      plt.show()

    # Definir colores ajustados
    negro = (0/255, 0/255, 0/255)               # Negro
    gris_azulado = (92/255, 108/255, 170/255)   # Gris azulado
    gris_claro = (233/255, 205/255, 205/255)    # Gris claro

    # Suponiendo que tienes `t`, `ecg_signal`, `ecg_highpassed`, y `ecg_filtered` definidos
    if plot_ecg.lower() == 's':
        # Graficar la señal de ECG antes y después de filtrar, tanto el paso alto como el paso bajo
        plt.figure(figsize=(12, 4))
        plt.plot(t, ecg_signal, label='ECG sin filtrar', color=negro)
        plt.plot(t, ecg_highpassed, label='ECG después de filtro de paso alto', color=gris_azulado)
        plt.plot(t, ecg_filtered, label='ECG después de filtro de paso alto y paso bajo', color=gris_claro)
        plt.title('Señal de ECG antes y después de filtrar', fontsize=18)
        plt.xlabel('Tiempo (s)', fontsize=16)
        plt.ylabel('Amplitud', fontsize=16)
        plt.legend(fontsize=16)
        plt.show()

    # Derivar la señal
    ecg_diff = np.diff(ecg_filtered)
    ecg_diff = np.append(ecg_diff, 0)  # Ajustar longitud de la señal

    if plot_ecg.lower() == 's':
      # graficar la señal derivada
      plt.figure(figsize=(12, 4))
      plt.plot(t, ecg_diff, label='ECG derivado')
      plt.title('Señal de ECG derivada', fontsize=18)
      plt.xlabel('Tiempo (s)', fontsize=16)
      plt.ylabel('Amplitud', fontsize=16)
      plt.legend(fontsize=16)
      plt.show()

    # Cuadrar la señal derivada
    ecg_squared = ecg_diff ** 2

    if plot_ecg.lower() == 's':
      # graficar la señal derivada al cuadrado
      plt.figure(figsize=(12, 4))
      plt.plot(t, ecg_squared, label='ECG derivado al cuadrado')
      plt.title('Señal de ECG derivada al cuadrado', fontsize=18)
      plt.xlabel('Tiempo (s)', fontsize=16)
      plt.ylabel('Amplitud', fontsize=16)
      plt.legend(fontsize=16)
      plt.show()

    # Integrar la señal
    window_size = int(0.15 * fs)  # Tamaño de la ventana de integración (0.15 segundos)
    ecg_integrated = np.convolve(ecg_squared, np.ones(window_size) / window_size, mode='same')

    if plot_ecg.lower() == 's':
      # graficar la señal integrada
      plt.figure(figsize=(12, 4))
      plt.plot(t, ecg_integrated, label='ECG integrado')
      plt.title('Señal de ECG integrada', fontsize=18)
      plt.xlabel('Tiempo (s)', fontsize=16)
      plt.ylabel('Amplitud', fontsize=16)
      plt.legend(fontsize=16)
      plt.show()

    # Detección de picos usando scipy
    peaks, _ = find_peaks(ecg_integrated, distance=fs*0.46, height=0.002)  # Ajusta los parámetros según tu señal

    # Verificar si se detectaron picos
    if len(peaks) == 0:
      print("No se detectaron picos")
      continue

    if plot_ecg.lower() == 's':
      # graficar la señal integrada con los picos detectados
      plt.figure(figsize=(12, 4))
      plt.plot(t, ecg_integrated, label='ECG integrado')
      plt.plot(t[peaks], ecg_integrated[peaks], 'x', label='Picos detectados', color='red')
      plt.title('Señal de ECG integrada con picos detectados', fontsize=18)
      plt.xlabel('Tiempo (s)', fontsize=16)
      plt.ylabel('Amplitud', fontsize=16)
      plt.legend(fontsize=16)
      plt.show()

    # Calcular intervalos RR (diferencias entre picos)
    print(f'picos: {peaks}')
    print(f'diff peaks: {np.diff(peaks)}')
    print(f'fs: {fs}')
    rr_intervals = np.diff(peaks) / fs  # Intervalos en segundos
    print(f'rr_intervals: {rr_intervals}')

    # Calcular BPM (latidos por minuto)
    bpm = 60 / rr_intervals
    print(f'bpm: {bpm}')

    if plot_ecg.lower() == 's':
      # graficar los valores de BPM
      plt.figure(figsize=(12, 4))
      plt.plot(t[peaks[1:]], bpm, label='BPM')
      plt.title('Frecuencia cardíaca (BPM)', fontsize=18)
      plt.xlabel('Tiempo (s)', fontsize=16)
      plt.ylabel('BPM', fontsize=16)
      plt.legend(fontsize=16)
      plt.show()

    # Crear DataFrame con el tiempo y los valores de BPM
    peak_times = t[peaks[1:]]  # Tiempos de los picos (omitimos el primer pico porque no tiene intervalo previo)
    bpm_df = pd.DataFrame({'Time': peak_times, 'BPM': bpm})

    # Definir intervalo de tiempo para promediar BPM (por ejemplo, cada 5 segundos)
    interval = 5  # segundos
    bpm_df['Interval'] = (bpm_df['Time'] // interval).astype(int)

    # Promediar BPM en cada intervalo
    avg_bpm_df = bpm_df.groupby('Interval').mean().reset_index()
    avg_bpm_df = avg_bpm_df[['Interval', 'BPM']]  # Mantener solo columnas relevantes
    avg_bpm_df['Time'] = avg_bpm_df['Interval'] * interval

    # Agregar las fases de sueño (sleep_stages)
    stages_per_interval = 30 // interval  # Número de intervalos de BPM por cada fase de sueño
    sleep_stages_repeated = np.repeat(sleep_stages, stages_per_interval)
    avg_bpm_df['SleepStage'] = sleep_stages_repeated[:len(avg_bpm_df)]

    # Dato de BPM con solo 2 decimales
    avg_bpm_df['BPM'] = avg_bpm_df['BPM'].round(2)

    # Crear un array para las fases de sueño NSRR
    # no se agrega porque es el mismo valor que SleepStage
    '''total_time = avg_bpm_df['Time'].max() + interval  # Tiempo total de grabación
    if math.isnan(total_time):
      print('Error: No se pudo calcular el tiempo total de grabación.')
      continue
    if math.isnan(interval):
      print('Error: No se pudo calcular el intervalo de tiempo.')
      continue
    sleep_stage_nsrr_array = np.zeros(int(total_time // interval), dtype=object)

    # Llenar el array de fases de sueño NSRR
    for stage in sleep_stages_nsrr:
      start_interval = int(stage['Start'] // interval)
      end_interval = int((stage['Start'] + stage['Duration']) // interval)
      sleep_stage_nsrr_array[start_interval:end_interval] = stage['Stage']

    # Agregar las fases de sueño NSRR a avg_bpm_df
    avg_bpm_df['SleepStage_NS'] = sleep_stage_nsrr_array[:len(avg_bpm_df)]'''

    # Si existe mas de 1/3 de datos de BPM por debajo de 40 o por encima de 150 (en conjunto), se descarta el archivo
    if len(avg_bpm_df[(avg_bpm_df['BPM'] < 40) | (avg_bpm_df['BPM'] > 150)]) > len(avg_bpm_df) / 3:
      print(f'Archivo descartado: {signal_file}')
      continue

    # crear un directorio para guardar los archivos CSV si no existe
    if not os.path.exists('bpm'):
      os.makedirs('bpm')

    # Guardar el DataFrame en un archivo CSV
    output_csv_path = f'bpm/bpm_data_{patient}_{recording_count}.csv'
    recording_count += 1
    avg_bpm_df.to_csv(output_csv_path, index=False)