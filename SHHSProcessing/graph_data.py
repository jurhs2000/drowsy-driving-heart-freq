import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.lines import Line2D

i = 1
# recorrer los archivos de la carpeta /bpm
dir = 'bpm'

# Definir colores ajustados
negro = (0/255, 0/255, 0/255)               # Negro
gris_azulado = (92/255, 108/255, 170/255)  # Gris azulado, similar al color "Slate Gray"
gris_claro = (233/255, 205/255, 205/255)

for file in os.listdir(dir):
    print(f'logId {file} ({i}/{len(os.listdir(dir))})')

    # abrir el archivo
    df = pd.read_csv(f'{dir}/{file}')

    hr_sleep_df = pd.DataFrame(columns=df.columns)

    last_level = None
    for hr in df.iterrows():
        level = hr[1]['SleepStage']
        if level != last_level:
            if last_level is not None:
                # Asignar estilo de línea según el nivel de sueño
                linestyle = '-' if last_level == 0 else '--' if last_level == 1 else ':'

                # Graficar la fase anterior con color y estilo correspondiente
                plt.plot(
                    hr_sleep_df['Time'],
                    hr_sleep_df['BPM'],
                    color=negro if last_level == 0 else gris_claro if last_level == 1 else gris_azulado,
                    linestyle=linestyle,
                    linewidth=1.5
                )
                hr_sleep_df = pd.DataFrame(columns=df.columns)  # Reiniciar el DataFrame

        hr_sleep_df.loc[len(hr_sleep_df)] = hr[1]  # Agregar fila
        last_level = level  # Actualizar nivel

    # Graficar la última fase
    linestyle = '-' if last_level == 0 else '--' if last_level == 1 else ':'
    plt.plot(
        hr_sleep_df['Time'],
        hr_sleep_df['BPM'],
        color=negro if last_level == 0 else gris_claro if last_level == 1 else gris_azulado,
        linestyle=linestyle,
        linewidth=1.5
    )

    # Título y etiquetas
    participant = file.split('_')[2]
    logId = file.split('_')[3].split('.')[0]
    plt.title(f'Participant {participant}\nHeart Rate vs Time for logId {logId}', fontsize=18)
    plt.xlabel('Time (seconds)', fontsize=16)
    plt.ylabel('Heart Rate (BPM)', fontsize=16)

    # Crear las líneas representativas para la leyenda
    legend_elements = [
        Line2D([0], [0], color=negro, linestyle='-', linewidth=1.5, label='Despierto (Nivel 0)'),
        Line2D([0], [0], color=gris_claro, linestyle='--', linewidth=1.5, label='Sueño Ligero (Nivel 1)'),
        Line2D([0], [0], color=gris_azulado, linestyle=':', linewidth=1.5, label='Sueño Profundo (Nivel 2)')
    ]

    # Agregar leyenda con ajustes de tamaño
    plt.legend(
        handles=legend_elements,
        loc='upper right',
        title='Niveles de Sueño',
        prop={'size': 13},           # Tamaño del texto de la leyenda
        title_fontsize=16             # Tamaño del título de la leyenda
    )

    # Ajuste de tamaño de las marcas de los ejes
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.show()
    i += 1
