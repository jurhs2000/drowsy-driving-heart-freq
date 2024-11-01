# Este programa funciono para leer los archivos de SHHS y unificar los datos de todas las visitas en un solo csv
# para posteriormente ser manipulado en participantsdata.py eligiendo solo los datos que se necesitan.

import os
import pandas as pd

# Ajusta el camino hacia la carpeta de datos
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
archive_path = os.path.join(base_path, 'shhs', 'datasets', 'archive')

# Inicializar una lista para almacenar los DataFrames
dataframes = []

# Recorrer todas las subcarpetas en la carpeta 'archive'
for subdir in os.listdir(archive_path):
    subdir_path = os.path.join(archive_path, subdir)
    if os.path.isdir(subdir_path):
        print(f'Explorando subcarpeta: {subdir}')
        for file in os.listdir(subdir_path):
            file_path = os.path.join(subdir_path, file)
            if file.endswith('.csv'):
                print(f'  Cargando archivo: {file}')
                try:
                    # Cargar el archivo CSV
                    df = pd.read_csv(file_path)
                    # Verificar si la columna 'pptid' existe
                    if 'pptid' in df.columns:
                        dataframes.append(df)
                    else:
                        print(f'  Ignorado (sin columna pptid): {file}')
                except Exception as e:
                    print(f'  Error al cargar {file}: {e}')

# Unir todos los DataFrames en uno solo utilizando 'pptid' como clave
if dataframes:
    merged_data = dataframes[0]
    for df in dataframes[1:]:
        merged_data = pd.merge(merged_data, df, on='pptid', how='outer', suffixes=('_old', ''))

        # Eliminar columnas antiguas
        for col in df.columns:
            if col != 'pptid' and f"{col}_old" in merged_data.columns:
                merged_data.drop(columns=[f"{col}_old"], inplace=True)

    # Guardar el DataFrame unido en un archivo CSV
    output_csv_path = os.path.join(base_path, 'merged_shhs_data.csv')
    merged_data.to_csv(output_csv_path, index=False)
    print(f'Datos fusionados guardados en {output_csv_path}')
else:
    print('No se encontraron archivos CSV con la columna pptid.')

# Mostrar las primeras filas del DataFrame fusionado si no está vacío
if not merged_data.empty:
    print(merged_data.head())
else:
    print("El DataFrame fusionado está vacío.")
