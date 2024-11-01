import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl

# Cargar el dataframe
df = pd.read_parquet('data/LSTM.parquet')

# Colores para cada clase
celeste_claro = (173/255, 216/255, 230/255)       # Sueño Profundo
naranja_brillante = (255/255, 127/255, 14/255)  # Sueño Ligero
gris_oscuro = (177/255, 177/255, 177/255)         # Despierto

# Diccionario de colores para cada SleepStage (usando claves como cadenas)
palette = {
    '0': gris_oscuro,        # Despierto
    '1': naranja_brillante,  # Sueño Ligero
    '2': celeste_claro         # Sueño Profundo
}

# Convertir la columna SleepStage a tipo string para que coincida con las claves del diccionario de colores
df['SleepStage'] = df['SleepStage'].astype(str)

# Gráfico de caja para mostrar la distribución de BPM en cada SleepStage
# Crear una paleta de color personalizada que va de azul claro (negativos) a blanco (cero) a azul oscuro (positivos)
# Personalización del mapa de color
cmap = mpl.colors.LinearSegmentedColormap.from_list("custom_cmap", ["#f7c6a5", "white", "#2c5da4"])

# Heatmap de correlación de todas las variables
plt.figure(figsize=(14, 10))  # Aumenta el tamaño de la figura
correlation_matrix = df.corr()  # Calcular la matriz de correlación
sns.heatmap(
    correlation_matrix, 
    annot=True, 
    cmap=cmap,  # Usar la paleta personalizada
    center=0, 
    fmt=".2f", 
    vmin=-1, 
    vmax=1,
    annot_kws={"size": 12},  # Tamaño del texto dentro de las celdas
    cbar_kws={'shrink': 0.8}  # Ajusta el tamaño de la barra de color si es necesario
)

# Ajustar el tamaño de fuente de los labels de los ejes
plt.xticks(rotation=45, ha='right', fontsize=12)  # Rota y aumenta el tamaño de los labels en el eje x
plt.yticks(rotation=0, fontsize=12)  # Aumenta el tamaño de los labels en el eje y

# Título de la gráfica
plt.title('Heatmap de Correlación entre Todas las Variables', fontsize=16)  # Aumenta el tamaño de la fuente del título

# Ajuste de los márgenes
plt.tight_layout()  # Ajusta automáticamente los márgenes para que todo el texto quede visible
plt.show()
