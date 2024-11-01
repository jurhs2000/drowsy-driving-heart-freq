import matplotlib.pyplot as plt
import numpy as np

# Datos de la tabla
classes = ['Clase 0', 'Clase 1', 'Clase 2']
models = ['LSTM', 'AutoGluon Tabular', 'AutoGluon Forecast']

# Métricas para cada clase y modelo
precision = {
    'LSTM': [0.66, 0.37, 0.82],
    'AutoGluon Tabular': [0.86, 0.59, 0.82],
    'AutoGluon Forecast': [0.95, 0.50, 0.96]
}
recall = {
    'LSTM': [0.74, 0.04, 0.84],
    'AutoGluon Tabular': [0.68, 0.33, 0.94],
    'AutoGluon Forecast': [0.89, 0.70, 0.96]
}
f1_score = {
    'LSTM': [0.70, 0.07, 0.83],
    'AutoGluon Tabular': [0.76, 0.43, 0.88],
    'AutoGluon Forecast': [0.92, 0.58, 0.96]
}

# Colores para cada clase
celeste_claro = (173/255, 216/255, 230/255)       # Sueño Profundo
naranja_brillante = (255/255, 127/255, 14/255)  # Sueño Ligero
gris_oscuro = (177/255, 177/255, 177/255)         # Despierto

# Colores para cada clase
colors = [celeste_claro, naranja_brillante, gris_oscuro]

# Configuración de la gráfica
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)  # Ajuste el tamaño
fig.suptitle("Comparación de Métricas de Clasificación por Modelo y Clase", fontsize=20)

# Función para plotear cada métrica
def plot_metric(ax, metric_data, title, ylabel=False):
    x = np.arange(len(models))
    width = 0.2  # Reduce el ancho de las barras

    # Plotear barras
    for i, (cls, color) in enumerate(zip(classes, colors)):
        ax.bar(x + i * width - width, [metric_data[model][i] for model in models], width, label=cls, color=color)
    
    ax.set_title(title, fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right', fontsize=12)  # Rota y ajusta etiquetas y aumenta el tamaño de la fuente
    if ylabel:
        ax.set_ylabel('Métrica', fontsize=14)
    if title == "Precisión":  # Colocar la leyenda solo una vez
        ax.legend(loc='upper left', fontsize=12)

# Graficar cada métrica
plot_metric(axs[0], precision, "Precisión", ylabel=True)
plot_metric(axs[1], recall, "Recall")
plot_metric(axs[2], f1_score, "F1-Score")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
