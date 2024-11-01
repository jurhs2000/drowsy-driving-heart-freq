import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Definir la matriz de confusión
confusion_matrix = np.array([[801616, 5507, 282562],
                             [47107, 6924, 136206],
                             [358068, 6465, 1929066]])

# Crear la visualización de la matriz de confusión
labels = ['Despierto', 'Sueño Ligero', 'Sueño Profundo']
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=labels)

# Dibujar la matriz de confusión
fig, ax = plt.subplots(figsize=(10, 7))  # Ajustar el tamaño de la figura para mejorar la legibilidad
disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')  # Mostrar valores enteros en las celdas

# Configurar tamaño de la fuente
plt.title('Matriz de Confusión', fontsize=18)  # Tamaño del título
plt.xlabel('Predicción', fontsize=14)  # Etiqueta del eje x
plt.ylabel('Valor Real', fontsize=14)  # Etiqueta del eje y

# Ajustar tamaño de fuente de las etiquetas de los ejes
ax.tick_params(axis='both', which='major', labelsize=12)

# Cambiar el tamaño de la fuente de los números dentro de las celdas
for texts in disp.text_.ravel():  # `disp.text_` contiene los textos en las celdas
    texts.set_fontsize(14)

plt.show()
