import matplotlib.pyplot as plt
import pandas as pd

# Datos de la tabla
data = {
    "Característica": ["PCA_3", "Time", "PCA_2", "PCA_1", "PCA_8", "Age_Binned_<25", "Age_Binned_>60", 
                       "Age_Binned_40_60", "height", "stress", "sleep_quality", "fatigue", "Age_Binned_25_40", 
                       "PCA_9", "sleep_duration", "PCA_10", "PCA_6", "PCA_7", "PCA_5", "PCA_4", "BPM_count"],
    "Importancia": [0.14068, 0.106, 0.08404, 0.07484, 0.0692, 0.04624, 0.0258, 0.02152, 0.02128, 
                    0.0166, 0.00924, 0.00876, 0.00768, 0.006, 0.0052, 0.00472, 0.00432, 0.00228, 0.00136, 
                    0.00116, 0.00104],
    "Desv. Est.": [0.00227, 0.00628, 0.004832, 0.006563, 0.003873, 0.00348, 0.002561, 0.001404, 0.003534,
                   0.002126, 0.002488, 0.003235, 0.000642, 0.001517, 0.00153, 0.001635, 0.00246, 0.001254,
                   0.001228, 0.001004, 0.001729],
    "p-value": [8.1293e-09, 1.4716e-06, 1.3056e-06, 7.0229e-06, 1.1725e-06, 3.8193e-06, 1.1503e-05, 
                2.1635e-06, 8.8054e-05, 3.1593e-05, 5.7378e-04, 1.8783e-03, 5.8009e-06, 4.5073e-04, 
                8.0367e-04, 1.4813e-03, 8.5758e-03, 7.6350e-03, 3.4237e-02, 3.055e-02, 1.2486e-01]
}

# Crear DataFrame
df = pd.DataFrame(data)

azul_marino = (31/255, 119/255, 180/255)
naranja_brillante = (255/255, 127/255, 14/255)
gris_oscuro = (77/255, 77/255, 77/255)

# Graficar
fig, ax1 = plt.subplots(figsize=(12, 8))

# Crear la barra de Importancia y Desviación Estándar
ax1.bar(df["Característica"], df["Importancia"], label="Importancia", color=azul_marino)
ax1.bar(df["Característica"], df["Desv. Est."], label="Desv. Est.", color=naranja_brillante, alpha=0.7)

# Eje secundario para p-value
ax2 = ax1.twinx()
ax2.plot(df["Característica"], df["p-value"], color=gris_oscuro, label="p-value", marker='o', linestyle='dashed')

# Configurar etiquetas y leyenda
ax1.set_xlabel("Característica")
ax1.set_ylabel("Importancia y Desv. Est.")
ax2.set_ylabel("p-value")
ax1.set_xticklabels(df["Característica"], rotation=90)
fig.tight_layout()
plt.subplots_adjust(top=0.88)  # Aumenta el margen superior
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

# Título con margen ajustado
plt.title("Importancia, Desviación Estándar y p-value de Características", pad=20)

plt.show()
