import matplotlib.pyplot as plt
import pandas as pd

# Datos de la tabla en formato de diccionario
data = {
    "Variable": [
        "BPM", "BPM_Mean_Acc", "BPM_Var_Acc", "BPM_Std_Acc", "BPM_Diff", "BPM_Acceleration", 
        "BPM_Mean_Diff", "Time_SleepStage", "SleepStage_Changes", "BPM_Trend", "BPM_Lag_1", 
        "BPM_Diff_Lag_1", "BPM_Acceleration_Lag_1", "BPM_Mean_Diff_Lag_1", "BPM_Mean_Acc_Lag_1", 
        "BPM_Trend_Lag_1", "Time_SleepStage_Lag_1", "BPM_Lag_2", "BPM_Diff_Lag_2", 
        "BPM_Acceleration_Lag_2", "BPM_Mean_Diff_Lag_2", "BPM_Mean_Acc_Lag_2", 
        "BPM_Trend_Lag_2", "Time_SleepStage_Lag_2", "BPM_Lag_3", "BPM_Diff_Lag_3", 
        "BPM_Acceleration_Lag_3", "BPM_Mean_Diff_Lag_3", "BPM_Mean_Acc_Lag_3", "BPM_Trend_Lag_3", 
        "Time_SleepStage_Lag_3", "BPM_Lag_4", "BPM_Diff_Lag_4", "BPM_Acceleration_Lag_4", 
        "BPM_Mean_Diff_Lag_4", "BPM_Mean_Acc_Lag_4", "BPM_Trend_Lag_4", "Time_SleepStage_Lag_4", 
        "BPM_Lag_5", "BPM_Diff_Lag_5", "BPM_Acceleration_Lag_5", "BPM_Mean_Diff_Lag_5", 
        "BPM_Mean_Acc_Lag_5", "BPM_Trend_Lag_5", "Time_SleepStage_Lag_5"
    ],
    "PCA_4": [
        -0.001394, 0.005062, -0.001528, -0.001664, -0.074448, -0.176521, -0.005965, 0.003252, -0.002250, 
        0.094358, 0.032325, 0.218103, 0.309057, 0.036569, 0.005116, -0.039277, 0.003289, -0.066483, 
        -0.294095, -0.374766, -0.087940, 0.004810, -0.079676, 0.003274, 0.066744, 0.326980, 0.393504, 
        0.079861, 0.005318, 0.042979, 0.003276, -0.081384, -0.325136, -0.301422, -0.106732, 0.004786, 
        -0.079050, 0.003282, 0.065913, 0.174379, 0.128013, 0.078738, 0.005403, 0.092966, 0.003251
    ],
    "PCA_5": [
        0.013141, -0.013884, 0.002727, 0.002726, -0.213412, -0.312580, 0.028118, -0.007559, 0.009818, 
        0.088550, 0.109827, 0.304638, 0.265767, 0.150098, -0.013765, 0.339343, -0.007559, -0.028182, 
        -0.135780, -0.136314, -0.023551, -0.014497, 0.041087, -0.007504, 0.033330, 0.090105, -0.103787, 
        0.053965, -0.014318, 0.023705, -0.007503, -0.007486, 0.262150, 0.358687, 0.002675, -0.014612, 
        -0.140123, -0.007432, -0.126242, -0.332269, -0.230612, -0.147096, -0.014828, -0.222887, -0.007379
    ]
}

# Crear el DataFrame
df = pd.DataFrame(data)

# Tomar valores absolutos y ordenar por PCA_4 y PCA_5
df["PCA_4_abs"] = df["PCA_4"].abs()
df["PCA_5_abs"] = df["PCA_5"].abs()
df = df.sort_values(by=["PCA_4_abs", "PCA_5_abs"], ascending=False)

# Graficar
fig, ax = plt.subplots(figsize=(16, 10))  # Ajuste del tamaño
ax.barh(df["Variable"], df["PCA_4_abs"], color=(31/255, 119/255, 180/255), label="PCA_4", height=0.35)
ax.barh(df["Variable"], df["PCA_5_abs"], color=(255/255, 127/255, 14/255), label="PCA_5", alpha=0.7, height=0.35)

# Configuración de la gráfica
ax.set_xlabel("Carga Absoluta", fontsize=14)
ax.set_ylabel("Variable", fontsize=14)
ax.set_title("Carga Absoluta de Variables en Componentes PCA_4 y PCA_5 (Ordenadas de Mayor a Menor)", fontsize=16)
ax.legend(fontsize=12)

# Aumentar el tamaño de las etiquetas del eje y
ax.tick_params(axis='y', labelsize=12)

# Ajustar márgenes para reducir el espacio en blanco
plt.subplots_adjust(top=0.95, bottom=0.05)  # Ajusta el espacio superior e inferior

plt.gca().invert_yaxis()
plt.show()
