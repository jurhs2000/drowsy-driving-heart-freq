import tensorflow as tf
import sys

# Ver la versión de Python
print(f"Versión de Python: {sys.version}")

# Ver la versión de TensorFlow
print(f"Versión de TensorFlow: {tf.__version__}")

# Listar los dispositivos físicos disponibles
devices = tf.config.list_physical_devices()
print("Dispositivos disponibles:")
for device in devices:
    print(device)

# Verificar específicamente si hay GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs detectadas: {gpus}")
else:
    print("No se detectaron GPUs.")

# Verificar el uso de CPUs y threads en TensorFlow
cpus = tf.config.list_physical_devices('CPU')
print(f"Número de CPUs detectadas: {len(cpus)}")

inter_op_threads = tf.config.threading.get_inter_op_parallelism_threads()
intra_op_threads = tf.config.threading.get_intra_op_parallelism_threads()
print(f"Threads inter_op utilizados: {inter_op_threads}")
print(f"Threads intra_op utilizados: {intra_op_threads}")

# Cargar el modelo entrenado
model_path = 'out/lstm/lstm_model_with_embeddings.keras'  # Reemplaza con la ruta a tu modelo
model = tf.keras.models.load_model(model_path)

# Resumen del modelo
print("\nResumen del modelo:")
model.summary()

# Obtener la configuración del modelo
print("\nConfiguración del modelo:")
print(model.get_config())

# Ver los pesos del modelo (opcional)
print("\nPesos del modelo (primeras 5 capas):")
for layer in model.layers[:5]:
    print(f"\nCapa: {layer.name}")
    weights = layer.get_weights()
    print(f"Pesos: {weights}")
