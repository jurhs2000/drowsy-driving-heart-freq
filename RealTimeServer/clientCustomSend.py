import asyncio
import websockets
import time
import json
import pandas as pd

df_to_save = pd.DataFrame(columns=['time', 'BPM'])

current_time = time.strftime('%Y-%m-%d %H:%M:%S')

# Función para enviar los datos del BPM al servidor
async def send_bpm_to_server():
    uri = "ws://127.0.0.1:2222"  # Dirección correcta para conexiones locales
    
    try:
        # Conectarse al servidor WebSocket
        async with websockets.connect(uri) as websocket:
            with open('bpm.txt', 'r') as file:
                for line in file:
                    row_time = current_time
                    bpm = line.strip()  # Leer BPM de cada línea del archivo
                    data_to_send = {
                        "dateTime": row_time,
                        "bpm": int(bpm)
                    }
                    await websocket.send(json.dumps(data_to_send))
                    print(f"BPM enviado: {bpm}")
                    df_to_save.loc[len(df_to_save)] = [data_to_send['dateTime'], data_to_send['bpm']]
                    # Agregar 5 segundos a current_time
                    current_time = (pd.to_datetime(current_time) + pd.Timedelta(seconds=5)).strftime('%Y-%m-%d %H:%M:%S')
                    #await asyncio.sleep(5)  # Esperar 5 segundos antes de enviar el siguiente valor
    except Exception as e:
        print(f"Error al conectarse o enviar datos: {e}")

# Ejecutar la función principal utilizando asyncio.run()
asyncio.run(send_bpm_to_server())

# Guardar los datos en un archivo CSV
df_to_save.to_csv('bpm_data.csv', index=False)
print("Datos guardados en 'bpm_data.csv'")
