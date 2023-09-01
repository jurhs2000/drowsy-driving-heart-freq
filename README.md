# drowsy-driving-heart-freq
Model to detect Drowsy Driving by a real-time heart frequency sensor.

### ¿Qué supuesto pretende probar con este prototipo?

Que por medio de los datos de frecuencia cardiaca se puede detectar fatiga durante largas horas de realizar una actividad.

### ¿Qué aprendió en este primer prototipo?

Que sí existe una diferencia cuando se está despierto y cuando se está dormido, pero no es tan marcada como se esperaba. De igual manera estos son datos necesitan ser procesados para quitar los outliers, normalizarlos y estandarizarlos para conocer mejor las diferencias entre las distintas fases de actividad. Tampoco hay que quedarse con estos datos ya que sabemos que luego nosotros obtendremos nuestros propios datos a partir del sensor de frecuencia cardiaca.

### ¿Cuáles serán los pasos para un siguiente prototipo?

Obtener estos datos de un sensor de frecuencia cardiaca.

### ¿Se necesita hacer algún cambio en el protocolo, derivado del aprendizaje en este prototipo?

En realidad no, ya se esperaba que sí existiera una diferencia entre los datos de frecuencia cardiaca cuando se está despierto y cuando se está dormido. Tal vez en fases más adelante cuando ya se implemente el modelo será necesario especificar en el protocolo qué tipo de modelo de aprendizaje se utilizará.

command to run in raspberry pi:
```python3 miband4_console.py -m F3:C5:B7:EA:FA:FA```