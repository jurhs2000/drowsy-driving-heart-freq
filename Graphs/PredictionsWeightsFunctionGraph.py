import numpy as np
import matplotlib.pyplot as plt

# Define the function for weighted predictions
def get_weights(predictions_length, decay_factor=0.75):
    weights = []
    for i in range(predictions_length):
        weight = decay_factor ** (predictions_length - i - 1)
        weights.append(weight)
    return weights

# Generate example weights for a set of predictions
predictions_length = 24  # Assume 10 predictions for the example
decay_factor = 0.75
weights = get_weights(predictions_length, decay_factor)

# Plot the weights
plt.figure(figsize=(8, 6))
plt.plot(range(predictions_length), weights, marker='o')
plt.title(f"Pesos aplicados a cada predicción con Factor de decaimiento = {decay_factor}")
plt.xlabel("Número de predicción")
plt.ylabel("Peso")
plt.grid(True)
plt.show()
