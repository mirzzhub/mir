# mir

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def build_model(activation, neurons, layers):
    """Builds and compiles a Keras sequential model."""
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    for _ in range(layers):
        model.add(Dense(neurons, activation=activation))
    model.add(Dropout(0.3))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

configurations = [('sigmoid', 64, 1), ('sigmoid', 128, 2),
                  ('relu', 64, 1), ('relu', 128, 2)]

results = {}
for activation, neurons, layers in configurations:
    print(f'Training model with {activation} activation, {neurons} neurons, {layers} layers...')
    model = build_model(activation, neurons, layers)
    model.fit(x_train, y_train, epochs=10, verbose=0)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    results[f'{activation} {neurons} {layers} layers'] = accuracy * 100
    print(f'Accuracy: {accuracy * 100:.2f}%')

plt.bar(results.keys(), results.values(), color=['blue', 'green', 'red', 'purple'])
plt.title('Accuracy of Models with Different Activations, Neurons & Layers')
plt.xlabel('Model Configurations')
plt.ylabel('Accuracy (%)')
plt.ylim(95.0, 100.0)  # Set y-axis limits for better visualization
plt.xticks(rotation=20)  # Rotate x-axis labels for readability
plt.show()
