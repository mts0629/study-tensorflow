"""
Tensorflow 2 quickstart
"""
# Setup the TensorFlow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist

(x_train, y_train) = (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build a machine learning model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(28 * 28,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)])

# Print logits
predictions = model(x_train[:1]).numpy()
print(predictions)

# Print probabilities
print(tf.nn.softmax(predictions).numpy())

# Define a loss function
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
print(loss_fn(y_train[:1], predictions).numpy())

model.compile(optimizer='adam',
    loss=loss_fn,
    metrics=['accuracy'])

print(model.summary())

# Train and evaluate the model
model.fit(x_train, y_train, epochs=5)

print(model.evaluate(x_test, y_test, verbose=2))

# Attach the softmax to the trained model to return probability
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()])

print(probability_model(x_test[:5]))
