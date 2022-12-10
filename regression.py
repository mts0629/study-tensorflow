"""
Basic regression: Predict fuel efficiency
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Use seaborn for pairplot
import seaborn as sns

# Make Numpy printouts easier to read
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# The Auto MPG dataset
# Get the data
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
               'Aceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)

dataset = raw_dataset.copy()
print(dataset.tail())

# Clean the data
print(dataset.isna().sum())

# Drop N/A rows
dataset = dataset.dropna()

# Adopt one-hot encode the values in the column
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
dataset.tail()

# Split the data into training and test sets
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset  = dataset.drop(train_dataset.index)

# Inspect the data
sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')
train_dataset.describe().transpose()

# Split features from labels
train_features = train_dataset.copy()
test_features  = test_dataset.copy()

train_labels = train_features.pop('MPG')
test_labels  = test_features.pop('MPG')

train_dataset.describe().transpose()[['mean', 'std']]

# The Normalization layer
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))
# Calculate the mean and variance, and store them in the layer
print(normalizer.mean.numpy())

# When the layer is called, it return the input data, with each feature
# independently normalized
first = np.array(train_features[:1])
with np.printoptions(precision=2, suppress=True):
    print('First example:', first)
    print()
    print('Normalized:', normalizer(first).numpy())

# Linear regression
# Predict 'MPG' from 'Horsepower'
horsepower = np.array(train_features['Horsepower'])
horsepower_normalizer = layers.Normalization(input_shape=[1,], axis=None)
horsepower_normalizer.adapt(horsepower)

# Build the Keras Sequential model
horsepower_model = tf.keras.Sequential([
    horsepower_normalizer,
    layers.Dense(units=1)])

print(horsepower_model.summary())

# Print the shape of output
print(horsepower_model.predict(horsepower[:10]))

# Mean squared error (MSE) or mean absolute error (MAE) are
# common loss functions used for regression problems
horsepower_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    # MAE is less sensitive to outliers
    loss='mean_absolute_error')

history = horsepower_model.fit(
    train_features['Horsepower'],
    train_labels,
    epochs=100,
    # Suppress logging
    verbose=0,
    # Calculate validation results on 20% of the learning data
    validation_split = 0.2)

# Visualize the model's training progress
# using the stats stored in the history object
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())

def plot_loss(history):
    plt.figure()
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)

    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')

plot_loss(history)

# Collect the results
test_results = {}
test_results['horsepower_model'] = horsepower_model.evaluate(
    test_features['Horsepower'],
    test_labels, verbose=0)

# Plot predictions
def plot_horsepower(x, y):
    plt.figure()
    plt.scatter(train_features['Horsepower'], train_labels, label='Data')
    plt.plot(x, y, color='k', label='Predictions')
    plt.xlabel('Horsepower')
    plt.ylabel('MPG')
    plt.legend()

x = tf.linspace(0.0, 250, 251)
y = horsepower_model.predict(x)

plot_horsepower(x, y)

# Linear regression with multiple inputs
linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)])

# Print the shape of output and kernel weights
print(linear_model.predict(train_features[:10]))
print(linear_model.layers[1].kernel)

linear_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

history = linear_model.fit(
    train_features,
    train_labels,
    epochs=100,
    verbose=0,
    validation_split=0.2)

plot_loss(history)

test_results['linear_model'] = linear_model.evaluate(
    test_features, test_labels, verbose=0)

# Regression by DNN
def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1),])

    model.compile(loss='mean_absolute_error',
                  optimizer=tf.keras.optimizers.Adam(0.001))

    return model

# Regression using a DNN and a single input
dnn_horsepower_model = build_and_compile_model(horsepower_normalizer)
print(dnn_horsepower_model.summary())

history = dnn_horsepower_model.fit(
    train_features['Horsepower'],
    train_labels,
    epochs=100,
    verbose=0,
    validation_split=0.2)

plot_loss(history)

x = tf.linspace(0.0, 250, 251)
y = dnn_horsepower_model.predict(x)

plot_horsepower(x, y)

test_results['dnn_horsepower_model'] = dnn_horsepower_model.evaluate(
    test_features['Horsepower'], test_labels,
    verbose=0)

# Regression by DNN with multiple inputs
dnn_model = build_and_compile_model(normalizer)
print(dnn_model.summary())

history = dnn_model.fit(
    train_features,
    train_labels,
    epochs=100,
    verbose=0,
    validation_split=0.2)

plot_loss(history)

test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)

# Performance
pd.DataFrame(test_results, index=['Mean absolute error [MPG]']).T

# Make predictions
test_predictions = dnn_model.predict(test_features).flatten()

# Plot true values and predictions with a scatter plot
plt.figure()
a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

# Check the error distribution with a histogram
plt.figure()
error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [MPG]')
_ = plt.ylabel('Count')

# Save the model
dnn_model.save('dnn_model')

# Re-evaluate with a reloaded model
reloaded = tf.keras.models.load_model('dnn_model')
test_results['reloaded'] = reloaded.evaluate(test_features, test_labels, verbose=0)
pd.DataFrame(test_results, index=['Mean absolute error [MPG]']).T
