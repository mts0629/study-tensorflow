import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# load MNIST dataset
mnist = tf.keras.datasets.mnist

# load training/test data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# preprocess, reshape and scaling to [0, 1]
train_images = train_images.reshape(-1, 28 * 28) / 255.0
test_images  = test_images.reshape(-1, 28 * 28) / 255.0

# define a sequential model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(28 * 28,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

# configure the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.metrics.SparseCategoricalAccuracy()]
)
print(model.summary())

# train the model
model.fit(train_images, train_labels, epochs=5)

# evaluate the model with test data
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)
print("\nTest Loss %0.2f, Test Accuracy %0.2f" % (test_loss, test_accuracy))

# append a softmax to output probability
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

# predict
predictions = probability_model.predict(test_images)

# create a confusion matrix
confusion_matrix = np.zeros((10, 10), dtype=np.int32)
for prediction, true_label in zip(predictions, test_labels):
    predicted_label = np.argmax(prediction)
    confusion_matrix[true_label, predicted_label] += 1

# plot a heatmap
fig, ax = plt.subplots()
axis = [n + 0.5 for n in range(10)]
ax.set_xticks(axis)
ax.set_yticks(axis)
axis_labels = [str(n) for n in range(10)]
ax.set_xticklabels(axis_labels)
ax.set_yticklabels(axis_labels)

ax.set_xlabel("predicted label")
ax.set_ylabel("true label")

ax.xaxis.tick_top()
ax.invert_yaxis()

heatmap = ax.imshow(confusion_matrix)

for i in range(10):
  for j in range(10):
    text = ax.text(j, i, confusion_matrix[i, j], ha="center", va="center", color="w")
fig.tight_layout()

plt.show()

# save the entire model as a SavedModel format
model.save("./mnist")

