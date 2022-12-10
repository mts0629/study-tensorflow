"""
Basic classification: Classify images of clothing
"""
# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# Import the Fashion MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Explore the data
print("train_images.shape = ", train_images.shape)
print("len(train_labels) = ", len(train_labels))
print("train_labels = ", train_labels)

print("test_images.shape = ", test_images.shape)
print("len(test_labels) = ", len(test_labels))
print("test_labels = ", test_labels)

# Preprocess the data
# Display the first image in the training set
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# Normalize values to a range of 0 to 1
train_images = train_images / 255.0
test_images  = test_images / 255.0

# Display the first 25 images from the training set
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

plt.show()

# Build the model
# Set up the layers
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10)])

# Compile the model
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"])

# Train the model
# Feed the model
model.fit(train_images, train_labels, epochs=10)

# Evaluate accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("\nTest accuracy: ", test_acc)

# Make predictions
# Attach a softmax layer to convert the logits to probabilties
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)

# Take a look at the first prediction
print("predictions[0] = ", predictions[0])
# See a label which has the highest confidence value
print("np.argmax(predictions[0]) = ", np.argmax(predictions[0]))
# See a corresponding label with test_images[0]
print("test_labels[0] = ", test_labels[0])

# Vefify predictions
# Plot the first X test images, their predicted labels, and the true labels
# Color correct predictions in blue and incorrect predictions in red

# Plot the test image, predicted label and the true label
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = "blue"
    else:
        color = "red"

    plt.xlabel("{} {:2.0f}% ({})".format(
        class_names[predicted_label],
        100 * np.max(predictions_array),
        class_names[true_label]),
        color=color)

# Plot labels and predictions with bar plot
def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])

    thisplot = plt.bar(range(10), predictions_array, color="#777777")

    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color("red")
    thisplot[true_label].set_color("blue")

num_rows = 5
num_cols = 3
num_images = num_rows * num_cols

plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)

    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)

plt.tight_layout()
plt.show()

# Use the trained model
# Grab an image from the test dataset
img = test_images[1]
print("img.shape = ", img.shape)

# Add the image to a batch where it's the only member
img = (np.expand_dims(img, 0))
print("expanded img.shape = ", img.shape)

# Predict and plot the image
predictions_single = probability_model.predict(img)
print("prediction = ", predictions_single)

plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()

print("predicted label = ", np.argmax(predictions_single[0]))

pred_label = np.argmax(predictions_single[0])
