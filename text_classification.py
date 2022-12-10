"""
Basic text classification
"""
import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses

# Sentiment analysis

# Download and explore the IMDB dataset (large moivie review dataset)
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
dataset = tf.keras.utils.get_file("aclImdb_v1", url,
                                  untar=True, cache_dir='.',
                                  cache_subdir='')

dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
os.listdir(dataset_dir)

train_dir = os.path.join(dataset_dir, 'train')
os.listdir(train_dir)

# Take a look at one of movie review (positive)
sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
with open(sample_file) as f:
    print(f.read())

# Load the dataset
# Remove additional folders not corresponding to positive/negative class
remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)

# Create a validation set using an 80:20 split of the training data
batch_size = 32
seed       = 42
raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed)

# Print the first 3 samples in the training dataset
for text_batch, label_batch in raw_train_ds.take(1):
    for i in range(3):
        print("Review", text_batch.numpy()[i])
        print("Label", label_batch.numpy()[i])

print("Label 0 corresponds to", raw_train_ds.class_names[0])
print("Label 1 corresponds to", raw_train_ds.class_names[1])

# Create a validation dataset
raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed)

# Create a test dataset
raw_test_ds = tf.keras.utils.text_dataset_from_directory(
        'aclImdb/test',
        batch_size=batch_size)

# Prepare the dataset for training
def custom_standarization(input_data):
    # Convert test to lowercase
    lowercase = tf.strings.lower(input_data)
    # Remove HTML tags and punctuations
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation),
                                    '')

# Create a TextVectorization layer to standardize, tokenize
# and vectorize the data
max_features    = 10000
sequence_length = 250
vectorize_layer = layers.TextVectorization(
    standardize=custom_standarization,
    max_tokens=max_features,
    # Create unique integer indices for each token
    output_mode='int',
    output_sequence_length=sequence_length)

# Make a text-only dataset (without labels), then call adapt
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

# Retrieve a batch (of 32 reviews and labels) from the dataset
text_batch, label_batch = next(iter(raw_train_ds))
first_review, first_label = text_batch[0], label_batch[0]
print("Review", first_review)
print("Label", raw_train_ds.class_names[first_label])
print("Vectorize review", vectorize_text(first_review, first_label))

# Lookup the token (string) that each integer is corresponding
print("1287 ---> ", vectorize_layer.get_vocabulary()[1287])
print(" 313 ---> ", vectorize_layer.get_vocabulary()[313])
print('Vocabulary size: {}'.format(vectorize_layer.get_vocabulary()))

# Apply preprocessing for each dataset
train_ds = raw_train_ds.map(vectorize_text)
val_ds   = raw_val_ds.map(vectorize_text)
test_ds  = raw_test_ds.map(vectorize_text)

# Configure the dataset for performance
# Create a performant on-disk cache which is more efficient to read than
# many small files
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds  = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Create the model
embedding_dim = 16
model = tf.keras.Sequential([
    # Embedding layer: encode one-hot vector to dense vector with low dimension
    layers.Embedding(max_features+1, embedding_dim),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(1)])

print(model.summary())

# Loss functions and optimizer
# Use the BinaryCrossentropy for a binary classification
model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

# Train the model
epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs)

# Evaluate the model
loss, accuracy = model.evaluate(test_ds)
print("Loss: ", loss)
print("Accuracy: ", accuracy)

# Create a plot of accuracy and loss over time
history_dict = history.history
print(history_dict.keys())

# Plot the training and valication loss/accuracy for comparison
acc      = history_dict['binary_accuracy']
val_acc  = history_dict['val_binary_accuracy']
loss     = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.plot(epochs, acc, 'ro', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()

# Export the model
# Include TextVectorization layer to make the model capable of
# processing raw strings
export_model = tf.keras.Sequential([
    vectorize_layer,
    model,
    layers.Activation('sigmoid')])

export_model.compile(
    loss=losses.BinaryCrossentropy(from_logits=False),
    optimizer="adam",
    metrics=['accuracy'])

# Test it with 'raw_test_ds', which yields raw strings
loss, accuracy = export_model.evaluate(raw_test_ds)
print("accuracy = ", accuracy)

# Inference on new data
examples = [
    "The movie was great!",
    "The movie was okey.",
    "The movie was terrible..."]

export_model.predict(examples)
