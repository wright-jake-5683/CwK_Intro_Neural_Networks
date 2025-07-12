import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load Fashion MNIST Dataset
fashion_mnist = tf.keras.datasets.fashion_mnist


(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Create a set of class names that correspond to the labels
class_names = ["T-Shirts/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]
labels = set(train_labels)

'''
for label in labels:
  print(label, "\t\t", class_names[label])

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
'''

# Normalize pixel data range from 0-255 to 0-1

#print(train_images[0])
train_images = train_images/ 255.0
#print(train_images[0])

test_images = test_images / 255.0

'''
plt.figure(figsize=(10,10))
for i in range(25):
  plt.subplot(6,5,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(train_images[i], cmap=plt.cm.binary)
  plt.xlabel(class_names[train_labels[i]])
plt.show()
'''

# Steps to set up Neural Network:
#  1.) Set up Layers
#  2.) Compliling the Model
#  3.) Training the Model


# Layers:
#. 1.) Flatten - Flattens the 28x28 2D numpy array for the image into a single 784 pixel array
#  2.) 2 x Dense (Fully-Connected)
#     -- First, a 128 node layer that processes a weighted sum of the inputs it recieves from the previous layer. The activation function is "relu" and introduces non-linearity into the network enabling it to learn
#     -- Second, a 10 node output layer (digits 0-9). Each neuron in this layer outputs a score indicating the likelihood of the image belonging to a specific class

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])


#model.summary()


model.compile(optimizer='adam',
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
metrics=['accuracy'])


# Feed and train the model
model.fit(train_images, train_labels, epochs=10)

# Attach a softmax layer to convert outputs to interrelated probabilities
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)


#print(predictions[400])

label = np.argmax(predictions[400])
print(f"Model predicts: {class_names[label]}")

plt.figure()
plt.imshow(test_images[400])
plt.colorbar()
plt.grid(False)
plt.show()