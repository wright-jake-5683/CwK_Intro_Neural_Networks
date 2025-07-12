from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

digit_mnist = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = digit_mnist.load_data()

categories = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
labels = set(train_labels)

#for label in labels:
#  print(f"Training Labels Index {label}: \t {categories[label]} \n")

#train_images[0]
'''
plt.figure(figsize=(10,10))
for i in range(10):
  plt.subplot(6,5,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(train_images[i], cmap="gray")
  plt.xlabel(categories[train_labels[i]])
plt.show()
'''

validation_images, train_images = train_images[:5000] / 255.0, train_images[5000:] / 255.0

validation_labels, train_labels = train_labels[:5000], train_labels[5000:]

test_images_norm = test_images / 255. 

validation_images[0]
train_images[0]


model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
# Sigmoid: probabilities produced by a Sigmoid are independent
# Softmax: Are outputs are interrelated. The sum of all outputs are 1

#model.summary()

model.compile(
    loss = "sparse_categorical_crossentropy",
    optimizer = "sgd",
    metrics = ["accuracy"]
)

history = model.fit(train_images, train_labels, epochs=10, validation_data = (validation_images, validation_labels), batch_size=32)

predictions = model.predict(test_images)

label = np.argmax(predictions[0])
print(f"Model predicts: {categories[label]}")

plt.figure()
plt.imshow(test_images[0])
plt.colorbar()
plt.grid(False)
plt.show()