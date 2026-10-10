
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models
from sklearn.metrics import classification_report

# Load the MNIST handwritten digit dataset
(X_train, y_train), (X_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

# Normalize pixel values from 0-255 to 0-1
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Build a Convolutional Neural Network (CNN)
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# Compile and train the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train[..., np.newaxis],
    y_train,
    epochs=3,
    batch_size=128,
    validation_split=0.1
)

# Evaluate on test data
loss, accuracy = model.evaluate(
    X_test[..., np.newaxis], y_test, verbose=0
)

print(f"Test accuracy: {accuracy:.4f}")

# Generate predictions and save a report
probabilities = model.predict(
    X_test[..., np.newaxis], verbose=0
)
predictions = np.argmax(probabilities, axis=1)

with open("classification_report.txt", "w") as file:
    file.write(f"Test accuracy: {accuracy:.4f}\n\n")
    file.write(classification_report(y_test, predictions))

# Save the trained model
model.save("handwritten_digits_model.keras")

# Save an example prediction image
plt.figure(figsize=(5, 5))
plt.imshow(X_test[0], cmap="gray")
plt.title(f"Actual: {y_test[0]}, Predicted: {predictions[0]}")
plt.axis("off")
plt.savefig("sample_prediction.png")
plt.close()

print("Model and evaluation report saved.")
