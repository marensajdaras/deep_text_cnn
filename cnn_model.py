import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import time

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Reduce dataset to 1,000 samples (Super-fast training)
x_train, y_train = x_train[:1000], y_train[:1000]
x_test, y_test = x_test[:200], y_test[:200]

# Normalize and reshape
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Use mixed precision training for speed
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Define a minimal CNN model
model = keras.Sequential([
    layers.Conv2D(8, (3,3), activation='relu', input_shape=(28,28,1)),  
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(16, activation='relu'),  
    layers.Dense(10, activation='softmax', dtype='float32')  
])

# Compile model with Adam optimizer
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train with smaller batch size & 4 epochs
start_time = time.time()
model.fit(x_train, y_train, epochs=4, batch_size=128, validation_data=(x_test, y_test), verbose=1)
end_time = time.time()

# Evaluate on test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Training completed in {end_time - start_time:.2f} seconds 🚀")
