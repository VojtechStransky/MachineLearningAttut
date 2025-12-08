import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def f(x):
    return -5.3 * np.cos(0.2 * x) + 1.8 * np.sin(-0.9 * x) + 0.8 * np.cos(5.4 - x)


train_examples = np.linspace(-80, 80, 1000).reshape(-1, 1)
train_labels = f(train_examples)
test_examples = np.linspace(-75, 75, 200).reshape(-1, 1)
test_labels = f(test_examples)

train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

plt.plot(train_examples, train_labels)
plt.show()

BATCH_SIZE = 500
SHUFFLE_BUFFER_SIZE = 1000

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(1,)),
        tf.keras.layers.Dense(500, activation="sigmoid"),
        tf.keras.layers.Dense(500, activation="sigmoid"),
        tf.keras.layers.Dense(500, activation="sigmoid"),
        tf.keras.layers.Dense(1),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=["mae"],
)

history = model.fit(train_dataset, epochs=2000, validation_data=test_dataset)

plt.figure(figsize=(12, 4))

# Loss curve
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val loss")
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Metric curve (MAE)
plt.subplot(1, 2, 2)
plt.plot(history.history["mae"], label="train MAE")
plt.plot(history.history["val_mae"], label="val MAE")
plt.title("MAE Curve")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.legend()

plt.tight_layout()
plt.show()

# Plot true function vs. network prediction
y_pred = model.predict(train_examples)

plt.figure(figsize=(8, 5))
plt.plot(train_examples, train_labels, label="True Function")
plt.plot(train_examples, y_pred, label="Neural Network Prediction")
plt.title("Function Approximation")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
