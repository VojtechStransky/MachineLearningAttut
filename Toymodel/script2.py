import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def f(x):
    return -5.3 * np.cos(0.2 * x) + 1.8 * np.sin(-0.9 * x) + 0.8 * np.cos(5.4 - x)


def df(x, y):
    h = 1e-5
    return (y(x + h) - y(x - h)) / (2 * h)


def gradient_descent(starting_point, learning_rate, iterations):
    x = starting_point
    for i in range(iterations):
        x = x - learning_rate * df(x, f)
    return x


learning_rate = 0.01
iterations = 1000

train_examples = np.linspace(-80, 80, 1000).reshape(-1, 1)
train_labels = gradient_descent(train_examples, learning_rate, iterations)
test_examples = np.linspace(-75, 75, 200).reshape(-1, 1)
test_labels = gradient_descent(test_examples, learning_rate, iterations)

train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

BATCH_SIZE = 500
SHUFFLE_BUFFER_SIZE = 1000

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(1,)),
        tf.keras.layers.Dense(600, activation="relu"),
        tf.keras.layers.Dense(600, activation="relu"),
        tf.keras.layers.Dense(600, activation="relu"),
        tf.keras.layers.Dense(1),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=["mae"],
)

history = model.fit(train_dataset, epochs=50000, validation_data=test_dataset)

model.save("model2.keras")

plt.figure(figsize=(20, 8))

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

plt.savefig("fig1.png")

# Plot true function vs. network prediction
y_pred = model.predict(train_examples)

plt.figure(figsize=(16, 10))
plt.plot(train_examples, train_labels, label="True Function")
plt.plot(train_examples, y_pred, label="Neural Network Prediction")
plt.title("Function Approximation")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.savefig("fig2.png")
