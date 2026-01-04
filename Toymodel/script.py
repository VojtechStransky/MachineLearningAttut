import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# autodiferencni metody, definice loss function, zohlednění korelace derivací, metrika derivace, derivace dat ve vstup, parametr vahy derivace


class DerivativeMSETrainer(tf.keras.Model):
    def __init__(self, model, loss_func, lambda_derivative=1.0):
        super().__init__()
        self.model = model
        self.lambda_derivative = lambda_derivative
        self.loss_func = loss_func

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)

    def train_step(self, data):
        x, y_true, dy_true = data

        x = tf.cast(x, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        dy_true = tf.cast(dy_true, tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)

            y_pred = self.model(x, training=True)

            mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))

            dy_pred = tape.gradient(y_pred, x)

            total_loss = self.loss_func(y_true, y_pred, dy_true, dy_pred)

        # Backpropagation
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return {"loss": total_loss, "mse": mse_loss}


Weight = 1


def f(x):
    return -5.3 * np.cos(0.2 * x) + 1.8 * np.sin(-0.9 * x) + 0.8 * np.cos(5.4 - x)


def df(x):
    return -1.62 * np.cos(0.9 * x) + 0.8 * np.sin(5.4 - x) + 1.06 * np.sin(0.2 * x)


def custom_loss(y_true, y_pred, dy_true, dy_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred)) + Weight * tf.reduce_mean(
        tf.square(dy_true - dy_pred)
    )


def trainAndEvaluateNetwork(neuronsNumber, activation, epochsNumber):
    folder = (
        "derivative_" + activation + "_" + str(neuronsNumber) + "_" + str(epochsNumber)
    )

    train_examples = np.linspace(-80, 80, 1000).reshape(
        -1, 1
    )  # overfit => more samples
    train_labels = f(train_examples)
    train_derivative_labels = df(train_examples)
    test_examples = np.linspace(-75, 75, 200).reshape(-1, 1)
    test_labels = f(test_examples)

    validation_examples = np.linspace(-78, 78, 200).reshape(-1, 1)
    validation_examples_over = np.linspace(-110, 110, 200).reshape(-1, 1)
    validation_labels = f(validation_examples)
    validation_labels_over = f(validation_examples_over)

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_examples, train_labels, train_derivative_labels)
    )
    test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))
    validation_dataset = tf.data.Dataset.from_tensor_slices(
        (validation_examples, validation_labels)
    )

    BATCH_SIZE = 20
    SHUFFLE_BUFFER_SIZE = 1000

    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)
    validation_dataset = validation_dataset.batch(BATCH_SIZE)

    model_base = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(1,)),
            tf.keras.layers.Normalization(axis=None),
            tf.keras.layers.Dense(neuronsNumber, activation=activation),
            tf.keras.layers.Dense(neuronsNumber, activation=activation),
            tf.keras.layers.Dense(neuronsNumber, activation=activation),
            tf.keras.layers.Dense(1),
        ]
    )

    model = DerivativeMSETrainer(
        model=model_base, loss_func=custom_loss, lambda_derivative=0.1
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005, epsilon=1e-7),  # step
        loss=tf.keras.losses.MeanSquaredError(),  # dummy
    )

    callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=4000)

    history = model.fit(
        train_dataset,
        epochs=epochsNumber,
        validation_data=test_dataset,
        callbacks=[callback],
    )

    len(history.history["loss"])

    if not os.path.exists(folder):
        os.makedirs(folder)

    # Loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.yscale("log")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(folder + "/metrics.png")

    # Plot true function vs. network prediction
    y_pred = model.predict(validation_examples)

    plt.figure(figsize=(8, 5))
    plt.plot(train_examples, train_labels, label="True Function")
    plt.plot(validation_examples, y_pred, label="Neural Network Prediction")
    plt.title("Function Approximation")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.savefig(folder + "/functions.png")

    plt.figure(figsize=(8, 5))
    plt.scatter(validation_labels, y_pred)
    plt.title("Neural network prediction")
    plt.xlabel("Real")
    plt.ylabel("Prediction")
    plt.legend()
    plt.savefig(folder + "/prediction.png")

    y_pred = model.predict(validation_examples_over)

    plt.figure(figsize=(8, 5))
    plt.plot(validation_examples_over, validation_labels_over, label="True Function")
    plt.plot(validation_examples_over, y_pred, label="Neural Network Prediction")
    plt.title("Overfit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.savefig(folder + "/overfit.png")

    loss = model.evaluate(validation_dataset)

    with open("logderivative.txt", "a") as myfile:
        myfile.write(folder + " " + str(loss) + "\n")
