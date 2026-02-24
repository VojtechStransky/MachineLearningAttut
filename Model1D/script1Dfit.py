import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


SEED = 42
DATA_PATH = "data_shuffled.txt"
SAVE_DIR = "results_residual50tuned"

BATCH_SIZE = 512

PHASE1_EPOCHS = 150
PHASE2_EPOCHS = 120
PHASE3_EPOCHS = 150

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15

DERIV_WEIGHT = 0.0
PHYS_WEIGHT = 0.0


tf.random.set_seed(SEED)
np.random.seed(SEED)

tf.keras.backend.set_floatx("float32")


def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna()

    X = df.iloc[:, 0:5].values.astype(np.float32)
    Y = df.iloc[:, 5:7].values.astype(np.float32)
    D = df.iloc[:, 7:9].values.astype(np.float32)

    return X, Y, D


def build_dataset(X, Y, D, max_examples=None):
    if max_examples is not None and max_examples < len(X):
        idx = np.random.choice(len(X), max_examples, replace=False)
        X, Y, D = X[idx], Y[idx], D[idx]

    ds = tf.data.Dataset.from_tensor_slices((X, Y, D))

    n = len(X)
    n_train = int(n * TRAIN_SPLIT)
    n_val = int(n * VAL_SPLIT)

    train = ds.take(n_train)
    rest = ds.skip(n_train)
    val = rest.take(int(n * VAL_SPLIT))
    test = rest.skip(int(n * VAL_SPLIT))

    train = (
        train.shuffle(buffer_size=min(n_train, 100000), reshuffle_each_iteration=True)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    val = val.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test = test.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train, val, test


@tf.function
def derivative_loss(y, y_pred, dy, dy_pred):
    mse = tf.reduce_mean(tf.square(y - y_pred))
    dmse = tf.reduce_mean(tf.square(dy - dy_pred))

    return mse + DERIV_WEIGHT * dmse


@tf.function
def physics_loss(y_pred):
    ksi = y_pred[:, 0]

    low = tf.square(tf.nn.relu(-ksi))
    high = tf.square(tf.nn.relu(ksi - 1.0))

    return tf.reduce_mean(low + high)


def residual_block(x, units):
    r = tf.keras.layers.Dense(units, activation="swish")(x)
    r = tf.keras.layers.Dense(units)(r)

    x = tf.keras.layers.Add()([x, r])
    x = tf.keras.layers.Activation("swish")(x)

    return x


def build_model():
    norm = tf.keras.layers.Normalization(name="norm")

    inp = tf.keras.Input((5,))

    x = norm(inp)

    x = tf.keras.layers.Dense(64, activation="swish")(x)

    x = residual_block(x, 64)
    x = residual_block(x, 64)

    shared = tf.keras.layers.Dense(64, activation="swish")(x)

    # KSI HEAD
    k = tf.keras.layers.Dense(32, activation="swish")(shared)
    k = tf.keras.layers.Dense(1)(k)

    # EPS HEAD
    e = tf.keras.layers.Dense(32, activation="swish")(shared)
    e = tf.keras.layers.Dense(1)(e)

    out = tf.keras.layers.Concatenate()([k, e])

    model = tf.keras.Model(inp, out)

    return model, norm


class BoundedModel(tf.keras.Model):
    def __init__(self, base):
        super().__init__()
        self.base = base

    def call(self, x, training=False):
        raw = self.base(x, training=training)

        ksi = 0.95 * tf.sigmoid(raw[:, 0:1]) + 0.025

        eps = raw[:, 1:2]

        return tf.concat([ksi, eps], axis=1)


class PINN(tf.keras.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def call(self, x, training=False):
        return self.model(x, training=training)

    @tf.function
    def train_step(self, data):
        x, y, d = data

        with tf.GradientTape() as t1, tf.GradientTape() as t2:
            t2.watch(x)

            y_pred = self(x, training=True)
            dy_pred = t2.gradient(y_pred, x)

            loss_data = derivative_loss(
                y[:, 0], y_pred[:, 0], d[:, 0], dy_pred[:, 0]
            ) + derivative_loss(y[:, 1], y_pred[:, 1], d[:, 1], dy_pred[:, 1])

            loss_phys = physics_loss(y_pred)

            loss = loss_data + PHYS_WEIGHT * loss_phys

        grads = t1.gradient(loss, self.trainable_variables)

        grads = [tf.clip_by_norm(g, 5.0) for g in grads]

        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return {"loss": loss, "data": loss_data, "phys": loss_phys}

    @tf.function
    def test_step(self, data):
        x, y, d = data

        with tf.GradientTape() as tape:
            tape.watch(x)

            y_pred = self(x, training=False)
            dy_pred = tape.gradient(y_pred, x)

            loss_data = derivative_loss(
                y[:, 0], y_pred[:, 0], d[:, 0], dy_pred[:, 0]
            ) + derivative_loss(y[:, 1], y_pred[:, 1], d[:, 1], dy_pred[:, 1])

            loss_phys = physics_loss(y_pred)

            loss = loss_data + PHYS_WEIGHT * loss_phys

        return {"loss": loss, "data": loss_data, "phys": loss_phys}


def train():
    global DERIV_WEIGHT, PHYS_WEIGHT

    print("Loading data...")

    X, Y, D = load_data(DATA_PATH)

    train_ds, val_ds, test_ds = build_dataset(X, Y, D, 70000)

    print("Building model...")

    base, norm = build_model()
    norm.adapt(X)

    bounded = BoundedModel(base)

    model = PINN(bounded)

    optimizer = tf.keras.optimizers.Adam(1e-3)

    model.compile(optimizer=optimizer)

    history_all = {"loss": [], "val_loss": []}

    # PHASE 1 — DATA ONLY

    DERIV_WEIGHT = 0.0
    PHYS_WEIGHT = 0.0

    print("\nPHASE 1: Data")

    h1 = model.fit(train_ds, validation_data=val_ds, epochs=PHASE1_EPOCHS)

    history_all["loss"] += h1.history["loss"]
    history_all["val_loss"] += h1.history["val_loss"]

    # PHASE 2 — DERIVATIVES

    DERIV_WEIGHT = 0.5
    PHYS_WEIGHT = 0.0

    optimizer.learning_rate = 5e-4

    print("\nPHASE 2: Derivatives")

    h2 = model.fit(train_ds, validation_data=val_ds, epochs=PHASE2_EPOCHS)

    history_all["loss"] += h2.history["loss"]
    history_all["val_loss"] += h2.history["val_loss"]

    # PHASE 3 — PHYSICS

    DERIV_WEIGHT = 0.1
    PHYS_WEIGHT = 0.01

    optimizer.learning_rate = 2e-4

    print("\nPHASE 3: Physics")

    h3 = model.fit(train_ds, validation_data=val_ds, epochs=PHASE3_EPOCHS)

    history_all["loss"] += h3.history["loss"]
    history_all["val_loss"] += h3.history["val_loss"]

    # PLOTS

    os.makedirs(SAVE_DIR, exist_ok=True)

    plt.figure(figsize=(8, 5))

    plt.plot(history_all["loss"])
    plt.plot(history_all["val_loss"])

    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend(["Train", "Val"])
    plt.title("Training Loss")

    plt.savefig(SAVE_DIR + "/loss.png")
    plt.close()

    preds = model.predict(test_ds)

    y_true = np.concatenate([y for _, y, _ in test_ds])

    plt.figure()

    plt.scatter(y_true[:, 0], preds[:, 0], alpha=0.3)
    plt.plot([0, 1], [0, 1], "r--")

    plt.xlabel("True ksi")
    plt.ylabel("Pred ksi")

    plt.savefig(SAVE_DIR + "/ksi.png")
    plt.close()

    plt.figure()

    plt.scatter(y_true[:, 1], preds[:, 1], alpha=0.3)

    mn = min(y_true[:, 1].min(), preds[:, 1].min())
    mx = max(y_true[:, 1].max(), preds[:, 1].max())

    plt.plot([mn, mx], [mn, mx], "r--")

    plt.xlabel("True eps")
    plt.ylabel("Pred eps")

    plt.savefig(SAVE_DIR + "/eps.png")
    plt.close()

    print("\nFinal test loss:")
    print(model.evaluate(test_ds))


if __name__ == "__main__":
    train()
