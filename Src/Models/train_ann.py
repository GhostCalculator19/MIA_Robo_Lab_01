# ANN - Artificial Neural Network
import joblib, os, yaml, json, datetime
import pandas as pd
import numpy as np

from typing import Tuple, Dict
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from dvclive import Live
from sklearn.model_selection import train_test_split

from train_utils import load_data
from tensorboard.backend.event_processing import event_accumulator

from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

class WeightsHistoryCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        for layer in self.model.layers:
            if hasattr(layer, "kernel"):
                weights = layer.kernel.numpy().flatten()

                if layer.name not in self.history:
                    self.history[layer.name] = []

                self.history[layer.name].append(weights)
        
def main() -> None:

    # Load configuration
    with open("Config/paramts.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    # Load data
    x_train, y_train = load_data(
        train_path=config["data"]["train_path"],
        target_column=config["data"]["target_column"]
    )

    y_train_log = np.log1p(y_train)
    y_train_log = y_train_log.reshape(-1, 1)
    
    # Load dataset params
    dataset_params = config["models"]["ann"]["dataset_params"]

    train_ds, val_ds = create_tf_dataset(
        #x_train=x_train.values,
        x_train=x_train,
        y_train=y_train_log,
        **dataset_params
        )

    # Load model params
    model_params = config["models"]["ann"]["model_params"]
    
    # Initialize model
    model: tf.keras.Model = create_ann_model(input_dim=x_train.shape[1], **model_params)

    os.makedirs("logs/fit/", exist_ok=True)
    log_dir: str = "logs/fit/" + datetime.datetime.now().strftime("%H-%M-%S_%d-%m-%Y")
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,          # гистограммы весов
        write_graph=True,
        write_images=True,
        update_freq="epoch",
        profile_batch=0,
        embeddings_freq=0
    )
    
    weights_history_cb = WeightsHistoryCallback()

    tf.summary.trace_on(graph=True, profiler=False)

    # Load train params
    train_params = config["models"]["ann"]["train_params"]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        callbacks=[tensorboard_callback, weights_history_cb
                   ],
        epochs=train_params["epochs"],
        verbose=train_params["verbose"]
    )

    # Learning Curves
    os.makedirs(config["reports"]["figures_path"], exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend(["Train", "Validation"])
    plt.savefig("Reports/figures/ann_loss_curve.png")

    # Weight Histograms
    for i, layer in enumerate(model.layers):
        if hasattr(layer, "kernel"):
            weights = layer.kernel.numpy().flatten()
            plt.figure(figsize=(12, 6))
            plt.hist(weights, bins=100)
            plt.title(f"Layer {i} Weight Distribution")
            plt.xlabel("Weight value")
            plt.ylabel("Frequency")
            plt.savefig(config["reports"]["figures_path"] + f"ann_weights_layer_{i}.png")
            plt.close()
    
    tf.keras.utils.plot_model(
        model,
        to_file=config["reports"]["figures_path"] + "ann_model_graph.png",
        show_shapes=True,
        show_layer_names=True,
        dpi=300
    )

    with tf.summary.create_file_writer(log_dir).as_default():
        tf.summary.trace_export(
            name="ANN_graph_trace",
            step=0,
            profiler_outdir=log_dir
        )
    
    writer = tf.summary.create_file_writer(log_dir)

    with writer.as_default():
        # ---- Нормы весов по слоям ----
        for layer in model.layers:
            if hasattr(layer, "kernel"):
                weights = layer.kernel
                weight_norm = tf.norm(weights)
                tf.summary.scalar(f"weights_norm/{layer.name}", weight_norm, step=0)
                tf.summary.histogram(f"weights_hist/{layer.name}", weights, step=0)

        writer.flush()

    # --- TensorBoard Scalars (дублируем для удобства выгрузки) ---
    writer = tf.summary.create_file_writer(log_dir)

    with writer.as_default():
        for epoch in range(len(history.history["loss"])):
            tf.summary.scalar("Loss/train", history.history["loss"][epoch], step=epoch)
            tf.summary.scalar("Loss/val", history.history["val_loss"][epoch], step=epoch)

            tf.summary.scalar("RMSE/train", history.history["rmse"][epoch], step=epoch)
            tf.summary.scalar("RMSE/val", history.history["val_rmse"][epoch], step=epoch)

            tf.summary.scalar("MAE/train", history.history["mae"][epoch], step=epoch)
            tf.summary.scalar("MAE/val", history.history["val_mae"][epoch], step=epoch)

            tf.summary.scalar("R2/train", history.history["r2_score"][epoch], step=epoch)
            tf.summary.scalar("R2/val", history.history["val_r2_score"][epoch], step=epoch)

        writer.flush()

    # Save Model
    os.makedirs(config["models"]["models_path"], exist_ok=True)
    model.save(config["models"]["models_path"] + "ann.keras")
    
    export_tb_scalars(log_dir=log_dir, save_dir=config["reports"]["figures_path"] + "tensorboard/")

    
    plot_weight_evolution(weights_history_cb.history, config["reports"]["figures_path"])

    # Predict (в лог-пространстве)
    y_pred_log = model.predict(x_train)

    # Возвращаем в исходное пространство
    y_pred = np.expm1(y_pred_log)
    y_true = y_train  # уже в оригинальном масштабе

    # Метрики
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R2: {r2}")
    
    metrics = {
    "train": {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2)
        }
    }

    os.makedirs("dvclive/xgboost", exist_ok=True)

    with open("dvclive/xgboost/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
# Model Creation
def create_ann_model(
    input_dim: int,
    n_hidden_layers: int,
    n_neurons: int,
    activation: str,
    learning_rate: float
) -> tf.keras.Model:

    # Input layer
    inputs: tf.keras.Input = tf.keras.Input(shape=(input_dim,), name="input_layer")

    # Hidden layers
    x = inputs

    for layer_idx in range(n_hidden_layers):
        x: tf.keras.layers.Dense = tf.keras.layers.Dense(units=n_neurons, activation=activation, kernel_initializer="he_normal", name=f"dense_hidden_{layer_idx+1}")(x)

    # Output layer (Regression)
    outputs: tf.Tensor = tf.keras.layers.Dense(units=1, activation="linear", name="output_layer")(x)

    # Model construction
    model: tf.keras.Model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ANN_regressor")

    # Compilation
    optimizer: tf.keras.optimizers.Adam = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=[
            tf.keras.metrics.RootMeanSquaredError(name='rmse'),
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.R2Score(name='r2_score'),
        ]
    )

    return model


def create_tf_dataset(
    x_train: np.ndarray,
    y_train: np.ndarray,
    validation_split: float,
    batch_size: int,
    random_state: int = 42
) -> tf.data.Dataset:

    # Split to train/val
    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=validation_split,
        random_state=random_state
        )

    # Make TF datasets
    train_ds: tf.data.Dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = (train_ds
                .shuffle(buffer_size=len(x_train), reshuffle_each_iteration=True)       # Перемешивает выборку данных на каждой эпохе
                .batch(batch_size)                                                      # Разделяет на батчи
                .cache()                                                                # Устраняет повторное чтение данных при каждой эпохе
                .prefetch(tf.data.AUTOTUNE))                                            # готовит следующий batch параллельно обучению текущего
    
    val_ds: tf.data.Dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_ds = (val_ds
              .batch(batch_size)
              .cache()
              .prefetch(tf.data.AUTOTUNE))

    return train_ds, val_ds



def export_tb_scalars(log_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    tags = ea.Tags()["scalars"]

    for tag in tags:
        events = ea.Scalars(tag)

        steps = [e.step for e in events]
        values = [e.value for e in events]

        plt.figure(figsize=(10, 5))
        plt.plot(steps, values)
        plt.title(tag)
        plt.xlabel("Step")
        plt.ylabel(tag)
        plt.grid()

        filename = tag.replace("/", "_") + ".png"
        plt.savefig(os.path.join(save_dir, filename))
        plt.close()

def plot_weight_evolution(weights_history, save_path):
    import numpy as np
    import matplotlib.pyplot as plt

    for layer_name, epochs_weights in weights_history.items():

        plt.figure(figsize=(10, 6))

        for i, weights in enumerate(epochs_weights):
            hist, bins = np.histogram(weights, bins=50, density=True)
            centers = (bins[:-1] + bins[1:]) / 2

            # смещение по оси Y (эпохи)
            plt.plot(centers, hist + i * 0.02, alpha=0.6)

        plt.title(f"{layer_name} Weight Distribution Evolution")
        plt.xlabel("Weight value")
        plt.ylabel("Epoch (stacked)")
        plt.grid()

        plt.savefig(f"{save_path}/{layer_name}_evolution.png", dpi=300)
        plt.close()

if __name__ == "__main__":
    main()