import os
import sys
import warnings
import argparse
import logging


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_PATH)
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.FATAL)


import tensorflow as tf
import time
import matplotlib.pyplot as plt

from src.utils import load_json_file
from src.digit_recognizer.dataset import Dataset
from src.utils import add_to_log
from src.digit_recognizer.model import Model
from src.utils import check_directory_path_existence
from src.utils import save_json_file
from src.utils import create_log
from src.utils import set_physical_devices_memory_limit


class Train(object):
    """Trains the digit recognition model based on the configuration."""

    def __init__(self, model_version: str) -> None:
        """Creates object attributes for the Train class.

        Creates object attributes for the Train class.

        Args:
            model_version: A string for the version of the current model.

        Returns:
            None.
        """
        # Asserts type & value of the arguments.
        assert isinstance(model_version, str), "Variable model_version of type 'str'."

        # Initalizes class variables.
        self.model_version = model_version
        self.best_validation_loss = None

    def load_model_configuration(self) -> None:
        """Loads the model configuration file for current version.

        Loads the model configuration file for current version.

        Args:
            None.

        Returns:
            None.
        """
        self.home_directory_path = os.getcwd()
        model_configuration_directory_path = (
            "{}/configs/models/digit_recognizer".format(self.home_directory_path)
        )
        self.model_configuration = load_json_file(
            "v{}".format(self.model_version), model_configuration_directory_path
        )

    def load_dataset(self) -> None:
        """Loads dataset based on model configuration.

        Loads dataset based on model configuration.

        Args:
            None.

        Returns:
            None.
        """
        # Initializes object for the Dataset class.
        self.dataset = Dataset(self.model_configuration)

        # Loads original train & test CSV files as dataframes.
        self.dataset.load_data()
        add_to_log(
            "No. of examples in the original train data: {}".format(
                len(self.dataset.original_train_data)
            )
        )
        add_to_log("")

        # Splits original train data into new train, validation & test data.
        self.dataset.split_dataset()
        add_to_log(
            "No. of examples in the new train data: {}".format(
                self.dataset.n_train_examples
            )
        )
        add_to_log(
            "No. of examples in the new validation data: {}".format(
                self.dataset.n_validation_examples
            )
        )
        add_to_log(
            "No. of examples in the new test data: {}".format(
                self.dataset.n_test_examples
            )
        )
        add_to_log("")

        # Converts split data tensorflow dataset and slices them based on batch size.
        self.dataset.shuffle_slice_dataset()
        add_to_log(
            "No. of train steps per epoch: {}".format(
                self.dataset.n_train_steps_per_epoch
            )
        )
        add_to_log(
            "No. of validation steps per epoch: {}".format(
                self.dataset.n_validation_steps_per_epoch
            )
        )
        add_to_log(
            "No. of test steps per epoch: {}".format(
                self.dataset.n_test_steps_per_epoch
            )
        )
        add_to_log("")

    def load_model(self, mode: str) -> None:
        """Loads model & other utilies for training.

        Loads model & other utilies for training.

        Args:
            mode: A string for mode by which the should be loaded, i.e., with latest checkpoints or not.

        Returns:
            None.
        """
        # Asserts type & value of the arguments.
        assert isinstance(mode, str), "Variable mode should be of type 'str'."
        assert mode in [
            "train",
            "predict",
        ], "Variable mode should have 'train' or 'predict' as value."

        # Loads model for current model configuration.
        self.model = Model(self.model_configuration)

        # Loads the optimizer.
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.model_configuration["model"]["optimizer"][
                "learning_rate"
            ]
        )

        # Creates checkpoint manager for the neural network model and loads the optimizer.
        self.checkpoint_directory_path = (
            "{}/models/digit_recognizer/v{}/checkpoints".format(
                self.home_directory_path, self.model_version
            )
        )
        checkpoint = tf.train.Checkpoint(model=self.model)
        self.manager = tf.train.CheckpointManager(
            checkpoint, directory=self.checkpoint_directory_path, max_to_keep=3
        )

        # If mode is predict, then the trained checkpoint is restored.
        if mode == "predict":
            checkpoint.restore(
                tf.train.latest_checkpoint(self.checkpoint_directory_path)
            )

        add_to_log("Finished loading model for current configuration.")
        add_to_log("")

    def generate_model_summary_and_plot(self, plot: bool) -> None:
        """Generates summary & plot for loaded model.

        Generates summary & plot for loaded model.

        Args:
            pool: A boolean value to whether generate model plot or not.

        Returns:
            None.
        """
        # Builds plottable graph for the model.
        model = self.model.build_graph()

        # Compiles the model to log the model summary.
        model_summary = list()
        model.summary(print_fn=lambda x: model_summary.append(x))
        model_summary = "\n".join(model_summary)
        add_to_log(model_summary)
        add_to_log("")

        # Creates the following directory path if it does not exist.
        self.reports_directory_path = check_directory_path_existence(
            "models/digit_recognizer/v{}/reports".format(self.model_version)
        )

        # Plots the model & saves it as a PNG file.
        if plot:
            tf.keras.utils.plot_model(
                model,
                "{}/model_plot.png".format(self.reports_directory_path),
                show_shapes=True,
                show_layer_names=True,
                expand_nested=True,
            )
            add_to_log(
                "Finished saving model plot at {}/model_plot.png.".format(
                    self.reports_directory_path
                )
            )
            add_to_log("")

    def initialize_model_history(self) -> None:
        """Creates empty dictionary for saving the model metrics for the current model.

        Creates empty dictionary for saving the model metrics for the current model.

        Args:
            None.

        Returns:
            None.
        """
        self.model_history = {
            "epoch": list(),
            "train_loss": list(),
            "validation_loss": list(),
            "train_accuracy": list(),
            "validation_accuracy": list(),
        }

    def initialize_metric_trackers(self) -> None:
        """Initializes trackers which computes the mean of all metrics.

        Initializes trackers which computes the mean of all metrics.

        Args:
            None.

        Returns:
            None.
        """
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.validation_loss = tf.keras.metrics.Mean(name="validation_loss")
        self.train_accuracy = tf.keras.metrics.Mean(name="train_accuracy")
        self.validation_accuracy = tf.keras.metrics.Mean(name="validation_accuracy")

    def update_model_history(self, epoch: int) -> None:
        """Updates model history dictionary with latest metrics & saves it as JSON file.

        Updates model history dictionary with latest metrics & saves it as JSON file.

        Args:
            epoch: An integer for the number of current epoch.

        Returns:
            None.
        """
        # Asserts type & value of the arguments.
        assert isinstance(epoch, int), "Variable epoch should be of type 'int'."

        # Updates the metrics dictionary with the metrics for the current training & validation metrics.
        self.model_history["epoch"].append(epoch + 1)
        self.model_history["train_loss"].append(
            str(round(self.train_loss.result().numpy(), 3))
        )
        self.model_history["validation_loss"].append(
            str(round(self.validation_loss.result().numpy(), 3))
        )
        self.model_history["train_accuracy"].append(
            str(round(self.train_accuracy.result().numpy(), 3))
        )
        self.model_history["validation_accuracy"].append(
            str(round(self.validation_accuracy.result().numpy(), 3))
        )

        # Saves the model history dictionary as a JSON file.
        save_json_file(
            self.model_history,
            "history",
            "models/digit_recognizer/v{}/reports".format(
                self.model_configuration["version"]
            ),
        )

    def compute_loss(
        self, target_batch: tf.Tensor, predicted_batch: tf.Tensor
    ) -> tf.Tensor:
        """Computes loss for the current batch using actual & predicted values.

        Computes loss for the current batch using actual & predicted values.

        Args:
            target_batch: A tensor for the the actual values for the current batch.
            predicted_batch: A tensor for the predicted values for the current batch.

        Returns:
            A tensor for the loss for the current batch.
        """
        # Asserts type & value of the arguments.
        assert isinstance(
            target_batch, tf.Tensor
        ), "Variable target_batch should be of type 'tf.Tensor'."
        assert isinstance(
            predicted_batch, tf.Tensor
        ), "Variable predicted_batch should be of type 'tf.Tensor'."

        # Computes loss for the current batch using actual values and predicted values.
        self.loss_object = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, reduction="none"
        )
        loss = self.loss_object(target_batch, predicted_batch)
        return loss

    def compute_accuracy(
        self, target_batch: tf.Tensor, predicted_batch: tf.Tensor
    ) -> tf.Tensor:
        """Computes accuracy for the current batch using actual & predicted values.

        Computes accuracy for the current batch using actual & predicted values.

        Args:
            target_batch: A tensor which contains the actual values for the current batch.
            predicted_batch: A tensor which contains the predicted values for the current batch.

        Returns:
            A tensor for the accuracy of current batch.
        """
        # Asserts type & value of the arguments.
        assert isinstance(
            target_batch, tf.Tensor
        ), "Variable target_batch should be of type 'tf.Tensor'."
        assert isinstance(
            predicted_batch, tf.Tensor
        ), "Variable predicted_batch should be of type 'tf.Tensor'."

        # Computes accuracy for the current batch using actual values and predicted values.
        accuracy = tf.keras.metrics.categorical_accuracy(target_batch, predicted_batch)
        return accuracy

    @tf.function
    def train_step(self, input_batch: tf.Tensor, target_batch: tf.Tensor) -> None:
        """Trains model using current input & target batches.

        Trains model using current input & target batches.

        Args:
            input_batch: A tensor for the input text from the current batch for training the model.
            target_batch: A tensor for the target text from the current batch for training and validating the model.

        Returns:
            None.
        """
        # Asserts type & value of the arguments.
        assert isinstance(
            input_batch, tf.Tensor
        ), "Variable input_batch should be of type 'tf.Tensor'."
        assert isinstance(
            target_batch, tf.Tensor
        ), "Variable target_batch should be of type 'tf.Tensor'."

        # Computes the model output for current batch, and metrics for current model output.
        with tf.GradientTape() as tape:
            predictions = self.model([input_batch], True, None)
            loss = self.compute_loss(target_batch, predictions)
            accuracy = self.compute_accuracy(target_batch, predictions)

        # Computes gradients using loss and model variables.
        gradients = tape.gradient(loss, self.model.trainable_variables)

        # Uses optimizer to apply the computed gradients on the combined model variables.
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Computes batch metrics and appends it to main metrics.
        self.train_loss(loss)
        self.train_accuracy(accuracy)

    def validation_step(self, input_batch: tf.Tensor, target_batch: tf.Tensor) -> None:
        """Validates model using current input & target batches.

        Validates model using current input & target batches.

        Args:
            input_batch: A tensor for the input text from the current batch for validating the model.
            target_batch: A tensor for the target text from the current batch for validating the model.

        Returns:
            None.
        """
        # Asserts type & value of the arguments.
        assert isinstance(
            input_batch, tf.Tensor
        ), "Variable input_batch should be of type 'tf.Tensor'."
        assert isinstance(
            target_batch, tf.Tensor
        ), "Variable target_batch should be of type 'tf.Tensor'."

        # Computes the model output for current batch, and metrics for current model output.
        predictions = self.model([input_batch], False, None)
        loss = self.compute_loss(target_batch, predictions)
        accuracy = self.compute_accuracy(target_batch, predictions)

        # Computes batch metrics and appends it to main metrics.
        self.validation_loss(loss)
        self.validation_accuracy(accuracy)

    def reset_trackers(self) -> None:
        """Resets states for trackers before the start of each epoch.

        Resets states for trackers before the start of each epoch.

        Args:
            None.

        Returns:
            None.
        """
        self.train_loss.reset_states()
        self.validation_loss.reset_states()
        self.train_accuracy.reset_states()
        self.validation_accuracy.reset_states()

    def train_model_per_epoch(self, epoch: int) -> None:
        """Trains the model using train dataset for current epoch.

        Trains the model using train dataset for current epoch.

        Args:
            epoch: An integer for the number of current epoch.

        Returns:
            None.
        """
        # Asserts type & value of the arguments.
        assert isinstance(epoch, int), "Variable current_epoch should be of type 'int'."

        # Iterates across batches in the train dataset.
        for batch, (images, labels) in enumerate(
            self.dataset.train_dataset.take(self.dataset.n_train_steps_per_epoch)
        ):
            batch_start_time = time.time()

            # Loads input & target sequences for current batch as tensors.
            input_batch, target_batch = self.dataset.load_input_target_batches(
                images.numpy(), labels.numpy()
            )

            # Trains the model using the current input and target batch.
            self.train_step(input_batch, target_batch)
            batch_end_time = time.time()

            add_to_log(
                "Epoch={}, Batch={}, Train loss={}, Train accuracy={}, Time taken={} sec.".format(
                    epoch + 1,
                    batch,
                    str(round(self.train_loss.result().numpy(), 3)),
                    str(round(self.train_accuracy.result().numpy(), 3)),
                    round(batch_end_time - batch_start_time, 3),
                )
            )
        add_to_log("")

    def validate_model_per_epoch(self, epoch: int) -> None:
        """Validates the model using validation dataset for current epoch.

        Validates the model using validation dataset for current epoch.

        Args:
            epoch: An integer for the number of current epoch.

        Returns:
            None.
        """
        # Asserts type & value of the arguments.
        assert isinstance(epoch, int), "Variable current_epoch should be of type 'int'."

        # Iterates across batches in the train dataset.
        for batch, (image_ids, label_ids) in enumerate(
            self.dataset.validation_dataset.take(
                self.dataset.n_validation_steps_per_epoch
            )
        ):
            batch_start_time = time.time()

            # Loads input & target sequences for current batch as tensors.
            input_batch, target_batch = self.dataset.load_input_target_batches(
                image_ids.numpy(), label_ids.numpy()
            )

            # Validates the model using the current input and target batch.
            self.validation_step(input_batch, target_batch)
            batch_end_time = time.time()

            add_to_log(
                "Epoch={}, Batch={}, Validation loss={}, Validation accuracy={}, Time taken={} sec.".format(
                    epoch + 1,
                    batch,
                    str(round(self.train_loss.result().numpy(), 3)),
                    str(round(self.train_accuracy.result().numpy(), 3)),
                    round(batch_end_time - batch_start_time, 3),
                )
            )
        add_to_log("")

    def save_model(self) -> None:
        """Saves the model after checking performance metrics in current epoch.

        Saves the model after checking performance metrics in current epoch.

        Args:
            None.

        Returns:
            None.
        """
        self.manager.save()
        add_to_log("Checkpoint saved at {}.".format(self.checkpoint_directory_path))

    def early_stopping(self) -> bool:
        """Stops the model from learning further if the performance has not improved from previous epoch.

        Stops the model from learning further if the performance has not improved from previous epoch.

        Args:
            None.

        Returns:
            None.
        """
        # If epoch = 1, then best validation loss is replaced with current validation loss, & the checkpoint is saved.
        if self.best_validation_loss is None:
            self.patience_count = 0
            self.best_validation_loss = str(
                round(self.validation_loss.result().numpy(), 3)
            )
            self.save_model()

        # If best validation loss is higher than current validation loss, the best validation loss is replaced with
        # current validation loss, & the checkpoint is saved.
        elif self.best_validation_loss > str(
            round(self.validation_loss.result().numpy(), 3)
        ):
            self.patience_count = 0
            add_to_log(
                "Best validation loss changed from {} to {}".format(
                    str(self.best_validation_loss),
                    str(round(self.validation_loss.result().numpy(), 3)),
                )
            )
            self.best_validation_loss = str(
                round(self.validation_loss.result().numpy(), 3)
            )
            self.save_model()

        # If best validation loss is not higher than the current validation loss, then the number of times the model
        # has not improved is incremented by 1.
        elif self.patience_count < 2:
            self.patience_count += 1
            add_to_log("Best validation loss did not improve.")
            add_to_log("Checkpoint not saved.")

        # If the number of times the model did not improve is greater than 4, then model is stopped from training.
        else:
            return False
        return True

    def fit(self) -> None:
        """Trains & validates the loaded model using train & validation dataset.

        Trains & validates the loaded model using train & validation dataset.

        Args:
            None.

        Returns:
            None.
        """
        # Initializes TensorFlow trackers which computes the mean of all metrics.
        self.initialize_metric_trackers()

        # Initializes model history dataframe.
        self.initialize_model_history()

        # Iterates across epochs for training the neural network model.
        for epoch in range(self.model_configuration["epochs"]):
            epoch_start_time = time.time()

            # Resets states for training and validation metrics before the start of each epoch.
            self.reset_trackers()

            # Trains the model using batces in the train dataset.
            self.train_model_per_epoch(epoch)

            # Validates the model using batches in the validation dataset.
            self.validate_model_per_epoch(epoch)

            # Updates model history dataframe with performance metrics for current epoch.
            self.update_model_history(epoch)

            epoch_end_time = time.time()
            add_to_log(
                "Epoch={}, Train loss={}, Validation loss={}, Train Accuracy={}, Validation Accuracy={}, "
                "Time taken={} sec.".format(
                    epoch + 1,
                    str(round(self.train_loss.result().numpy(), 3)),
                    str(round(self.validation_loss.result().numpy(), 3)),
                    str(round(self.train_accuracy.result().numpy(), 3)),
                    str(round(self.validation_accuracy.result().numpy(), 3)),
                    round(epoch_end_time - epoch_start_time, 3),
                )
            )

            # Stops the model from learning further if the performance has not improved from previous epoch.
            model_training_status = self.early_stopping()
            if not model_training_status:
                add_to_log(
                    "Model did not improve after 4th time. Model stopped from training further."
                )
                add_to_log("")
                break
            add_to_log("")

    def generate_model_history_plot(self, metric_name: str) -> None:
        """Generates plot for model training & validation history.

        Generates plot for model training & validation history.

        Args:
            metric_name: A string for the name of the current metric for which the plot should be generated.

        Returns:
            None.
        """
        # Asserts type & value of the arguments.
        assert isinstance(
            metric_name, str
        ), "Variable metric_name should be of type 'str'."
        assert metric_name in [
            "loss",
            "accuracy",
        ], "Variable metric_name should have value as 'loss' or 'accuracy'."

        # Specifications used to generate the plot, i.e., font size and size of the plot.
        font = {"size": 28}
        plt.rc("font", **font)
        plt.figure(num=None, figsize=(30, 15))

        # Converts train and validation metrics from string format to floating point format.
        epochs = [i for i in range(1, len(self.model_history) + 1)]
        train_metrics = self.model_history["train_{}".format(metric_name)]
        train_metrics = [float(train_metrics[i]) for i in range(len(train_metrics))]
        validation_metrics = self.model_history["validation_{}".format(metric_name)]
        validation_metrics = [
            float(validation_metrics[i]) for i in range(len(validation_metrics))
        ]

        # Generates plot for training and validation metrics
        plt.plot(
            epochs,
            train_metrics,
            color="orange",
            linewidth=3,
            label="train_{}".format(metric_name),
        )
        plt.plot(
            epochs,
            validation_metrics,
            color="blue",
            linewidth=3,
            label="validation_{}".format(metric_name),
        )

        # Generates the plot for the epochs vs metrics.
        plt.xlabel("epochs")
        plt.ylabel(metric_name)
        plt.xticks(epochs)
        plt.legend(loc="upper left")
        plt.grid(color="black", linestyle="-.", linewidth=2, alpha=0.3)

        # Saves plot using the following path.
        plt.savefig(
            "{}/model_history_{}.png".format(
                self.reports_directory_path,
                metric_name,
            )
        )
        plt.close()

    def test_model(self) -> None:
        """Tests the trained model using the test dataset.

        Tests the trained model using the test dataset.

        Args:
            None.

        Returns:
            None.
        """
        # Resets states for validation metrics.
        self.reset_trackers()

        # Iterates across batches in the train dataset.
        for batch, (image_ids, label_ids) in enumerate(
            self.dataset.test_dataset.take(self.dataset.n_test_steps_per_epoch)
        ):
            # Loads input & target sequences for current batch as tensors.
            input_batch, target_batch = self.dataset.load_input_target_batches(
                image_ids.numpy(), label_ids.numpy()
            )

            # Tests the model using the current input and target batch.
            self.validation_step(input_batch, target_batch)

        add_to_log(
            "Test loss={}.".format(str(round(self.validation_loss.result().numpy(), 3)))
        )
        add_to_log(
            "Test accuracy={}.".format(
                str(round(self.validation_accuracy.result().numpy(), 3))
            ),
        )
        add_to_log("")


def main():
    # Parses the arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-mv",
        "--model_version",
        type=str,
        required=True,
        help="Version by which the trained model files should be saved as.",
    )
    args = parser.parse_args()

    # Creates an logger object for storing terminal output.
    create_log("train_v{}".format(args.model_version), "logs/digit_recognizer")
    add_to_log("")

    # Sets memory limit of GPU if found in the system.
    set_physical_devices_memory_limit()

    # Creates an object for the Train class.
    trainer = Train(args.model_version)

    # Loads model configuration for current model version.
    trainer.load_model_configuration()

    # Loads dataset based on dataset version in the model configuration.
    trainer.load_dataset()

    # Loads model & other utilies for training it.
    trainer.load_model()

    # Generates summary and plot for loaded model.
    trainer.generate_model_summary_and_plot(True)

    # Trains & validates the model using train & validation dataset.
    trainer.fit()

    # Generates model history plots for all performance metrics.
    trainer.generate_model_history_plot("loss")
    trainer.generate_model_history_plot("accuracy")

    # Loads the model with latest checkpoint.
    trainer.load_model("predict")

    # Tests the model using the test dataset.
    trainer.test_model()


if __name__ == "__main__":
    main()
