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

from src.utils import load_json_file
from src.digit_recognizer.dataset import Dataset
from src.utils import add_to_log
from src.digit_recognizer.model import Model
from src.utils import check_directory_path_existence
from src.utils import save_json_file


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
        add_to_log(
            "No. of examples in the original test data: {}".format(
                len(self.dataset.original_test_data)
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

    def load_model(self) -> None:
        """Loads model & other utilies for training.

        Loads model & other utilies for training.

        Args:
            None.

        Returns:
            None.
        """
        # Loads model for current model configuration.
        self.model = Model(self.model_configuration)

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
        add_to_log("Finished loading model for current configuration.")
        add_to_log("")

    def generate_model_summary_and_plot(self) -> None:
        """Generates summary & plot for loaded model.

        Generates summary & plot for loaded model.

        Args:
            None.

        Returns:
            None.
        """
        # Compiles the model to log the model summary.
        model_summary = list()
        self.model.build().summary(print_fn=lambda x: model_summary.append(x))
        model_summary = "\n".join(model_summary)
        add_to_log(model_summary)
        add_to_log("")

        # Creates the following directory path if it does not exist.
        self.reports_directory_path = check_directory_path_existence(
            "models/digit_recognizer/v{}/reports".format(self.model_version)
        )

        # Plots the model & saves it as a PNG file.
        tf.keras.utils.plot_model(
            self.model.build(),
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
        self.loss_object = tf.keras.losses.BinaryCrossentropy(
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
        accuracy = tf.keras.metrics.binary_accuracy(target_batch, predicted_batch)
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
