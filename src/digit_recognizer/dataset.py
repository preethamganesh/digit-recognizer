import os

from typing import Dict, Any, List
import pandas as pd
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np


class Dataset(object):
    """Loads the dataset based on model configuration."""

    def __init__(self, model_configuration: Dict[str, Any]) -> None:
        """Creates object attributes for the Dataset class.

        Creates object attributes for the Dataset class.

        Args:
            model_configuration: A dictionary for the configuration of model's current version.

        Returns:
            None.
        """
        # Asserts type & value of the arguments.
        assert isinstance(
            model_configuration, dict
        ), "Variable model_configuration should be of type 'dict'."

        # Initalizes class variables.
        self.model_configuration = model_configuration

    def load_data(self) -> None:
        """Loads original train & test CSV files as dataframes.

        Loads original train & test CSV files as dataframes.

        Args:
            None.

        Returns:
            None.
        """
        self.home_directory_path = os.getcwd()
        self.original_train_data = pd.read_csv(
            "{}/data/raw_data/digit_recognizer/train.csv".format(
                self.home_directory_path
            )
        )

    def split_dataset(self) -> None:
        """Splits original train data into new train, validation & test data.

        Splits original train data into new train, validation & test data.

        Args:
            None.

        Returns:
            None.
        """
        # Computes number of examples in the train, validation and test datasets.
        self.n_total_examples = len(self.original_train_data)
        self.n_validation_examples = int(
            self.model_configuration["validation_data_percentage"]
            * self.n_total_examples
        )
        self.n_test_examples = int(
            self.model_configuration["test_data_percentage"] * self.n_total_examples
        )
        self.n_train_examples = (
            self.n_total_examples - self.n_validation_examples - self.n_test_examples
        )

        # Shuffles the original train data.
        self.original_train_data = shuffle(self.original_train_data)

        # Splits the original train data into new train, validation & test data.
        self.new_validation_data = self.original_train_data[
            : self.n_validation_examples
        ]
        self.new_test_data = self.original_train_data[
            self.n_validation_examples : self.n_test_examples
            + self.n_validation_examples
        ]
        self.new_train_data = self.original_train_data[
            self.n_test_examples + self.n_validation_examples :
        ]

    def shuffle_slice_dataset(self) -> None:
        """Converts split data into tensor dataset & slices them based on batch size.

        Converts split data into input & target data. Zips the input & target data, and slices them based on batch size.

        Args:
            None.

        Returns:
            None.
        """
        # Zips images & classes into single tensor, and shuffles it.
        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            (
                self.new_train_data.drop(columns=["label"]),
                list(self.new_train_data["label"]),
            )
        )
        self.validation_dataset = tf.data.Dataset.from_tensor_slices(
            (
                self.new_validation_data.drop(columns=["label"]),
                list(self.new_validation_data["label"]),
            )
        )
        self.test_dataset = tf.data.Dataset.from_tensor_slices(
            (
                self.new_test_data.drop(columns=["label"]),
                list(self.new_test_data["label"]),
            )
        )

        # Slices the combined dataset based on batch size, and drops remainder values.
        self.train_dataset = self.train_dataset.batch(
            self.model_configuration["batch_size"], drop_remainder=True
        )
        self.validation_dataset = self.validation_dataset.batch(
            self.model_configuration["batch_size"], drop_remainder=True
        )
        self.test_dataset = self.test_dataset.batch(
            self.model_configuration["batch_size"], drop_remainder=True
        )

        # Computes number of steps per epoch for all dataset.
        self.n_train_steps_per_epoch = (
            len(self.new_train_data) // self.model_configuration["batch_size"]
        )
        self.n_validation_steps_per_epoch = (
            len(self.new_validation_data) // self.model_configuration["batch_size"]
        )
        self.n_test_steps_per_epoch = (
            len(self.new_test_data) // self.model_configuration["batch_size"]
        )

    def load_input_target_batches(
        self, images: np.ndarray, labels: np.ndarray
    ) -> List[tf.Tensor]:
        """Load input & target batchs for images & labels.

        Load input & target batchs for images & labels.

        Args:
            images: A NumPy array for images in current batch.
            labels: A NumPy array for labels in current batch.

        Returns:
            A list of tensors for the input & target batches generated from images & labels.
        """
        # Checks types & values of arguments.
        assert isinstance(
            images, np.ndarray
        ), "Variable images should be of type 'np.ndarray'."
        assert isinstance(
            labels, np.ndarray
        ), "Variable labels should be of type 'np.ndarray'."

        # Converts images into tensor of shape (batch, height, width, n_channels), and converts pixels into 0 - 1 range.
        input_batch = tf.convert_to_tensor(
            images.reshape(
                (
                    self.model_configuration["batch_size"],
                    self.model_configuration["final_image_height"],
                    self.model_configuration["final_image_width"],
                    self.model_configuration["n_channels"],
                )
            )
        )
        input_batch = tf.cast(input_batch, dtype=tf.float32)
        input_batch /= 255.0

        # Converts labels into categorical tensor.
        target_batch = tf.keras.utils.to_categorical(
            labels, num_classes=self.model_configuration["n_classes"]
        )
        return [input_batch, target_batch]
