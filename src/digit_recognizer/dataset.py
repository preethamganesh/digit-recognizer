import os

from typing import Dict, Any
import pandas as pd
from sklearn.utils import shuffle


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
        self.original_test_data = pd.read_csv(
            "{}/data/raw_data/digit_recognizer/test.csv".format(
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
            self.n_test_examples : self.n_test_examples + self.n_validation_examples
        ]
        self.new_train_data = self.original_train_data[
            self.n_test_examples + self.n_validation_examples :
        ]
