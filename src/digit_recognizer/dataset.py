import os

from typing import Dict, Any
import pandas as pd


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
