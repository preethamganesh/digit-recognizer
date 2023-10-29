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


from src.utils import load_json_file
from src.digit_recognizer.dataset import Dataset
from src.utils import add_to_log


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
