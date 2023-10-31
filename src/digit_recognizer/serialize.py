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
from typing import List

from src.digit_recognizer.train import Train
from src.utils import add_to_log
from src.utils import save_json_file
from src.utils import create_log
from src.utils import set_physical_devices_memory_limit


class ExportModel(tf.Module):
    """"""

    def __init__(self, model: tf.keras.Model, image_shape: List[int]) -> None:
        """Initializes the variables in the class.

        Initializes the variables in the class.

        Args:
            model: A tensorflow model for the model trained with latest checkpoints.
            image_shape: A list of integer for the size of input image.

        Returns:
            None.
        """
        # Asserts type of input arguments.
        assert isinstance(
            model, tf.keras.Model
        ), "Variable model should be of type 'tensorflow.keras.Model'."
        assert isinstance(
            image_shape, list
        ), "Variable image_shape should be of type 'tensorflow.keras.Model'."

        # Initializes class variables.
        self.model = model
        self.image_shape = image_shape


def serialize_model(model_version: str) -> None:
    """Serializes model files in the serialized model directory.

    Serializes model files in the serialized model directory.

    Args:
        model_version: A string for the version of the model about to be serialized.

    Returns:
        None.
    """
    # Checks type & values of arguments.
    assert isinstance(
        model_version, str
    ), "Variable model_version should be of type 'str'."

    # Creates an object for the Train class.
    trainer = Train(model_version)

    # Loads model configuration for current model version.
    trainer.load_model_configuration()

    # Loads the model with latest checkpoint.
    trainer.load_model("predict")

    # Build the tensorflow model graph, and logs the model summary.
    model = trainer.model.build_graph()
    model_summary = list()
    model.summary(print_fn=lambda x: model_summary.append(x))
    model_summary = "\n".join(model_summary)
    add_to_log(model_summary)
    add_to_log("")

    # Saves the updated model configuration dictionary as a JSON file.
    save_json_file(
        trainer.model_configuration,
        "v{}".format(model_version),
        "models/digit_recognizer/v{}/serialized".format(model_version),
    )
    add_to_log("")

    # Creates the input layer using the model configuration.
    inputs = [
        tf.ones(
            shape=(
                1,
                trainer.model_configuration["final_image_height"],
                trainer.model_configuration["final_image_width"],
                trainer.model_configuration["n_channels"],
            ),
            dtype=tf.float32,
        )
    ]

    # Passes the sample inputs through the model to create a callable object.
    _ = trainer.model(inputs, False, None)

    # Saves the tensorflow object created from the loaded model.
    home_directory_path = os.getcwd()
    tf.saved_model.save(
        trainer.model,
        "{}/models/digit_recognizer/v{}/serialized/model".format(
            home_directory_path, model_version
        ),
    )

    # Loads the serialized model to check if the loaded model is callable.
    model = tf.saved_model.load(
        "{}/models/digit_recognizer/v{}/serialized/model".format(
            home_directory_path, model_version
        )
    )
    _ = model(inputs, False, None)
    add_to_log("Finished serializing model & configuration files.")
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
    create_log("serialize_v{}".format(args.model_version), "logs/digit_recognizer")
    add_to_log("")

    # Sets memory limit of GPU if found in the system.
    set_physical_devices_memory_limit()

    start_time = time.time()

    # Serializes model files in the serialized model directory.
    serialize_model(args.model_version)

    add_to_log(
        "Finished saving the serialized model & its files in {} sec.".format(
            round(time.time() - start_time, 3)
        )
    )
    add_to_log("")


if __name__ == "__main__":
    main()
