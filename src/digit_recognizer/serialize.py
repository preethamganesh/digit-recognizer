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

from src.digit_recognizer.train import Train
from src.utils import add_to_log
from src.utils import save_json_file


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
        tf.keras.layers.Input(
            shape=(
                trainer.model_configuration["final_image_height"],
                trainer.model_configuration["final_image_width"],
                trainer.model_configuration["n_channels"],
            )
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
