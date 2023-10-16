import tensorflow as tf
from typing import Dict, Any


class Model(tf.keras.Model):
    """Recognizes digit in an image."""

    def __init__(self, model_configuration: Dict[str, Any]) -> None:
        """Initializes the layers in the recognition model.

        Initializes the layers in the recognition model, by adding convolutional, pooling, dropout & dense layers.

        Args:
            model_configuration: A dictionary for the configuration of model's current version.

        Returns:
            None.
        """
        super(Model, self).__init__()

        # Asserts type of input arguments.
        assert isinstance(
            model_configuration, dict
        ), "Variable model_configuration should be of type 'dict'."

        # Initializes class variables.
        self.model_configuration = model_configuration
        self.model_layers = dict()

        # Iterates across layers in the layers arrangement.
        self.model_layers = dict()
        for name in self.model_configuration["digit_recognition"]["arrangement"]:
            layer = self.model_configuration["digit_recognition"]["configuration"][name]

            # If layer's name is like 'conv2d_', a Conv2D layer is initialized based on layer configuration.
            if name.split("_")[0] == "conv2d":
                self.model_layers[name] = tf.keras.layers.Conv2D(
                    filters=layer["filters"],
                    kernel_size=layer["kernel_size"],
                    padding=layer["padding"],
                    strides=layer["strides"],
                    activation=layer["activation"],
                    name=name,
                )

            # If layer's name is like 'maxpool2d_', a MaxPool2D layer is initialized based on layer configuration.
            elif name.split("_")[0] == "maxpool2d":
                self.model_layers[name] = tf.keras.layers.MaxPool2D(
                    pool_size=layer["pool_size"],
                    strides=layer["strides"],
                    padding=layer["padding"],
                    name=name,
                )

            # If layer's name is like 'dropout_', a Dropout layer is initialized based on layer configuration.
            elif name.split("_")[0] == "dropout":
                self.model_layers[name] = tf.keras.layers.Dropout(rate=layer["rate"])

            # If layer's name is like 'dense_', a Dropout layer is initialized based on layer configuration.
            elif name.split("_")[0] == "dense":
                self.model_layers[name] = tf.keras.layers.Dense(
                    units=layer["units"],
                    activation=layer["activation"],
                    name=name,
                )

            # If layer's name is like 'flatten_', a Flatten layer is initialized.
            elif name.split("_")[0] == "flatten":
                self.model_layers[name] = tf.keras.layers.Flatten(name=name)
