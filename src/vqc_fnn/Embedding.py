import pennylane as qml
import math
import numpy as np


SUPPORTED_METHODS = ("angle", "amplitude", "iqp")


class EmbeddingLayer:
    """
    The Embedding layer Handles the encoding of classical data into quantum states.
    Supported methods: 'angle', 'amplitude', 'iqp'
    """

    def __init__(self, method="angle", rotation="Y", pad_to=None):

        self.method = method.lower()
        self.rotation = rotation
        self.pad_to = pad_to

        if self.method not in SUPPORTED_METHODS:

            raise ValueError(
                f"Unsupported embedding method: '{method}'. "
                f"Choose from {SUPPORTED_METHODS}"
            )

    def apply(self, features, wires):
        """This function applies the chosen embedding to the quantum tape."""

        if self.method == "angle":
            qml.AngleEmbedding(features=features, wires=wires, rotation=self.rotation)

        elif self.method == "amplitude":
            feats = self._maybe_pad(features, 2 ** len(wires))
            qml.AmplitudeEmbedding(features=feats, wires=wires, normalize=True)

        elif self.method == "iqp":
            qml.IQPEmbedding(features=features, wires=wires)

    def get_required_qubits(self, num_features):
        """This Function calculatesthe number of qubits are needed for
         the given feature count."""

        if self.method == "angle":
            return num_features
        elif self.method == "amplitude":
            return math.ceil(math.log2(max(num_features, 2)))
        elif self.method == "iqp":
            return num_features

    # I need to change the method, so that instead of accepting the target length it calcuate the target 
    # length from calling the get_required_qubits method by acceptinig only the number of feature and 
    # assuming the embeding method is AmplitudeEmbedding.

    @staticmethod
    def _maybe_pad(features, target_length):
        """This function pads a feature vector with zeros to reach *target_length*."""

        if hasattr(features, "__len__") and len(features) >= target_length:
            return features
        pad_width = target_length - len(features)
        return np.pad(features, (0, pad_width), mode="constant")
