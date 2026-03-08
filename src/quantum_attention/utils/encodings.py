import pennylane as qml
import numpy as np
from typing import List

def angle_encoding(
    vector: List[float],
    qubits: List[int],
):
    """
    Encodes a classical vector using rotation angles.

    Args:
        vector (List[float]):
            Classical input vector.

        qubits (List[int]):
            Target qubits for encoding.

    """

    if len(vector) != len(qubits):
        raise ValueError(
            "Vector length must equal number of qubits for angle encoding."
        )

    for value, q in zip(vector, qubits):
        qml.RY(value, wires=q)


def amplitude_encoding(
    vector: List[float],
    qubits: List[int],
):
    """
    Encodes a vector into quantum amplitudes.

    Args:
        vector (List[float]):
            Classical input vector.

        qubits (List[int]):
            Target qubits.

 
    """

    vector = np.array(vector, dtype=float)

    dim = 2 ** len(qubits)

    if len(vector) != dim:
        raise ValueError(
            f"Vector length must be {dim} for {len(qubits)} qubits."
        )

    norm = np.linalg.norm(vector)

    if norm == 0:
        raise ValueError("Input vector cannot be zero.")

    normalized = vector / norm

    qml.AmplitudeEmbedding(
        normalized,
        wires=qubits,
        normalize=False
    )


def amplitude_state_prep(
    vector: List[float],
    wires: List[int],
):
    """
    Internal helper function that prepares |x>.

    Args:
        vector (List[float]):
            Classical vector.

        wires (List[int]):
            Data register qubits.
    """

    qml.AmplitudeEmbedding(
        vector,
        wires=wires,
        normalize=True
    )


def controlled_amplitude_encoding(
    vector: List[float],
    data_qubits: List[int],
    control_qubits: List[int],
    control_value: List[int],
):
    """
    Loads a vector into the data register only if the
    control register equals the specified binary value.

    Args:
        vector (List[float]):
            Embedding vector x_i.

        data_qubits (List[int]):
            Qubits storing the vector state.

        control_qubits (List[int]):
            Index register qubits.

        control_value (List[int]):
            Binary pattern that activates the loading.

    """

    controlled_prep = qml.ctrl(
        amplitude_state_prep,
        control=control_qubits,
        control_values=control_value
    )

    controlled_prep(vector, wires=data_qubits)

