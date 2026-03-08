import pennylane as qml
from typing import List


def controlled_swap(control: int, q1: int, q2: int):
    """
    Controlled SWAP (Fredkin gate).

    Args:
        control (int): control qubit
        q1 (int): first target qubit
        q2 (int): second target qubit
    """

    qml.CSWAP(wires=[control, q1, q2])


def prepare_uniform_superposition(qubits: List[int]):
    """
    Prepares uniform superposition across qubits.

    Args:
        qubits (List[int])
    """

    for q in qubits:
        qml.Hadamard(wires=q)


def entangling_layer(qubits: List[int]):
    """
    Simple chain entanglement layer.

    Args:
        qubits (List[int])
    """

    for i in range(len(qubits) - 1):
        qml.CNOT(wires=[qubits[i], qubits[i + 1]])


def parameterized_rotation_layer(
    qubits: List[int],
    parameters: List[float],
):
    """
    Parameterized rotation layer.

    Args:
        qubits (List[int])
        parameters (List[float])

   
    """

    for q, theta in zip(qubits, parameters):
        qml.RY(theta, wires=q)


def grover_diffusion(qubits: List[int]):
    """
    Grover diffusion operator.

    Args:
        qubits (List[int])
    """

   
    for q in qubits:
        qml.Hadamard(wires=q)

    for q in qubits:
        qml.PauliX(wires=q)

   
    qml.Hadamard(wires=qubits[-1])
    qml.MultiControlledX(
        control_wires=qubits[:-1],
        wires=qubits[-1],
    )
    qml.Hadamard(wires=qubits[-1])

  
    for q in qubits:
        qml.PauliX(wires=q)

   
    for q in qubits:
        qml.Hadamard(wires=q)