#! /usr/bin/python3

import sys
import pennylane as qml
from pennylane import numpy as np

dev = qml.device("default.qubit", wires=1, shots=1)


@qml.qnode(dev)
def is_bomb(angle):
    """Construct a circuit at implements a one shot measurement at the bomb.

    Args:
        - angle (float): transmissivity of the Beam splitter, corresponding
        to a rotation around the Y axis.

    Returns:
        - (np.ndarray): a length-1 array representing result of the one-shot measurement
    """

    # QHACK #
    qml.Hadamard(wires=0)
    qml.RY(2*angle, wires=0)
    qml.Hadamard(wires=0)
    # QHACK #

    return qml.sample(qml.PauliZ(0))


@qml.qnode(dev)
def bomb_tester(angle):
    """Construct a circuit that implements a final one-shot measurement, given that the bomb does not explode

    Args:
        - angle (float): transmissivity of the Beam splitter right before the final detectors

    Returns:
        - (np.ndarray): a length-1 array representing result of the one-shot measurement
    """

    # QHACK #
    #qml.Hadamard(wires=0)
    qml.RY(2*angle, wires=0)
    #qml.Hadamard(wires=0)
    # QHACK #

    return qml.sample(qml.PauliZ(0))


def simulate(angle, n):
    """Concatenate n bomb circuits and a final measurement, and return the results of 10000 one-shot measurements

    Args:
        - angle (float): transmissivity of all the beam splitters, taken to be identical.
        - n (int): number of bomb circuits concatenated

    Returns:
        - (float): number of bombs successfully tested / number of bombs that didn't explode.
    """

    # QHACK #
    ### https://arxiv.org/pdf/quant-ph/9708016.pdf
    D_beeps = 0
    C_beeps = 0
    bombs_not_exploded = 0
    
    for i in range(1):
        exploded = 0
        test = is_bomb(angle*n % 1.5707)
        print(test)
        if test != 1:
            exploded = 1
        if exploded == 0:
            bombs_not_exploded += 1
            test2 = bomb_tester(angle)
            if test2 != 1:
                D_beeps += 1
            else:
                C_beeps += 1
    p = (D_beeps/bombs_not_exploded)
    return p
    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    output = simulate(float(inputs[0]), int(inputs[1]))
    print(f"{output}")
