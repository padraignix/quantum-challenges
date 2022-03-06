#! /usr/bin/python3

import sys
import pennylane as qml
from pennylane import numpy as np


def binary_list(m, n):
    """Converts number m to binary encoded on a list of length n

    Args:
        - m (int): Number to convert to binary
        - n (int): Number of wires in the circuit

    Returns:
        - (list(int)): Binary stored as a list of length n
    """

    arr = []
    # QHACK #
    arr = [int(x) for x in bin(m)[2:]]
    if n > len(arr):
        for _ in range(n - len(arr)):
            arr.insert(0, 0)
    # QHACK #
    return arr


def basis_states(n):
    """Given a number n, returns a list of all binary_list(m,n) for m < 2**n, thus providing all basis states
         for a circuit of n wires

    Args:
        - n(int): integer representing the number of wires in the circuit

    Returns:
        - (list(list(int))): list of basis states represented as lists of 0s and 1s.
    """

    arr = []

    # QHACK #
    for m in range(2 ** n):
        arr.append(binary_list(m, n))
    # QHACK #

    return arr


def is_particle_preserving(circuit, n):
    """Given a circuit and its number of wires n, returns 1 if it preserves the number of particles, and 0 if it does not

    Args:
        - circuit (qml.QNode): A QNode that has a state such as [0,0,1,0] as an input and outputs the final state after performing
        quantum operation
        - n (int): the number of wires of circuit

    Returns:
        - (bool): True / False according to whether the input circuit preserves the number of particles or not
    """

    # QHACK #
    basis = basis_states(n)
    for j in range(len(basis)):
        states = [np.binary_repr(i, width=n) for i in range(len(circuit(basis[j]))) if circuit(basis[j])[i] != 0]
        state_counts = []
        for s in states:
            state_counts.append(s.count('1'))
        if len(set(state_counts)) != 1:
            return False
    return True
    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(";")
    gate_list = []
    wire_list = []
    param_list = []
    i = 1

    while i < len(inputs):
        gate_obj = getattr(qml, str(inputs[i]))
        gate_wires = gate_obj.num_wires
        input_wires = list(map(int, str(inputs[i + 1]).split(",")))
        gate_list.append(str(inputs[i]))
        wire_list.append(input_wires)
        if "non_parametric_ops" not in gate_obj.__module__.split("."):
            input_params = list(map(float, str(inputs[i + 2]).split(",")))
            param_list.append(input_params)
            i += 1
        i += 2

    wire_list = np.array(wire_list, dtype=object)
    param_list = np.array(param_list, dtype=object)

    n = int(inputs[0])
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def circ(gate_list, wire_list, param_list, state):
        qml.BasisState(np.array(state), wires=range(n))
        j = 0
        for i in range(len(gate_list)):
            gate = getattr(qml, str(gate_list[i]))
            if "non_parametric_ops" not in gate.__module__.split("."):
                gate(*param_list[j], wires=[int(w) for w in wire_list[i]])
                j += 1
            else:
                gate(wires=[int(w) for w in wire_list[i]])
        return qml.state()

    def circuit(state):
        return circ(gate_list, wire_list, param_list, state)

    output = is_particle_preserving(circuit, n)

    print(output)
