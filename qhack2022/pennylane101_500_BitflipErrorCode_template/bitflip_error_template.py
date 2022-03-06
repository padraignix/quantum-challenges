#! /usr/bin/python3

from ftplib import parse150
import sys
import pennylane as qml
from pennylane import numpy as np


def error_wire(circuit_output):
    """Function that returns an error readout.

    Args:
        - circuit_output (?): the output of the `circuit` function.

    Returns:
        - (np.ndarray): a length-4 array that reveals the statistics of the
        error channel. It should display your algorithm's statistical prediction for
        whether an error occurred on wire `k` (k in {1,2,3}). The zeroth element represents
        the probability that a bitflip error does not occur.

        e.g., [0.28, 0.0, 0.72, 0.0] means a 28% chance no bitflip error occurs, but if one
        does occur it occurs on qubit #2 with a 72% chance.
    """
    #print(circuit_output)
    '''
    temp = 0
    result =[]

    z0 = circuit_output[0]
    z1 = circuit_output[1]
    z2 = circuit_output[2]
    prob = circuit_output [3:]
    print("Prob Wire [1-2]: {}".format(prob))
    print("ExpVal 0: {}".format(z0))
    print("ExpVal 1: {}".format(z1))
    print("ExpVal 2: {}".format(z2))
    
    if (z0 < z1) & (z0 < z2):
        print("Q0 is tampered")
        temp = 0
    if (z1 < z2) & (z1 < z0):
        print("Q1 is tampered")
        temp = 1
    if (z2 < z1) & (z2 < z0):
        print("Q2 is tampered")
        temp = 2
    
    if(temp == 0):
        result.append(prob[0])
        result.append(prob[3])
        result.append(0.0)
        result.append(0.0)
    elif(temp == 1):
        result.append(prob[0])
        result.append(0.0)
        result.append(prob[2])
        result.append(0.0)
    elif(temp == 2):
        result.append(prob[0])
        result.append(0.0)
        result.append(0.0)
        result.append(prob[1])

    print("Result formatted: {}".format(result))
    print("==========")
    '''
    result = []
    result.append(circuit_output[0])
    result.append(circuit_output[3])
    result.append(circuit_output[2])
    result.append(circuit_output[1])
    # QHACK #
    return result
    # process the circuit output here and return which qubit was the victim of a bitflip error!

    # QHACK #


dev = qml.device("default.mixed", wires=3)


@qml.qnode(dev)
def circuit(p, alpha, tampered_wire):
    """A quantum circuit that will be able to identify bitflip errors.

    DO NOT MODIFY any already-written lines in this function.

    Args:
        p (float): The bit flip probability
        alpha (float): The parameter used to calculate `density_matrix(alpha)`
        tampered_wire (int): The wire that may or may not be flipped (zero-index)

    Returns:
        Some expectation value, state, probs, ... you decide!
    """

    qml.QubitDensityMatrix(density_matrix(alpha), wires=[0, 1, 2])

    # QHACK #
    # put any input processing gates here
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[0, 2])

    qml.BitFlip(p, wires=int(tampered_wire))

    # put any gates here after the bitflip error has occurred
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[0, 2])
    qml.Toffoli(wires=[2, 1, 0])

    # return something!
    #return qml.probs(wires=[0])
    '''
    print(qml.probs(wires=[0]))
    p1 = qml.probs(wires=[1])
    p2 = qml.probs(wires=[2])
    print("Prob 0: {}".format(p0))
    print("Prob 1: {}".format(p1))
    print("Prob 2: {}".format(p2))

    print("0x1: ", qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)))
    print("1x2: ", qml.expval(qml.PauliZ(1) @ qml.PauliZ(2)))
    print("0x2: ", qml.expval(qml.PauliZ(0) @ qml.PauliZ(2)))
    '''
    #return qml.expval(qml.PauliZ(0)@ qml.PauliZ(1))
    return qml.probs(wires=[1,2])
    #return [qml.expval(qml.PauliZ(i)) for i in range(3)]
    #return [qml.probs(wires=[i]) for i in range(3)]
    #return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2)), qml.probs(wires=[1,2])


def density_matrix(alpha):
    """Creates a density matrix from a pure state."""
    # DO NOT MODIFY anything in this code block
    psi = alpha * np.array([1, 0], dtype=float) + np.sqrt(1 - alpha**2) * np.array(
        [0, 1], dtype=float
    )
    psi = np.kron(psi, np.array([1, 0, 0, 0], dtype=float))
    return np.outer(psi, np.conj(psi))


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = np.array(sys.stdin.read().split(","), dtype=float)
    p, alpha, tampered_wire = inputs[0], inputs[1], int(inputs[2])

    error_readout = np.zeros(4, dtype=float)
    circuit_output = circuit(p, alpha, tampered_wire)
    error_readout = error_wire(circuit_output)

    print(*error_readout, sep=",")
