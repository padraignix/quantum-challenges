#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml


def qRAM(thetas):
    """Function that generates the superposition state explained above given the thetas angles.

    Args:
        - thetas (list(float)): list of angles to apply in the rotations.

    Returns:
        - (list(complex)): final state.
    """

    # QHACK #

    # Use this space to create auxiliary functions if you need it.
    def call_theta(i):
        return thetas[i]*5/8
    # QHACK #

    dev = qml.device("default.qubit", wires=range(4))

    @qml.qnode(dev)
    def circuit():

        # QHACK #

        # 000
        # 001
        # 010
        # 011
        # 100
        # 101
        # 110
        # 111
        qml.Hadamard(0)
        qml.Hadamard(1)
        qml.Hadamard(2)
        #qml.Hadamard(3)
        # Create your circuit: the first three qubits will refer to the index, the fourth to the RY rotation.

        for i in range(8):
            
            if i == 0:
                angle = call_theta(i)
                ### address 000
                qml.PauliX(0)
                qml.PauliX(1)
                qml.PauliX(2)
                qml.CRY((angle/3),wires=[0,3])
                qml.CRY(angle/3,wires=[1,3])
                qml.CRY(angle/3,wires=[2,3])
                qml.PauliX(0)
                qml.PauliX(1)
                qml.PauliX(2)

            if i == 1:
                angle = call_theta(i)
                ### address 001
                #qml.PauliX(0)
                qml.PauliX(1)
                qml.PauliX(2)
                qml.CRY((angle/3),wires=[0,3])
                qml.CRY(angle/3,wires=[1,3])
                qml.CRY(angle/3,wires=[2,3])
                #qml.PauliX(0)
                qml.PauliX(1)
                qml.PauliX(2)

            if i == 2:
                angle = call_theta(i)
                ### address 010
                qml.PauliX(0)
                #qml.PauliX(1)
                qml.PauliX(2)
                qml.CRY((angle/3),wires=[0,3])
                qml.CRY(angle/3,wires=[1,3])
                qml.CRY(angle/3,wires=[2,3])
                qml.PauliX(0)
                #qml.PauliX(1)
                qml.PauliX(2)
        
            if i == 3:
                angle = call_theta(i)
                ### address 011
                #qml.PauliX(0)
                #qml.PauliX(1)
                qml.PauliX(2)
                qml.CRY((angle/3),wires=[0,3])
                qml.CRY(angle/3,wires=[1,3])
                qml.CRY(angle/3,wires=[2,3])
                #qml.PauliX(0)
                #qml.PauliX(1)
                qml.PauliX(2)

            if i == 4:
                angle = call_theta(i)
                ### address 100
                qml.PauliX(0)
                qml.PauliX(1)
                #qml.PauliX(2)
                qml.CRY((angle/3),wires=[0,3])
                qml.CRY(angle/3,wires=[1,3])
                qml.CRY(angle/3,wires=[2,3])
                qml.PauliX(0)
                qml.PauliX(1)
                #qml.PauliX(2)
    
            if i == 5:
                angle = call_theta(i)
                ### address 101
                #qml.PauliX(0)
                qml.PauliX(1)
                #qml.PauliX(2)
                qml.CRY((angle/3),wires=[0,3])
                qml.CRY(angle/3,wires=[1,3])
                qml.CRY(angle/3,wires=[2,3])
                #qml.PauliX(0)
                qml.PauliX(1)
                #qml.PauliX(2)

            if i == 6:
                angle = call_theta(i)
                ### address 110
                qml.PauliX(0)
                #qml.PauliX(1)
                #qml.PauliX(2)
                qml.CRY((angle/3),wires=[0,3])
                qml.CRY(angle/3,wires=[1,3])
                qml.CRY(angle/3,wires=[2,3])
                qml.PauliX(0)
                #qml.PauliX(1)
                #qml.PauliX(2)

            if i == 7:
                angle = call_theta(i)
                ### address 111
                #qml.PauliX(0)
                #qml.PauliX(1)
                #qml.PauliX(2)
                qml.CRY((angle/3),wires=[0,3])
                qml.CRY(angle/3,wires=[1,3])
                qml.CRY(angle/3,wires=[2,3])
                #qml.PauliX(0)
                #qml.PauliX(1)
                #qml.PauliX(2)
        # QHACK #

        return qml.state()

    return circuit()


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    thetas = np.array(inputs, dtype=float)

    output = qRAM(thetas)
    output = [float(i.real.round(6)) for i in output]
    print(*output, sep=",")
