#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml


def switch(oracle):
    """Function that, given an oracle, returns a list of switches that work by executing a
    single circuit with a single shot. The code you write for this challenge should be completely
    contained within this function between the # QHACK # comment markers.

    Args:
        - oracle (function): oracle that simulates the behavior of the lights.

    Returns:
        - (list(int)): List with the switches that work. Example: [0,2].
    """

    dev = qml.device("default.qubit", wires=[0, 1, 2, "light"], shots=1)

    @qml.qnode(dev)
    def circuit():

        # QHACK #
        #### Phase kick back
        # You are allowed to place operations before and after the oracle without any problem.
    

        qml.PauliX(0)
        oracle()
        qml.Hadamard(0)
        qml.Hadamard("light")
        qml.CNOT(wires=[0,"light"])
        qml.Hadamard("light")
        qml.Hadamard(0)
        qml.PauliX(0)

        qml.CNOT(wires=[0,'light'])

        qml.PauliX(0)
        qml.PauliX(1)
        oracle()
        qml.Hadamard(1)
        qml.Hadamard("light")
        qml.CNOT(wires=[1,"light"])
        qml.Hadamard("light")
        qml.Hadamard(1)
        qml.PauliX(1)
        qml.PauliX(0)

        qml.CNOT(wires=[1,'light']) 

        qml.PauliX(0)
        qml.PauliX(1)
        qml.PauliX(2)
        oracle()
        qml.Hadamard(2)
        qml.Hadamard("light")
        qml.CNOT(wires=[2,"light"])
        qml.Hadamard("light")
        qml.Hadamard(2)
        qml.PauliX(2)
        qml.PauliX(1)
        qml.PauliX(0)

        qml.CNOT(wires=[2,'light']) 
        # QHACK #

        #return qml.sample(wires=[0,1,2,"light"])
        return qml.sample(wires=range(3))

    sample = circuit()

    result = []
    for i in range(len(sample)):
        if sample[i] == 1:
            result.append(i)
    #print(sample)
    return result
    # QHACK #

    # Process the received sample and return the requested list.

    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    numbers = [int(i) for i in inputs]

    def oracle():
        for i in numbers:
            qml.CNOT(wires=[i, "light"])

    output = switch(oracle)
    print(*output, sep=",")
