#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml


dev = qml.device("default.qubit", wires=[0, 1, "sol"], shots=1)


def find_the_car(oracle):
    """Function which, given an oracle, returns which door that the car is behind.

    Args:
        - oracle (function): function that will act as an oracle. The first two qubits (0,1)
        will refer to the door and the third ("sol") to the answer.

    Returns:
        - (int): 0, 1, 2, or 3. The door that the car is behind.
    """

    @qml.qnode(dev)
    def circuit1():
        # QHACK #
        oracle()
        qml.PauliX(1)
        oracle()
        # QHACK #
        return qml.sample()

    @qml.qnode(dev)
    def circuit2():
        # QHACK #
        oracle()
        qml.PauliX(0)
        oracle()
        # QHACK #
        return qml.sample()

    sol1 = circuit1()
    sol2 = circuit2()

    # QHACK #
    '''
    So the two circuits check for 
    00 + 01 and 00 + 10
    Then depending on the output it's 100% defined

    If 1/0 = 01 door
    If 0/1 = 10 door
    If 1/1 = 00 door
    If 0/0 = 11 door
    '''
    #print(sol1)
    #print(sol2)
    # process sol1 and sol2 to determine which door the car is behind.
    sol11 = sol1[2]
    sol21 = sol2[2]
    #print(sol11)
    #print(sol21)
    if sol11 == 0 and sol21 == 0:
        return 3
    if sol11 == 1 and sol21 == 1:
        return 0
    if sol11 == 1 and sol21 == 0:
        return 1
    if sol11 == 0 and sol21 == 1:
        return 2
    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    numbers = [int(i) for i in inputs]

    def oracle():
        if numbers[0] == 1:
            qml.PauliX(wires=0)
        if numbers[1] == 1:
            qml.PauliX(wires=1)
        qml.Toffoli(wires=[0, 1, "sol"])
        if numbers[0] == 1:
            qml.PauliX(wires=0)
        if numbers[1] == 1:
            qml.PauliX(wires=1)

    output = find_the_car(oracle)
    print(f"{output}")
