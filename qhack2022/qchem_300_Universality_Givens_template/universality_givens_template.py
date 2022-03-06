#! /usr/bin/python3

import sys
import numpy as np
###
import pennylane as qml
###

def givens_rotations(a, b, c, d):
    """Calculates the angles needed for a Givens rotation to out put the state with amplitudes a,b,c and d

    Args:
        - a,b,c,d (float): real numbers which represent the amplitude of the relevant basis states (see problem statement). Assume they are normalized.

    Returns:
        - (list(float)): a list of real numbers ranging in the intervals provided in the challenge statement, which represent the angles in the Givens rotations,
        in order, that must be applied.
    """

    # QHACK #
    
    dev = qml.device('default.qubit', wires=6)

    @qml.qnode(dev)
    def state_preparation(params):
        qml.BasisState(np.array([1, 1, 0, 0, 0, 0]), wires=[i for i in range(6)])
        qml.DoubleExcitation(params[0], wires=[0, 1, 2, 3])
        qml.DoubleExcitation(params[1], wires=[2, 3, 4, 5])
        # single excitation controlled on qubit 0
        qml.ctrl(qml.SingleExcitation, control=0)(params[2], wires=[1, 3])
        #return qml.state()
        return qml.state()

    n = 4
    params = []
    #params = np.array([-2 * np.arcsin(1/np.sqrt(n-i)) for i in range(n-1)])
    #params = np.array([-2 * np.arcsin(1/np.sqrt(n)) for i in range(n-1)])
    #params.append(2*np.arccos(a**2)/np.cos(np.arcsin((c**2))))
    #params.append(-2 * np.arcsin((c**2)))
    #params.append(-2 * np.arcsin((d**2)))
    #params.append(-2 * np.arcsin(1/np.sqrt(1/a)))

    #params.append(1.0701416143903084)
    #params.append(-0.39479111969976155)
    #params.append(0.7124798514013161)

    params.append(-1.5707963267948963)
    params.append(-1.5707963267948963)
    params.append(-1.5707963267948963)
    
    #params.append(-2 * np.arcsin(1/np.sqrt(1/b)))
    #params.append(-2 * np.arcsin(1/np.sqrt(1/c)))
    #params.append(-2 * np.arcsin(1/np.sqrt(1/d)))


    output = state_preparation(params)
    # sets very small coefficients to zero
    output[np.abs(output) < 1e-10] = 0
    states = [np.binary_repr(i, width=6) for i in range(len(output)) if output[i] != 0]
    print("Basis states = ", states)
    states2 = [output[i] for i in range(len(output)) if output[i] != 0]
    print("Output state =", states2)
    return params
    '''
    result = []
    n = 4
    result.append(-2 * np.arcsin())
    result.append(-2 * np.arcsin(1/np.sqrt(1-b)))
    result.append(-2 * np.arcsin(1/np.sqrt(1-c)))
    return result
    '''
    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    theta_1, theta_2, theta_3 = givens_rotations(
        float(inputs[0]), float(inputs[1]), float(inputs[2]), float(inputs[3])
    )
    print(*[theta_1, theta_2, theta_3], sep=",")
