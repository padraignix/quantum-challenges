import sys
import pennylane as qml
from pennylane import numpy as np


def second_renyi_entropy(rho):
    """Computes the second Renyi entropy of a given density matrix."""
    # DO NOT MODIFY anything in this code block
    rho_diag_2 = np.diagonal(rho) ** 2.0
    return -np.real(np.log(np.sum(rho_diag_2)))


def compute_entanglement(theta):
    """Computes the second Renyi entropy of circuits with and without a tardigrade present.

    Args:
        - theta (float): the angle that defines the state psi_ABT

    Returns:
        - (float): The entanglement entropy of qubit B with no tardigrade
        initially present
        - (float): The entanglement entropy of qubit B where the tardigrade
        was initially present
    """

    dev = qml.device("default.qubit", wires=3)

    '''
    The compute_entanglement function will be where
    you need to prepare the two states given above (see mu_B and rho_B)
    µB = TrA(µAB) 
    ρB = TrA,T (ρABT )
    '''
    # QHACK #

    ### Paper https://arxiv.org/pdf/2112.07978.pdf

    result = []
    dev_n = qml.device("default.qubit", wires=2)
    @qml.qnode(dev_n)
    def circuit_no_t():
        qml.Hadamard(wires=0)
        qml.RX(theta, wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.density_matrix(wires=[1])

    test1 = circuit_no_t()
    result.append(second_renyi_entropy(test1))
    #print(result)

    @qml.qnode(dev)
    def circuit_theta(theta):

        ## trying to set the statevector directly from formula
        qml.QubitStateVector(np.array([0, (1/np.sqrt(2))*np.sin(theta/2), (1/np.sqrt(2))*np.cos(theta/2), 0, (1/np.sqrt(2))*1, 0, 0, 0]), wires=range(3))
        return qml.density_matrix(wires=[1])

    test2 = circuit_theta(theta)
    result.append(second_renyi_entropy(test2))

    return result
    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    theta = np.array(sys.stdin.read(), dtype=float)

    S2_without_tardigrade, S2_with_tardigrade = compute_entanglement(theta)
    print(*[S2_without_tardigrade, S2_with_tardigrade], sep=",")
