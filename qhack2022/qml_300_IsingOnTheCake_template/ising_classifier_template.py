import sys
import pennylane as qml
from pennylane import numpy as np
import pennylane.optimize as optimize

DATA_SIZE = 250


def square_loss(labels, predictions):
    """Computes the standard square loss between model predictions and true labels.

    Args:
        - labels (list(int)): True labels (1/-1 for the ordered/disordered phases)
        - predictions (list(int)): Model predictions (1/-1 for the ordered/disordered phases)

    Returns:
        - loss (float): the square loss
    """

    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2

    loss = loss / len(labels)
    return loss


def accuracy(labels, predictions):
    """Computes the accuracy of the model's predictions against the true labels.

    Args:
        - labels (list(int)): True labels (1/-1 for the ordered/disordered phases)
        - predictions (list(int)): Model predictions (1/-1 for the ordered/disordered phases)

    Returns:
        - acc (float): The accuracy.
    """

    acc = 0
    for l, p in zip(labels, predictions):
        if abs(l - p) < 1e-5:
            acc = acc + 1
    acc = acc / len(labels)

    return acc


def classify_ising_data(ising_configs, labels):
    """Learn the phases of the classical Ising model.

    Args:
        - ising_configs (np.ndarray): 250 rows of binary (0 and 1) Ising model configurations
        - labels (np.ndarray): 250 rows of labels (1 or -1)

    Returns:
        - predictions (list(int)): Your final model predictions

    Feel free to add any other functions than `cost` and `circuit` within the "# QHACK #" markers 
    that you might need.
    """

    # QHACK #

    num_wires = ising_configs.shape[1] 
    dev = qml.device("default.qubit", wires=num_wires) 

    # Define a variational circuit below with your needed arguments and return something meaningful
    @qml.qnode(dev)
    def circuit(p1, p2, p3, p4):# delete this comment and put arguments here):
        ###??? Could we just do something silly like RY based on 0/1 ising data passed?
        ###     how would this be trained though... this is essentially "correct" 100%
        # qml.RY(p1 * np.pi, wires=1)
        # qml.RY(p2 * np.pi, wires=2)
        # qml.RY(p3 * np.pi, wires=3)
        # qml.RY(p4 * np.pi, wires=4)
        qml.Rot(p1[0], p1[1], p1[2], wires=1)
        qml.Rot(p2[0], p2[1], p2[2], wires=2)
        qml.Rot(p3[0], p3[1], p3[2], wires=3)
        qml.Rot(p4[0], p4[1], p4[2], wires=4)
        return [qml.expval(qml.PauliZ(i)) for i in range(num_wires)]

    # Define a cost function below with your needed arguments
    def cost(var1, var2, var3, var4):
        # the circuit function returns a numpy array of Pauli-Z expectation values
        spins = circuit(var1, var2)# delete this comment and put arguments here):

        # QHACK #
        
        # Insert an expression for your model predictions here
        
        #??????
        # total = spin[0] + spin[1] + spin[2] + spin[3]
        # if total == 4:
        #   predictions = 1
        # elif total == 0:
        #   predictions = 1
        # else:
        #   predictions = -1
        #########
        
        ### Patrick - if going with the spins/angles approach from example
        #             need to include spin 3 as well
        predictions = -(1 * spins[0] * spins[1]) - (-1 * spins[1] * spins[2])
        ### Y = ...
        # QHACK #
        ### Patrick - need to pick out and assign Y from labels before this call
        return square_loss(Y, predictions) # DO NOT MODIFY this line

    # optimize your circuit here

    ### Patrick - need to add training/optimization loop here
        ## for loop blah...
        #
        #
    # QHACK #

    return predictions


if __name__ == "__main__":
    inputs = np.array(
        sys.stdin.read().split(","), dtype=int, requires_grad=False
    ).reshape(DATA_SIZE, -1)
    ising_configs = inputs[:, :-1]
    labels = inputs[:, -1]
    predictions = classify_ising_data(ising_configs, labels)
    print(*predictions, sep=",")
