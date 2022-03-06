#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml


def distance(A, B):
    """Function that returns the distance between two vectors.

    Args:
        - A (list[int]): person's information: [age, minutes spent watching TV].
        - B (list[int]): person's information: [age, minutes spent watching TV].

    Returns:
        - (float): distance between the two feature vectors.
    """

    # QHACK #

    # The Swap test is a method that allows you to calculate |<A|B>|^2 , you could use it to help you.
    # The qml.AmplitudeEmbedding operator could help you too.

    # dev = qml.device("default.qubit", ...
    # @qml.qnode(dev)
    ### https://discuss.pennylane.ai/t/swap-test-using-different-registers/801/2
    '''
    ### https://discuss.pennylane.ai/t/swap-test-using-different-registers/801/4

    When we concatenate the two vector of features (each containing 2n features), 
    we shift to a higher-order feature space with dimension 2n . 
    The concatenated vector will contain 22n features. When we then embed the 
    concatenated features into a quantum circuit, the number of qubits needed for 
    the encoding also increases (and will be 2n , equal to the dimension of the feature 
    space).In the n=1 case this will mean that we can concatenate two input vectors of 
    length 2, to create a single feature vector of length 4. These features are then 
    embedded into log24=2 qubits. We can then compare the state of these two qubits by 
    introducing an ancilla qubit and performing the SWAP test.
    '''
    dev = qml.device('default.qubit', wires=3)
    @qml.qnode(dev)
    @qml.transforms.merge_amplitude_embedding
    def swaptest(a,b):
        
        qml.AmplitudeEmbedding(a, wires=[1], normalize=True)
        qml.AmplitudeEmbedding(b, wires=[2], normalize=True)
        qml.Hadamard(wires=0)
        qml.CSWAP(wires=[0,1,2])
        qml.Hadamard(wires=0)
        return qml.expval(qml.PauliZ(0))

    #ab = np.concatenate([A,B])
    #print (ab)
    out = swaptest(A,B)
    #print(out)
    dist = np.sqrt(2*(1-np.sqrt(out)))
    #print(dist)
    return dist
    # QHACK #


def predict(dataset, new, k):
    """Function that given a dataset, determines if a new person do like Beatles or not.

    Args:
        - dataset (list): List with the age, minutes that different people watch TV, and if they like Beatles.
        - new (list(int)): Age and TV minutes of the person we want to classify.
        - k (int): number of nearby neighbors to be taken into account.

    Returns:
        - (str): "YES" if they like Beatles, "NO" otherwise.
    """

    # DO NOT MODIFY anything in this code block

    def k_nearest_classes():
        """Function that returns a list of k near neighbors."""
        distances = []
        for data in dataset:
            distances.append(distance(data[0], new))
        nearest = []
        for _ in range(k):
            indx = np.argmin(distances)
            nearest.append(indx)
            distances[indx] += 2

        return [dataset[i][1] for i in nearest]

    output = k_nearest_classes()

    return (
        "YES" if len([i for i in output if i == "YES"]) > len(output) / 2 else "NO",
        float(distance(dataset[0][0], new)),
    )


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    dataset = []
    new = [int(inputs[0]), int(inputs[1])]
    k = int(inputs[2])
    for i in range(3, len(inputs), 3):
        dataset.append([[int(inputs[i + 0]), int(inputs[i + 1])], str(inputs[i + 2])])

    output = predict(dataset, new, k)
    sol = 0 if output[0] == "YES" else 1
    print(f"{sol},{output[1]}")
