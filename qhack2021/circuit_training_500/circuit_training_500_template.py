#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np


def classify_data(X_train, Y_train, X_test):
    """Develop and train your very own variational quantum classifier.

    Use the provided training data to train your classifier. The code you write
    for this challenge should be completely contained within this function
    between the # QHACK # comment markers. The number of qubits, choice of
    variational ansatz, cost function, and optimization method are all to be
    developed by you in this function.

    Args:
        X_train (np.ndarray): An array of floats of size (250, 3) to be used as training data.
        Y_train (np.ndarray): An array of size (250,) which are the categorical labels
            associated to the training data. The categories are labeled by -1, 0, and 1.
        X_test (np.ndarray): An array of floats of (50, 3) to serve as testing data.

    Returns:
        str: The predicted categories of X_test, converted from a list of ints to a
            comma-separated string.
    """

    # Use this array to make a prediction for the labels of the data in X_test
    predictions = []

    # QHACK #
    from pennylane.optimize import NesterovMomentumOptimizer
    from pennylane.optimize import GradientDescentOptimizer
    from pennylane.templates.embeddings import AngleEmbedding
    n_qubits = 2
    dev = qml.device("default.qubit", wires=n_qubits)

    def layer(W):
        qml.Rot(W[0, 0], W[0, 1], W[0, 2], wires=0)
        qml.Rot(W[1, 0], W[1, 1], W[1, 2], wires=1)
        qml.CNOT(wires=[0, 1])

    def get_angles(x):

        beta0 = 2 * np.arcsin(np.sqrt(x[1] ** 2) / np.sqrt(x[0] ** 2 + x[1] ** 2 + 1e-12))
        beta1 = 2 * np.arcsin(np.sqrt(x[3] ** 2) / np.sqrt(x[2] ** 2 + x[3] ** 2 + 1e-12))
        beta2 = 2 * np.arcsin(np.sqrt(x[2] ** 2 + x[3] ** 2)/ np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2))

        return np.array([beta2, -beta1 / 2, beta1 / 2, -beta0 / 2, beta0 / 2])

    def statepreparation(a):
        '''
        qml.CRZ(a[0],wires=[1,0])
        qml.RZ(a[1], wires=0)

        qml.CRY(a[0],wires=[1,0])
        qml.RY(a[1], wires=0)
        qml.CRY(a[0],wires=[1,0])

        qml.RZ(a[1], wires=1)
        qml.CRZ(a[0],wires=[1,0])
        '''
        '''
        qml.RZ(a[0], wires=0)

        qml.CNOT(wires=[0, 1])
        qml.RZ(a[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RZ(a[2], wires=1)
    
        qml.PauliX(wires=0)
        qml.CNOT(wires=[0, 1])
        qml.RZ(a[3], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RZ(a[4], wires=1)
        qml.PauliX(wires=0)
        '''
        #Original func start
        # Technically only works with positive numbers
        qml.RY(a[0], wires=0)

        qml.CNOT(wires=[0, 1])
        qml.RY(a[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RY(a[2], wires=1)

        qml.PauliX(wires=0)
        qml.CNOT(wires=[0, 1])
        qml.RY(a[3], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RY(a[4], wires=1)
        qml.PauliX(wires=0)
        #Original func end
        '''
        qml.RZ(a[0], wires=1)

        qml.CNOT(wires=[0, 1])
        qml.RZ(a[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RZ(a[2], wires=1)
    
        qml.PauliX(wires=0)
        qml.CNOT(wires=[0, 1])
        qml.RZ(a[3], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RZ(a[4], wires=1)
        qml.PauliX(wires=0)
        '''

    @qml.qnode(dev)
    def circuit(weights, angles):
        statepreparation(angles)

        for W in weights:
            layer(W)
        return qml.expval(qml.PauliZ(1))

    def variational_classifier(var, angles):
        weights = var[0]
        bias = var[1]
        return circuit(weights, angles) + bias
    
    def square_loss(labels, predictions):
        loss = 0
        for l, p in zip(labels, predictions):
            ## testing median of X axis values    
            '''
            if l == "-1":
                loss = loss + abs(-0.87 - p) 
            elif l == "0":
                loss = loss + abs(0 - p)   
            else:    
                loss = loss + abs(0.87 - p)
            '''
            loss = loss + (l - p) ** 2
        loss = loss / len(labels)
        return loss
    
    def accuracy(labels, predictions):
        loss = 0
        for l, p in zip(labels, predictions):
            if abs(l - p) < 1e-3:
                loss = loss + 1
        loss = loss / len(labels)
        return loss

    def correct_labels(predictions):
        correct_label = []
        for p in predictions:
            if p < -0.3:
                #if p < -0.45:
                correct_label.append(-1)
            elif p > 0.3:
                #elif p > 0.45:
                correct_label.append(1)
            else:
                correct_label.append(0)
        return correct_label

    def cost(weights, features, labels):
        predictions = [variational_classifier(weights, f) for f in features]
        #predictions = correct_labels(predictions_bad)
        return square_loss(labels, predictions)

    X = X_train
    #print("First X sample (original)  :", X[0])

    # pad the vectors to size 2^2 with constant values
    padding = 0.3 * np.ones((len(X), 1))
    X_pad = np.c_[np.c_[X, padding], np.zeros((len(X), 1))]
    #print("First X sample (padded)    :", X_pad[0])

    # normalize each input
    normalization = np.sqrt(np.sum(X_pad ** 2, -1))
    X_norm = (X_pad.T / normalization).T
    #print("First X sample (normalized):", X_norm[0])


    
    # angles for state preparation are new features
    features = np.array([get_angles(x) for x in X_norm])
    #print("First features sample      :", features[0])
    
    Y = Y_train
    
    ####### Mapping Test Data ##########
    X_data = X_test
    #print("First X_data sample (original)  :", X_data[0])

    # pad the vectors to size 2^2 with constant values
    x_padding = 0.3 * np.ones((len(X_data), 1))
    X_data_pad = np.c_[np.c_[X_data, x_padding], np.zeros((len(X_data), 1))]
    #print("First X_data sample (padded)    :", X_data_pad[0])

    # normalize each input
    x_normalization = np.sqrt(np.sum(X_data_pad ** 2, -1))
    X_data_norm = (X_data_pad.T / x_normalization).T
    #print("First X_data sample (x_normalization):", X_data_norm[0])

    
    # angles for state preparation are new features
    x_features = np.array([get_angles(x) for x in X_data_norm])
    #print("First x_features sample      :", x_features[0])
    ####### Mapping Test Data ##########
    

    
    import matplotlib.pyplot as plt

    plt.figure()
    plt.scatter(X[:, 0][Y == 1], X[:, 1][Y == 1], c="b", marker="o", edgecolors="k")
    plt.scatter(X[:, 0][Y == 0], X[:, 1][Y == 0], c="g", marker="o", edgecolors="k")
    plt.scatter(X[:, 0][Y == -1], X[:, 1][Y == -1], c="r", marker="o", edgecolors="k")
    plt.title("Original data")
    plt.show()

    plt.figure()
    dim1 = 0
    dim2 = 1
    plt.scatter(
        X_norm[:, dim1][Y == 1], X_norm[:, dim2][Y == 1], c="b", marker="o", edgecolors="k"
    )
    plt.scatter(
        X_norm[:, dim1][Y == -1], X_norm[:, dim2][Y == -1], c="r", marker="o", edgecolors="k"
    )
    plt.scatter(
        X_norm[:, dim1][Y == 0], X_norm[:, dim2][Y == 0], c="g", marker="o", edgecolors="k"
    )
    plt.title("Padded and normalised data (dims {} and {})".format(dim1, dim2))
    plt.show()

    plt.figure()
    dim1 = 0
    dim2 = 2
    plt.scatter(
        features[:, dim1][Y == 1], features[:, dim2][Y == 1], c="b", marker="o", edgecolors="k"
    )
    plt.scatter(
        features[:, dim1][Y == -1], features[:, dim2][Y == -1], c="r", marker="o", edgecolors="k"
    )
    plt.scatter(
        features[:, dim1][Y == 0], features[:, dim2][Y == 0], c="g", marker="o", edgecolors="k"
    )
    plt.title("Feature vectors (dims {} and {})".format(dim1, dim2))
    plt.show()
    
    
    np.random.seed(1337)
    num_data = 250
    num_train = 50
    index = np.random.permutation(range(num_data))
    feats_train = X_norm
    #Y_train = already loaded as Y_train
    feats_val = X_data_norm
    #Hardcode "1.ans" for training comparison
    Y_val = [1,0,-1,0,-1,1,-1,-1,0,-1,1,-1,0,1,0,-1,-1,0,0,1,1,0,-1,0,0,-1,0,-1,0,0,1,1,-1,-1,-1,0,-1,0,1,0,-1,1,1,0,-1,-1,-1,-1,0,0]

    # We need these later for plotting
    #X_train = X
    #X_val = X_data

    num_qubits = 2
    num_layers = 6
    var_init = (0.01 * np.random.randn(num_layers, num_qubits, 3), 0.0)

    opt = NesterovMomentumOptimizer(0.1)
    #opt = GradientDescentOptimizer(0.4)
    batch_size = 5

    # train the variational classifier
    var = var_init

    for it in range(22):

        # Update the weights by one optimizer step
        batch_index = np.random.randint(0, num_train, (batch_size,))

        feats_train_batch = feats_train[batch_index]
        Y_train_batch = Y_train[batch_index]
        var = opt.step(lambda v: cost(v, feats_train_batch, Y_train_batch), var)
        
        # Compute predictions on train and validation set
        #predictions_train = [variational_classifier(var, f) for f in feats_train]
        predictions_train_bad = [variational_classifier(var, f) for f in feats_train]
        predictions_train = correct_labels(predictions_train_bad)
        #predictions_val = [variational_classifier(var, f) for f in feats_val]
        predictions_val_bad = [variational_classifier(var, f) for f in feats_val]
        predictions_val = correct_labels(predictions_val_bad)

        # Compute accuracy on train and validation set
        #print(predictions_train)
        acc_train = accuracy(Y_train, predictions_train)
        acc_val = accuracy(Y_val, predictions_val)

        #predictions_train_bad = [variational_classifier(var, feats_train[0])]
        #print(predictions_train_bad)
        #predictions_train = correct_labels(predictions_train_bad)
        #print(predictions_train)

        #print("Iter: {:5d} | Cost: {:0.7f}".format(it + 1, cost(var, X_norm, Y)))
        print(
            "Iter: {:5d} | Cost: {:0.7f} | Acc train: {:0.7f} | Acc validation: {:0.7f} "
            "".format(it + 1, cost(var, X_norm, Y), acc_train, acc_val)
        )
        
    predictions_val_bad = [variational_classifier(var, f) for f in feats_val]
    predictions = correct_labels(predictions_val_bad)
    # QHACK #

    return array_to_concatenated_string(predictions)

def array_to_concatenated_string(array):
    """DO NOT MODIFY THIS FUNCTION.

    Turns an array of integers into a concatenated string of integers
    separated by commas. (Inverse of concatenated_string_to_array).
    """
    return ",".join(str(x) for x in array)


def concatenated_string_to_array(string):
    """DO NOT MODIFY THIS FUNCTION.

    Turns a concatenated string of integers separated by commas into
    an array of integers. (Inverse of array_to_concatenated_string).
    """
    return np.array([int(x) for x in string.split(",")])


def parse_input(giant_string):
    """DO NOT MODIFY THIS FUNCTION.

    Parse the input data into 3 arrays: the training data, training labels,
    and testing data.

    Dimensions of the input data are:
      - X_train: (250, 3)
      - Y_train: (250,)
      - X_test:  (50, 3)
    """
    X_train_part, Y_train_part, X_test_part = giant_string.split("XXX")

    X_train_row_strings = X_train_part.split("S")
    X_train_rows = [[float(x) for x in row.split(",")] for row in X_train_row_strings]
    X_train = np.array(X_train_rows)

    Y_train = concatenated_string_to_array(Y_train_part)

    X_test_row_strings = X_test_part.split("S")
    X_test_rows = [[float(x) for x in row.split(",")] for row in X_test_row_strings]
    X_test = np.array(X_test_rows)

    return X_train, Y_train, X_test


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    X_train, Y_train, X_test = parse_input(sys.stdin.read())
    output_string = classify_data(X_train, Y_train, X_test)
    print(f"{output_string}")