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
    num_qubits = 3
    n = 3
    from sklearn.preprocessing import normalize
    from pennylane.templates.embeddings import AngleEmbedding
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
    from qiskit.quantum_info import Statevector
    #data = normalize(X_train)
    #print(data)

    #_, training_input, test_input, class_labels = ad_hoc_data(training_size=training_size, test_size=test_size, n=n, gap=0.3, plot_data=False)
    training_input = X_train
    test_input = X_test
    class_labels = ['-1', '0', '1']
    
    #Y_train

    sv = Statevector.from_label('0' * n)
    feature_map = ZZFeatureMap(n, reps=1)
    var_form = RealAmplitudes(n, reps=1)
    circuit = feature_map.combine(var_form)
    
    #print(circuit)
    
    def get_data_dict(params, x):
        parameters = {}
        for i, p in enumerate(feature_map.ordered_parameters):
            parameters[p] = x[i]
        for i, p in enumerate(var_form.ordered_parameters):
            parameters[p] = params[i]
        return parameters
    
    def assign_label(bit_string, class_labels):
        hamming_weight = sum([int(k) for k in list(bit_string)])
        #print(bit_string)
        ### this needs to change to check for bit sets
    
        is_odd_parity = hamming_weight & 1
        if is_odd_parity:
            return class_labels[1]
        else:
            return class_labels[0]
    
    def return_probabilities(counts, class_labels):
        shots = sum(counts.values())
        #print(shots)
        result = {class_labels[0]: 0, class_labels[1]: 0, class_labels[2]: 0}
        for key, item in counts.items():
            #print(key)
            label = assign_label(key, class_labels)
            result[label] += counts[key]/shots
        return result

    def classify(x_list, params, class_labels):
        qc_list = []
        for x in x_list:
            circ_ = circuit.assign_parameters(get_data_dict(params, x))
            qc = sv.evolve(circ_)
            qc_list += [qc]
            print (qc_list)
        probs = []
        for qc in qc_list:
            counts = qc.to_counts()
            print(counts)
            prob = return_probabilities(counts, class_labels)
            probs += [prob]
        return probs


    x = np.asarray([training_input[1]])
    params = np.random.normal(0, np.pi, (num_qubits, n, 3))
    test = classify(x, params=np.array([-0.9, -0.4, 0.5, 0,5, 0.8, -0.5, -0.2, 0,5]), class_labels=class_labels)
    print(test)

    '''
    dev = qml.device('default.qubit', wires=num_qubits)

    @qml.qnode(dev)
    def circuit(data):
    # apply Hadamards to all qubits in the circuit
        for i in range(num_qubits):
            qml.Hadamard(wires=i)
        for i in range(len(data)):
            AngleEmbedding(features=data[i], wires=range(num_qubits), rotation='Y')
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2))
    circuit(data)

    print(circuit.draw())
    '''


    # QHACK #

    #return array_to_concatenated_string(predictions)


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