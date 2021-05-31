#! /usr/bin/python3
import json
import sys
import networkx as nx
import numpy as np
import pennylane as qml


# DO NOT MODIFY any of these parameters
NODES = 6
N_LAYERS = 10


def find_max_independent_set(graph, params):
    """Find the maximum independent set of an input graph given some optimized QAOA parameters.
    The code you write for this challenge should be completely contained within this function
    between the # QHACK # comment markers. You should create a device, set up the QAOA ansatz circuit
    and measure the probabilities of that circuit using the given optimized parameters. Your next
    step will be to analyze the probabilities and determine the maximum independent set of the
    graph. Return the maximum independent set as an ordered list of nodes.
    Args:
        graph (nx.Graph): A NetworkX graph
        params (np.ndarray): Optimized QAOA parameters of shape (2, 10)
    Returns:
        list[int]: the maximum independent set, specified as a list of nodes in ascending order
    """

    max_ind_set = []

    # QHACK #
    from pennylane import qaoa
    wires = range(6)
    cost_h, mixer_h = qaoa.max_independent_set(graph, constrained=True)
    #print("Cost Hamiltonian", cost_h)
    #print("Mixer Hamiltonian", mixer_h)
    
    def qaoa_layer(gamma, alpha):
        qaoa.cost_layer(gamma, cost_h)
        qaoa.mixer_layer(alpha, mixer_h)

    def circuit1(params, **kwargs):
        qml.layer(qaoa_layer, N_LAYERS, params[0], params[1])

    def probability_circuit(params):
        circuit1(params)
        return qml.probs(wires=wires)

    dev = qml.device("default.qubit", wires=6)
    circuit = qml.QNode(probability_circuit, dev)
    probs = circuit(params)
    solution = np.max(probs)
    result = np.where(probs == np.amax(probs))
   
    test = result[0].item()
    def get_bin(x, n=0):
        return format(x, 'b').zfill(n)

    test_format = get_bin(test,6)
    string_ans = test_format

    i = 0
    for char in string_ans:
        if char == "1":
            max_ind_set.append(i)
        i+=1
    
    # QHACK #

    return max_ind_set


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Load and process input
    graph_string = sys.stdin.read()
    graph_data = json.loads(graph_string)

    params = np.array(graph_data.pop("params"))
    graph = nx.json_graph.adjacency_graph(graph_data)

    max_independent_set = find_max_independent_set(graph, params)

    print(max_independent_set)