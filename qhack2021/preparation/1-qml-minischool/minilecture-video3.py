from qiskit.ml.datasets import *
from qiskit import QuantumCircuit
from qiskit.aqua.components.optimizers import COBYLA
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
import numpy as np
import matplotlib.pyplot as plt
from qiskit.quantum_info import Statevector

#%matplotlib inline

###
###     1 Variational quantum classifier 
###

# size of training data set
training_size = 100
# size of test data set
test_size = 20
# dimension of data sets
n = 2
# construct training and test data

_, training_input, test_input, class_labels = ad_hoc_data(training_size=training_size, test_size=test_size, n=n, gap=0.3, plot_data=False)
#print(class_labels)

sv = Statevector.from_label('0' * n)
feature_map = ZZFeatureMap(n, reps=1)
var_form = RealAmplitudes(n, reps=1)
circuit = feature_map.combine(var_form)
#circuit.draw(output="mpl")
#print(circuit)


def get_data_dict(params, x):
    parameters = {}
    for i, p in enumerate(feature_map.ordered_parameters):
        parameters[p] = x[i]
    for i, p in enumerate(var_form.ordered_parameters):
        parameters[p] = params[i]
    return parameters

data = [0.1, 1.2]
params = np.array([0.1, 1.2, 0.02, 0.1])
circ_ = circuit.assign_parameters(get_data_dict(params, data))
#circ_.draw(plot_barriers=True)
#print(circ_)


def assign_label(bit_string, class_labels):
    hamming_weight = sum([int(k) for k in list(bit_string)])
    is_odd_parity = hamming_weight & 1
    if is_odd_parity:
        return class_labels[1]
    else:
        return class_labels[0]

def return_probabilities(counts, class_labels):
    shots = sum(counts.values())
    result = {class_labels[0]: 0, class_labels[1]: 0}
    for key, item in counts.items():
        label = assign_label(key, class_labels)
        result[label] += counts[key]/shots
    return result

#print (return_probabilities({'00' : 10, '01': 10, '11': 20}, class_labels))

def classify(x_list, params, class_labels):
    qc_list = []
    for x in x_list:
        circ_ = circuit.assign_parameters(get_data_dict(params, x))
        qc = sv.evolve(circ_)
        qc_list += [qc]
    probs = []
    for qc in qc_list:
        counts = qc.to_counts()
        prob = return_probabilities(counts, class_labels)
        probs += [prob]
    return probs

# classify a test data point
x = np.asarray([[0.5, 0.9]])
#print(classify(x, params=np.array([0.8, -0.5, 1.5, 0,5]), class_labels=class_labels))

def cost_estimate_sigmoid(probs, expected_label): # probability of labels vs actual labels
    p = probs.get(expected_label)
    sig = None
    if np.isclose(p, 0.0):
        sig = 1
    elif np.isclose(p, 1.0):
        sig = 0
    else:
        denominator = np.sqrt(2*p*(1-p))
        x = np.sqrt(200)*(0.5-p)/denominator
        sig = 1/(1+np.exp(-x))
    return sig

'''
x = np.linspace(0, 1, 20)
y = [cost_estimate_sigmoid({'A': x_, 'B': 1-x_}, 'A') for x_ in x]
plt.plot(x, y)
plt.xlabel('Probability of assigning the correct class')
plt.ylabel('Cost value')
plt.show()
'''

def cost_function(training_input, class_labels, params, shots=100, print_value=False):
    # map training input to list of labels and list of samples
    cost = 0
    training_labels = []
    training_samples = []
    for label, samples in training_input.items():
        for sample in samples:
            training_labels += [label]
            training_samples += [sample]
    # classify all samples
    probs = classify(training_samples, params, class_labels)
    
    # evaluate costs for all classified samples
    for i, prob in enumerate(probs):
        cost += cost_estimate_sigmoid(prob, training_labels[i])
    cost /= len(training_samples)
    
    # print resulting objective function
    if print_value:
        print('%.4f' % cost)
    
    # return objective value
    return cost

#print(cost_function(training_input, class_labels, params))

####
####    1.1 Train the classifier
####

# setup the optimizer
optimizer = COBYLA(maxiter=100)
# define objective function for training
objective_function = lambda params: cost_function(training_input, class_labels, params, print_value=True)
# randomly initialize the parameters
np.random.seed(137)
init_params = 2*np.pi*np.random.rand(n*(1)*2)
# train classifier
opt_params, value, _ = optimizer.optimize(len(init_params), objective_function, initial_point=init_params)
# print results
#print()
#print('opt_params:', opt_params)
#print('opt_value: ', value)


####
####    1.2 Train the classifier
####

# collect coordinates of test data
test_label_0_x = [x[0] for x in test_input[class_labels[0]]]
test_label_0_y = [x[1] for x in test_input[class_labels[0]]]
test_label_1_x = [x[0] for x in test_input[class_labels[1]]]
test_label_1_y = [x[1] for x in test_input[class_labels[1]]]
# initialize lists for misclassified datapoints
test_label_misclassified_x = []
test_label_misclassified_y = []

# evaluate test data
for label, samples in test_input.items():
    # classify samples
    results = classify(samples, opt_params, class_labels)
    # analyze results
    for i, result in enumerate(results):
        # assign label
        assigned_label = class_labels[np.argmax([p for p in result.values()])]
        print('----------------------------------------------------')
        print('Data point: ', samples[i])
        print('Label: ', label)
        print('Assigned: ', assigned_label)
        print('Probabilities: ', result)
        if label != assigned_label:
            print('Classification:', 'INCORRECT')
            test_label_misclassified_x += [samples[i][0]]
            test_label_misclassified_y += [samples[i][1]]
        else:
            print('Classification:', 'CORRECT')
# compute fraction of misclassified samples
total = len(test_label_0_x) + len(test_label_1_x)
num_misclassified = len(test_label_misclassified_x)
print()
print(100*(1-num_misclassified/total), "% of the test data was correctly classified!")
# plot results
plt.figure()
plt.scatter(test_label_0_x, test_label_0_y, c='b', label=class_labels[0], linewidths=5)
plt.scatter(test_label_1_x, test_label_1_y, c='g', label=class_labels[1], linewidths=5)
plt.scatter(test_label_misclassified_x, test_label_misclassified_y, linewidths=20, s=1, facecolors='none',
edgecolors='r')
plt.legend()
plt.show()
