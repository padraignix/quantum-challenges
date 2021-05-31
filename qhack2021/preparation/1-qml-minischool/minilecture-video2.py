
import pennylane as qml
from pennylane import numpy as np

### Basis Encoding
'''
# Basis encoding example with PennyLane
# import the template
from pennylane.templates.embeddings import BasisEmbedding

data = np.array([1,1,1])
dev = qml.device('default.qubit', wires=3)

@qml.qnode(dev)

def circuit(data):
    BasisEmbedding(features=data, wires=range(3))
    return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2))

circuit(data)
'''

from sklearn.datasets import load_iris # import Iris data set
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize # import a normilisation function
np.random.seed(42) # set a seed

x, Y = load_iris().data, load_iris().target
x, Y = shuffle(x,Y)
# take the first 5
x = x[:5]
Y = Y[:5]

#print(x, Y)

data = normalize(x)

### Angle Encoding - Pennylane
'''
print(data)

num_qubits = 4

dev = qml.device('default.qubit', wires=num_qubits)
'''
'''
@qml.qnode(dev)

def circuit(data):
# apply Hadamards to all qubits in the circuit
    for i in range(num_qubits):
        qml.Hadamard(wires=i)
        AngleEmbedding(features=data, wires=range(num_qubits), rotation='Y')
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2)), qml.expval(qml.PauliZ(3))

circuit(data[3])
'''
'''
@qml.qnode(dev)
def circuit(data):
# apply Hadamards to all qubits in the circuit
    for i in range(num_qubits):
        qml.Hadamard(wires=i)
    for i in range(len(data)):
        AngleEmbedding(features=data[i], wires=range(num_qubits), rotation='Y')
    return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2)), qml.expval(qml.PauliZ(3))

circuit(data)
'''

### Angle Encoding - Qiskit
'''
from qiskit import *
from qiskit.circuit.library import ZFeatureMap

featuremap_circ = ZFeatureMap(4, reps=1)

print(featuremap_circ)

# assign data to circuit parameters
circ_data0 = featuremap_circ.assign_parameters(data[0]/2)

# print
print(circ_data0)

# combine the featuremap circuit with assigned parameters of the second data point
circ_data1 = circ_data0.combine(featuremap_circ.assign_parameters(data[1]/2))
print(circ_data1)

circ = QuantumCircuit(4)
for i in range(len(data)):
    circ = circ.combine(featuremap_circ.assign_parameters(data[i]/2))

print(circ)
'''

### Higher Order Encoding - Qiskit
from qiskit.circuit.library import ZZFeatureMap

featuremap_circ = ZZFeatureMap(4, reps=1)
print(featuremap_circ)

