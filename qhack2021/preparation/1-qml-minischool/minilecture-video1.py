# insert imports
import pennylane as qml
from pennylane import numpy as np

# default.qubit, other devices available
dev = qml.device('default.qubit', wires=1)

# wrap device in qml.qnode
# define circuit, RX and RY gates, <Z>
@qml.qnode(dev)

def circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=0)
    return qml.expval(qml.PauliZ(0))

# set some parameters & print the circuit
#params = [0,0]
#val = circuit(params)

#define cost(x)
def cost(x):
    return circuit(x)

#define init_params as np array and check cost
init_params = np.array([0.11, 0.5])
#cost(init_params)

# initialise the optimizer with stepsize
opt = qml.GradientDescentOptimizer(stepsize=0.4)
# set the number of steps
steps = 100
# set the initial parameter values
params = init_params

for i in range(steps):
# update the circuit parameters after each step relative to cost & params
    params = opt.step(cost, params)
    if (i + 1) % 5 == 0:
        print("Cost after step {:5d}: {: .7f}".format(i + 1, cost(params)))
        print("Optimized rotation angles: {}".format(params))