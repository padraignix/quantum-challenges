import json
import pennylane as qml
import pennylane.numpy as np

WIRES = 2
LAYERS = 5
NUM_PARAMETERS = LAYERS * WIRES * 3

initial_params = np.random.random(NUM_PARAMETERS)

def variational_circuit(params,hamiltonian):
    """
    This is a template variational quantum circuit containing a fixed layout of gates with variable
    parameters. To be used as a QNode, it must either be wrapped with the @qml.qnode decorator or
    converted using the qml.QNode function.

    The output of this circuit is the expectation value of a Hamiltonian, somehow encoded in
    the hamiltonian argument

    Args:
        - params (np.ndarray): An array of optimizable parameters of shape (30,)
        - hamiltonian (np.ndarray): An array of real parameters encoding the Hamiltonian
        whose expectation value is returned.
    
    Returns:
        (float): The expectation value of the Hamiltonian
    """
    parameters = params.reshape((LAYERS, WIRES, 3))
    qml.templates.StronglyEntanglingLayers(parameters, wires=range(WIRES))
    return qml.expval(qml.Hermitian(hamiltonian, wires = [0,1]))

def optimize_circuit(params,hamiltonian):
    """Minimize the variational circuit and return its minimum value.
    You should create a device and convert the variational_circuit function 
    into an executable QNode. 
    Next, you should minimize the variational circuit using gradient-based 
    optimization to update the input params. 
    Return the optimized value of the QNode as a single floating-point number.

    Args:
        - params (np.ndarray): Input parameters to be optimized, of dimension 30
        - hamiltonian (np.ndarray): An array of real parameters encoding the Hamiltonian
        whose expectation value you should minimize.
    Returns:
        float: the value of the optimized QNode
    """


    dev = qml.device('default.qubit', wires = 2) # Initialize the device.
    circuit = qml.QNode(variational_circuit, dev) # Instantiate the QNode from variational_circuit.
    # Write your code to minimize the circuit
    ## https://pennylane.ai/qml/demos/tutorial_vqls/
    opt = qml.GradientDescentOptimizer(0.8)
    #cost_history = []
    cost_r = 0
    steps = 50

    #initial_params.require_grad = True
    #params = np.ones(NUM_PARAMETERS, requires_grad = True) * initial_params
    params = initial_params
    print(qml.math.requires_grad(params))

    for it in range(steps):
        print("Step {:3d}".format(it))
        circuit = qml.QNode(variational_circuit, dev)
        print(f'before: {params[0]}')
        params, cost = opt.step_and_cost(circuit, params, hamiltonian)
        print("Step {:3d} Cost = {:9.7f}".format(it, cost))
        print(f'after: {params[0]}')
        #cost_history.append(cost)
        params = np.array(params[0])
        cost_r = cost
    return cost_r # Return the value of the minimized QNode


# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
       
    ins = np.array(json.loads(test_case_input), requires_grad = False)
    hamiltonian = np.array(ins,float).reshape((2 ** WIRES), (2 ** WIRES))
    np.random.seed(1967)
    initial_params = np.random.random(NUM_PARAMETERS)
    out = str(optimize_circuit(initial_params,hamiltonian))    
    
    return out


def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    assert np.isclose(solution_output, expected_output, rtol=5e-2)


# These are the public test cases
test_cases = [
    ('[0.863327072347624,0.0167108057202516,0.07991447085492759,0.0854049026262154,0.0167108057202516,0.8237963773906136,-0.07695947154193797,0.03131548733285282,0.07991447085492759,-0.07695947154193795,0.8355417021014687,-0.11345916130631205,0.08540490262621539,0.03131548733285283,-0.11345916130631205,0.758156886827099]', '0.61745341'),
    ('[0.32158897156285354,-0.20689268438270836,0.12366748295758379,-0.11737425017261123,-0.20689268438270836,0.7747346055276305,-0.05159966365446514,0.08215539696259792,0.12366748295758379,-0.05159966365446514,0.5769050487087416,0.3853362904758938,-0.11737425017261123,0.08215539696259792,0.3853362904758938,0.3986256655167206]', '0.00246488')
]

# This will run the public test cases locally
for i, (input_, expected_output) in enumerate(test_cases):
    print(f"Running test case {i} with input '{input_}'...")

    try:
        output = run(input_)

    except Exception as exc:
        print(f"Runtime Error. {exc}")

    else:
        if message := check(output, expected_output):
            print(f"Wrong Answer. Have: '{output}'. Want: '{expected_output}'.")

        else:
            print("Correct!")