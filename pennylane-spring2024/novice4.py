import json
import pennylane as qml
import pennylane.numpy as np

def compute_hessian(num_wires, w):
    """
    This function must create a circuit with num_wire qubits 
    as per the challenge statement and return the Hessian of such
    circuit evaluated on w.

    Args:
        - num_wires = The number of wires in the circuit
        - w (np.ndarray): A list of length num_wires + 2 containing float parameters. 
        The Hessian is evaluated at this point in parameter space.

    Returns:
        Union(tuple, np.ndarray): A matrix representing the Hessian calculated via 
        qml.gradients.parameter_shift_hessian
    """
    


    # Define your device and QNode
    # efine a QNode which implements the circuit above and returns the expectation value  ⟨Z0⊗Zn−1⟩ ⟨Z0​⊗Zn−1​⟩
    # of the tensor product of Pauli-Z operators on the first and last qubits.

    dev = qml.device('default.qubit', wires = num_wires) # Initialize the device.
   
    @qml.qnode(dev)
    def circuit(w):
        #qml.templates.StronglyEntanglingLayers(w, wires=range(num_wires))
        for i in range(num_wires):
            qml.RX(w[i], wires=i)
        for i in range(num_wires):
            qml.CNOT(wires=[i, (i + 1) % num_wires])
        qml.RY(w[num_wires], wires=1)
        for i in range(num_wires):
            qml.CNOT(wires=[i, (i + 1) % num_wires])       
        qml.RX(w[num_wires + 1], wires=num_wires - 1)

        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(num_wires - 1))

    # Return the Hessian using qml.gradient.param_shift_hessian
    return qml.gradients.param_shift_hessian(circuit)(w)


# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    
    ins = json.loads(test_case_input)
    wires = ins[0]
    weights = np.array(ins[1], requires_grad = True)
    output = compute_hessian(wires, weights)
    
    if isinstance(output,(tuple)):
        output = np.array(output).numpy().round(3)    
        return str([elem.tolist() for elem in output])
    
    elif isinstance(output,(np.tensor)):
        
        return str(output.tolist())
    
def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    
    assert np.allclose(solution_output, expected_output, atol=1e-2), "Your function does not calculate the Hessian correctly."


# These are the public test cases
test_cases = [
    ('[3,[0.1,0.2,0.1,0.2,0.7]]', '[[0.013, 0.0, 0.013, 0.006, 0.002], [0.0, -0.621, 0.077, 0.125, -0.604], [0.013, 0.077, -0.608, -0.628, -0.073], [0.006, 0.125, -0.628, 0.138, -0.044], [0.002, -0.604, -0.073, -0.044, -0.608]]'),
    ('[4,[0.78,0.23,0.54,-0.8,-0.3,0.0]]', '[[0.0, 0.0, 0.0, 0.0, 0.0, 0.128], [0.0, -0.582, 0.082, -0.14, 0.0, -0.343], [0.0, 0.082, -0.582, -0.359, 0.0, -0.057], [0.0, -0.14, -0.359, -0.582, 0.0, 0.204], [0.0, 0.0, 0.0, 0.0, 0.0, 0.393], [0.128, -0.343, -0.057, 0.204, 0.393, -0.582]]')
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