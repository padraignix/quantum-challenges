import json
import pennylane as qml
import pennylane.numpy as np

dev = qml.device('default.qutrit', wires = 1)

@qml.qnode(dev)
def prepare_qutrit(chi, eta):
    """
    This QNode prepares the state |phi> as defined in the statement and
    computes the measurement probabilities in the qutrit computational basis.
    
    Args: 
        - chi (float): The angle chi parametrizing the state |phi>.
        - eta (float): The angle eta parametrizing the state |eta>.
    Returns:
        - (np.array(float)): The measurement probabilities in the computational
        basis after preparing the state.
    
    """


    # Put your code here #


    return qml.probs(wires=0)


def evaluate_sum(chi, eta_array):
    """
    This QNode computes the sum S as in the statement.
    
    Args: 
        chi (float): The angle chi parametrizing the states |phi_i>.
        eta_array (float): Contains the angles eta_i parametrizing the state |eta_i>.
    Returns:
        (np.array(float)): The sum S as defined in the statement.
        
    """


    # Put your code here


# These functions are responsible for testing the solution.


def run(test_case_input: str) -> str:
    ins = json.loads(test_case_input)
    outs = evaluate_sum(*ins)
    
    return str(outs)


def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    assert np.allclose(solution_output,expected_output, atol = 1e-4), "Not the correct sum!"


# These are the public test cases
test_cases = [
    ('[0.838283, [0.6283189, 1.884956, 3.141593, 4.398230, 5.654867]]', '2.236069'),
    ('[0.4, [1.047198, 2.094395, 3.141593, 4.18879, 5.235988]]', '4.241767')
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