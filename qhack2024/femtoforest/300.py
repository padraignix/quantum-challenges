import json
import pennylane as qml
import pennylane.numpy as np

def beam_splitter(r):
    """
    Returns the beam splitter matrix.

    Args:
        - r (float): The reflection coefficient of the beam splitter.
    Returns:
        - (np.array): 2 x 2 matrix that represents the beam
        splitter matrix.    
    """

    # Put your code here
    t = np.sqrt(1 - r ** 2)
    return np.array([[r, t], [t, -r]])

dev = qml.device('default.qubit')

@qml.qnode(dev)
def mz_interferometer(r):
    """
    This QNode returns the probability that either A or C
    detect a photon, and the probability that D detects a photon.
    
    Args:
        - r (float): The reflection coefficient of the beam splitters.
    Returns: 
        - np.array(float): An array of shape (2,), where the first 
        element is the probability of detection at A or C,
        and the second element is the probability of detection at D.
    """


    # Put your code here
    
    ## Get unitary for beam splitter
    U = beam_splitter(r)

    ## Def apply first beam splitter
    def apply_beam_split(u):
        qml.QubitUnitary(u, wires=[0])

    ## Apply first beam splitter
    apply_beam_split(U)

    ## measure output for 0 to see if terminated
    m_0 = qml.measure(wires=[0])
    qml.cond(m_0 == 1,apply_beam_split)(U)

    return qml.probs([0])


# These functions are responsible for testing the solution.


def run(test_case_input: str) -> str:
    ins = json.loads(test_case_input)
    outs = mz_interferometer(ins).tolist()
    
    return str(outs)


def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    assert np.allclose(solution_output,expected_output), "Not the correct probabilities"


# These are the public test cases
test_cases = [
    ('0.5', '[0.8125, 0.1875]'),
    ('0.577350269', '[0.777778, 0.222222]')
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