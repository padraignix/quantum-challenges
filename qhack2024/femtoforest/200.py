import json
import pennylane as qml
import pennylane.numpy as np

def U():
    """
    Creates the gate that checks the parity of the number of forests.
    It should not return anything, you simply need to add the gates.
    """


    # Put your code here #
    qml.CNOT(wires=[0,8])

    qml.Toffoli(wires=[0,1,8])
    qml.CNOT(wires=[1,8])

    qml.Toffoli(wires=[1,2,8])
    qml.CNOT(wires=[2,8])

    qml.Toffoli(wires=[2,3,8])
    qml.CNOT(wires=[3,8])

    qml.Toffoli(wires=[3,4,8])
    qml.CNOT(wires=[4,8])

    qml.Toffoli(wires=[4,5,8])
    qml.CNOT(wires=[5,8]) 

    qml.Toffoli(wires=[5,6,8])
    qml.CNOT(wires=[6,8]) 

    qml.Toffoli(wires=[6,7,8])
    qml.CNOT(wires=[7,8]) 


# These functions are responsible for testing the solution.


def run(test_case_input: str) -> str:

    input = json.loads(test_case_input)
    wires_input = [0,1,2,3,4,5,6,7]

    dev = qml.device("default.qubit", wires = 10, shots = 10)

    @qml.qnode(dev)
    def circuit():
      qml.BasisEmbedding(input, wires = wires_input)

      U()

      return qml.probs(wires = 8)

    return str(float(circuit()[1]))


def check(have: str, want: str) -> None:

    assert np.isclose(float(have), float(want)), "Wrong answer!"


# These are the public test cases
test_cases = [
    ('[1,0,1,1,0,1,1,1]', '1'),
    ('[0,0,0,0,0,1,0,1]', '0')
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