import json
import pennylane as qml
import pennylane.numpy as np
import scipy

dev = qml.device("default.qubit", wires=["player1", "player2", "goalie"])


# Put any extra functions you want here
'''
def return_probs(modifier, xyz):
    value = qml.probs(wires="goalie")
    print(value.tolist())
    if xyz == 'x' or xyz == 'y':
        return [modifier*value[0], (1/modifier)*value[1]]
    if xyz == 'z':
        return [modifier*value[0], (1/modifier)*value[1]]
'''
    

def state_prep(player_coeffs, goalie_coeffs):
    """
    Contains quantum operations that prepare |psi> and |phi>. We recommend using
    qml.StatePrep!

    Args:
        - player_coeffs (list(float)): 
            The coefficients, alpha, beta, and kappa (in that order) that describe 
            the quantum state of players 1 and 2:

            |psi> = alpha|01> + beta|10> + kappa(|00> + |11>)

        - goalie_coeffs (list(float)):
            The coefficients, gamma and delta (in that order) that describe 
            the quantum state of the goalie:

            |phi> = gamma|0> + delta|1>
    """


    alpha, beta, kappa = player_coeffs
    gamma, delta = goalie_coeffs

    # Put your code here #
    ## State preparation for 00,01,10,11 - needing double kappa
    qml.StatePrep([kappa, alpha, beta, kappa], wires=["player1", "player2"])
    ## State preparation for goalie
    #qml.StatePrep([gamma, delta], wires=["goalie"])

@qml.qnode(dev)
def save_percentage(player_coeffs, goalie_coeffs, x, y, z):
    """
    Calculates the save percentage of the goalie.

    NOTE: This QNode may only contain 7 operations or less (counting any 
    operations used in the state_prep function) and must use three conditional
    measurements (i.e., 3 instances of qml.cond).

    Args:
        - player_coeffs (list(float)): 
            The coefficients, alpha, beta, and kappa (in that order) that describe 
            the quantum state of players 1 and 2:

            |psi> = alpha|01> + beta|10> + kappa(|00> + |11>)

        - goalie_coeffs (list(float)):
            The coefficients, gamma and delta (in that order) that describe 
            the quantum state of the goalie:

            |phi> = gamma|0> + delta|1>
        
        - x, y, z (float): 
            The amounts that affect the goalie's save percentage based on 
            measuring the players.

    Returns:
        - (numpy.tensor): The save percentage of the goalie.
    """
    state_prep(player_coeffs, goalie_coeffs)
   
    # Put your code here #

    # measure player 1
    m_0 = qml.measure(wires=["player1"])
    m_1 = qml.measure(wires=["player2"])

    # if player1 == 1, adjust goalie state
    qml.cond(m_0 & ~m_1, qml.RY)(x, wires=["goalie"])
    # if player2 == 1, adjust goalie state
    qml.cond(~m_0 & m_1, qml.RY)(y, wires=["goalie"])
    # if player1 == 0 and player2 == 0, or both 1, adjust goalie state
    qml.cond((m_0 & m_1) | (~m_0 & ~m_1), qml.RY)(-z, wires=["goalie"])

    return qml.probs(wires="goalie")

# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    ins = json.loads(test_case_input)
    player_coeffs, goalie_coeffs, x, y, z = ins
    output = save_percentage(player_coeffs, goalie_coeffs, x, y, z).tolist()
    return str(output)

def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    sp = solution_output
    _sp = json.loads(expected_output)
    print(f'calculated: {sp}')
    print(f'expected: {_sp}')

    ops = save_percentage.tape._ops
    num_ops = len(ops)
    num_cond = [op.name for op in ops].count('Conditional')
    names = [op.name for op in ops]
    state_prep_check = ('StatePrep' or 'MottonenStatePreparation' or 'AmplitudeEmbedding') in names

    assert np.allclose(sp, _sp, rtol=1e-4), "Your calculated save percentage is incorrect."
    assert num_ops < 8, "You used more than 7 operations in your save_percentage function."
    assert num_ops > 2, "You definitely need more than 2 operations..."
    assert state_prep_check, "You can use StatePrep, MottonenStatePreparation, or AmplitudeEmbedding to prepare states."
    assert num_cond == 3, "You haven't used exactly 3 qml.cond operators."


# These are the public test cases
test_cases = [
    ('[[0.74199663, 0.17932039, 0.45677413], [0.28034464, 0.95989941], 0.999, 0.99, 0.98]', '[0.08584767923415959, 0.9141523336414634]'),
    ('[[0.09737041, 0.40230525, 0.64368839], [0.00111111, 0.99999938], 0.9, 0.95, 0.92]', '[0.06629469110239884, 0.9337053066603161]')
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