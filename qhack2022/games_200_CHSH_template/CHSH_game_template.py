#! /usr/bin/python3

from ftplib import parse150
import sys
import pennylane as qml
from pennylane import numpy as np


dev = qml.device("default.qubit", wires=2)
#dev = qml.device("default.qubit", wires=2, shots=1)

def prepare_entangled(alpha, beta):
    """Construct a circuit that prepares the (not necessarily maximally) entangled state in terms of alpha and beta
    Do not forget to normalize.

    Args:
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>
    """

    # QHACK #
        ## instead of bell state max entangle of H + CNOT, have to rotate 
        ## based on alpha/beta
    #norm = preprocessing.normalize(np.array([alpha,beta]))
    alpha_n = alpha / (alpha+beta)
    beta_n = beta / (alpha+beta)
    qml.QubitStateVector(np.array([np.sqrt(alpha_n),0,0,np.sqrt(beta_n)]), wires=range(2))
    # QHACK #

@qml.qnode(dev)
def chsh_circuit(theta_A0, theta_A1, theta_B0, theta_B1, x, y, alpha, beta):

    """Construct a circuit that implements Alice's and Bob's measurements in the rotated bases

    Args:
        - theta_A0 (float): angle that Alice chooses when she receives x=0
        - theta_A1 (float): angle that Alice chooses when she receives x=1
        - theta_B0 (float): angle that Bob chooses when he receives x=0
        - theta_B1 (float): angle that Bob chooses when he receives x=1
        - x (int): bit received by Alice
        - y (int): bit received by Bob
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>

    Returns:
        - (np.tensor): Probabilities of each basis state
    """

    prepare_entangled(alpha, beta)

    # QHACK #
    if x == 0:
        qml.RY(theta_A0,wires=0)
    elif x == 1:
        qml.RY(theta_A1,wires=0)

    if y == 0:
        qml.RY(theta_B0,wires=1)
    elif y == 1:
        qml.RY(theta_B1,wires=1)

    # QHACK #

    return qml.probs(wires=[0, 1])
    #return qml.sample(wires=[0,1])

def winning_prob(params, alpha, beta):
    """Define a function that returns the probability of Alice and Bob winning the game.

    Args:
        - params (list(float)): List containing [theta_A0,theta_A1,theta_B0,theta_B1]
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>

    Returns:
        - (float): Probability of winning the game
    """

    
    # QHACK #
    
    x_r = np.random.random()
    y_r = np.random.random()

    if x_r >= 1/2:
        x = 1
    else:
        x = 0
    if y_r >= 1/2:
        y = 1
    else:
        y = 0

    probability_raw = chsh_circuit(params[0], params[1], params[2], params[3], x, y, alpha, beta)
    print(probability_raw)
    if x == 0 and y == 0:
        return probability_raw[0]
    if x == 1 and y == 0:
        return probability_raw[1]
    if x == 0 and y == 1:
        return probability_raw[2]
    if x == 1 and y == 1:
        return probability_raw[3]
    
    '''
    p0 = np.cos(params[0] - params[2])**2
    p1 = np.cos(params[0] - params[3])**2
    p2 = np.cos(params[1] - params[2])**2
    p3 = np.sin(params[1] - params[3])**2

    return p0/4 + p1/4 + p2/4 + p3/4
    '''
    #print("Win Perc: ", winning/100)
    #return winning/1000
    # QHACK #
    

def optimize(alpha, beta):
    """Define a function that optimizes theta_A0, theta_A1, theta_B0, theta_B1 to maximize the probability of winning the game

    Args:
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>

    Returns:
        - (float): Probability of winning
    """

    def cost(params):
        """Define a cost function that only depends on params, given alpha and beta fixed"""

    # QHACK #
        value = winning_prob(params, alpha, beta)
        r_value = 1 - value
        print(r_value)
        return r_value
    ### https://en.wikipedia.org/wiki/CHSH_game

    #Initialize parameters, choose an optimization method and number of steps
    #init_params = np.array([0,0,0,0],requires_grad=True)
    init_params = np.array([0, np.pi/4, np.pi/8, -np.pi/8], requires_grad=True)
    params = init_params
    '''
    opt = qml.GradientDescentOptimizer(0.01)
    steps = 200

    # QHACK #
    
    # set the initial parameter values
    params = init_params

    for i in range(steps):
        # update the circuit parameters 
        # QHACK #
        #params = opt.step(cost, params)
        params = np.clip(opt.step(cost, params), -2 * np.pi, 2 * np.pi)
        print(params)
        # QHACK #
    '''
    '''
    opt = qml.GradientDescentOptimizer(0.1)
    steps = 200
    for i in range(steps):
        # update the circuit parameters 
        # QHACK #
        #params = opt.step(cost, params)
        params = np.clip(opt.step(cost, params), -2 * np.pi, 2 * np.pi)
        print(params)
        # QHACK #
    '''
    return winning_prob(params, alpha, beta)


if __name__ == '__main__':
    inputs = sys.stdin.read().split(",")
    output = optimize(float(inputs[0]), float(inputs[1]))
    print(f"{output}")