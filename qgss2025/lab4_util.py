import math

import numpy as np
import itertools
from qiskit.quantum_info import Pauli, PauliList
from typing import List, Tuple, Dict, Optional, Set, Union
import sys
from numpy.linalg import matrix_rank
from numpy.linalg import matrix_power as m_power

from qiskit.quantum_info import Statevector

import matplotlib.pyplot as plt
import numpy as np


def bring_states():
    state_list= [0, 0, 0, 0, 0, 0,
                 0, -1/(2*np.sqrt(2))*1j, 1/(2*np.sqrt(2))*1j,
                 0, 0, 0,0, 0, 0,0, 0, 0,
                 0, 0, 0,0, 0, 0,0, 0, 0,
                 0, 0, 0,0, 0, 0,0, 0, 0,
                 0, 0, 0,0, 0, 0,0, 0, 0,
                 0, 0, 0,0, 0, 0,
                 0, -1/(2*np.sqrt(2))*1j, 0,0, 0, 0,
                 0, 0, 1/(2*np.sqrt(2))*1j,0, 0, 0,
                 0, 0, 0,0, 0, 0, 0, 0, 0,
                 0, 0, 0,0, 0, 0,0, 0, 0,
                 0, -1/(2*np.sqrt(2))*1j, 0,0, 0, 0,
                 0, 0, 0,0, 0, 0,
                 1/(2*np.sqrt(2))*1j, 0, 0,0, -1/(2*np.sqrt(2))*1j, 0,
                 0, 0, 0,0, 0, 0,0, 0, 0,
                 0, 0, 1/(2*np.sqrt(2))*1j,0, 0, 0,
                 0, 0, 0,0, 0, 0,0, 0, 0,
                 0, 0, 0,0, 0]
    State = Statevector(state_list)
    return State


def hamming_distance(s1: Union[str, List[int], Tuple[int]],
                     s2: Union[str, List[int], Tuple[int]]):
    distance = 0
    for i in range(len(s1)):
        # Convert characters to integers if input is string
        bit1 = int(s1[i]) if isinstance(s1, str) else s1[i]
        bit2 = int(s2[i]) if isinstance(s2, str) else s2[i]
        if bit1 != bit2:
            distance += 1
    return distance


def minimum_distance(code: List[Union[str, List[int], Tuple[int]]]) -> int:
    """
    Calculates the minimum Hamming distance for a given code.

    Args:
        code: A list where each element is a codeword
              (string, list, or tuple of 0s and 1s).
              Assumes all codewords have the same length.

    Returns:
        The minimum Hamming distance (d) of the code.
        Returns float('inf') if the code has less than 2 codewords.
    """
    num_codewords = len(code)
    if num_codewords < 2:
        # Minimum distance is not well-defined or is infinite
        return float('inf')

    # Assuming all codewords have the same length as the first one
    codeword_length = len(code[0])
    min_dist = codeword_length + 1 # Initialize with a value larger than any possible distance

    for i in range(num_codewords):
        for j in range(i + 1, num_codewords):
            dist = hamming_distance(code[i], code[j])
            if dist < min_dist:
                min_dist = dist

    return min_dist


# matrix rank over GF(2) 
def matrixRank(mat):
    M=mat.copy() # mat should be an np.array
    m=len(M) # number of rows
    pivots={} # dictionary mapping pivot row --> pivot column
    # row reduction
    for row in range(m):
        pos = next((index for index,value in enumerate(M[row]) if value != 0), -1) #finds position of first nonzero element (or -1 if all 0s)
        if pos>-1:
            for row2 in range(m):
                if row2!=row and M[row2][pos]==1:
                    M[row2]=((M[row2]+M[row]) % 2)
            pivots[row]=pos
    return len(pivots)




def generate_stabilizer_plots(hx_matrix, hz_matrix):
    
    def get_qubit_coords(label):
        if not 0 <= label <= 143: return None
        if label < 72:
            col_idx, row_idx = divmod(label, 6)
            return (col_idx * 2, row_idx * 2)
        else:
            normalized_label = label - 72
            col_idx, row_idx = divmod(normalized_label, 6)
            return (col_idx * 2 + 1, row_idx * 2 + 1)

    def get_stabilizer_coords(index, stabilizer_type):
        if not 0 <= index <= 71: return None
        col_idx, row_idx = divmod(index, 6)
        if stabilizer_type.upper() == 'X':
            return (col_idx * 2, row_idx * 2 + 1)
        elif stabilizer_type.upper() == 'Z':
            return (col_idx * 2 + 1, row_idx * 2)
        else:
            return None
        

    stabilizersZ_to_show = [
        {'index': 5, 'type': 'Z'},
        {'index': 66, 'type': 'Z'},
    ]

    stabilizersX_to_show = [
        {'index': 5, 'type': 'X'},
        {'index': 66, 'type': 'X'},
    ]

    def _create_plot(matrix, stabilizers, title):
        WIDTH, HEIGHT = 24, 12
        NEUTRAL_COLOR, GRID_COLOR = '#d3d3d3', '#e0e0e0'
        HIGHLIGHT_COLORS = ['gold', 'cyan', 'magenta', 'lime']
        
        fig, ax = plt.subplots(figsize=(18, 9))
        ax.set_aspect('equal')


        for j in range(HEIGHT):
            ax.plot([-0.5, WIDTH - 0.5], [j, j], color=GRID_COLOR, linestyle=':', linewidth=1, zorder=1)
        for i in range(WIDTH):
            ax.plot([i, i], [-0.5, HEIGHT - 0.5], color=GRID_COLOR, linestyle=':', linewidth=1, zorder=1)
        for i in range(WIDTH):
            for j in range(HEIGHT):
                if (i % 2) == (j % 2):
                    ax.scatter(i, j, s=50, c=NEUTRAL_COLOR, marker='o', zorder=2)
        

        for i, stab_info in enumerate(stabilizers):
            stabilizer_index = stab_info['index']
            stabilizer_type = stab_info['type'].upper()
            color = HIGHLIGHT_COLORS[i % len(HIGHLIGHT_COLORS)] 
            
            stab_coords = get_stabilizer_coords(stabilizer_index, stabilizer_type)
            if stab_coords:
                sx, sy = stab_coords
                ax.scatter(sx, sy, s=250, c=color, marker='o', zorder=3, label=f"Stabilizer {stabilizer_type}{stabilizer_index}")
                
            user_row = matrix[stabilizer_index]
            connected_qubit_labels = np.where(user_row == 1)[0]
            for label in connected_qubit_labels:
                coords = get_qubit_coords(label)
                if coords:
                    cx, cy = coords
                    ax.scatter(cx, cy, s=300, c=color, marker='o', zorder=4, alpha=0.9)
        
        ax.set_xlim(-1.5, WIDTH); ax.set_ylim(-1.5, HEIGHT)
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False); ax.spines['left'].set_visible(False)
        fig.suptitle(title, fontsize=16)
        ax.legend()
        plt.show()
        
        ax.set_xlim(-1.5, WIDTH); ax.set_ylim(-1.5, HEIGHT)
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False); ax.spines['left'].set_visible(False)
        fig.suptitle(title, fontsize=16)
        ax.legend()
        plt.show()

    _create_plot(hx_matrix, stabilizersX_to_show, "X-Stabilizer Check - 5, 66")
    _create_plot(hz_matrix, stabilizersZ_to_show, "Z-Stabilizer Check - 5, 66")
