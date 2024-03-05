import json
import pennylane as qml
import pennylane.numpy as np

def potential_energy_surface(symbols, bond_lengths):
    """Calculates the molecular energy over various bond lengths (AKA the 
    potential energy surface) using the Hartree Fock method.
    
    Args:
        symbols (list(string)): 
            A list of atomic symbols that comprise the diatomic molecule of interest.
        bond_lengths (numpy.tensor): Bond lengths to calculate the energy over.

        
    Returns:
        hf_energies (numpy.tensor): 
            The Hartree Fock energies at every bond length value.
    """


    hf_energies = []

    # Put your code here #
    for bond_length in bond_lengths:
        #https://mitiq.readthedocs.io/en/stable/examples/molecular_hydrogen_pennylane.html
        ang_to_bohr = 1.8897259886 # pennylane specifies coordinates in terms of Bohr radius
        geometry = ang_to_bohr * np.array([[0., 0., - 0.5 * bond_length],[0., 0., 0.5 * bond_length]])
        #geometry = np.array([[0., 0., 0.],[0., 0., bond_length]])
        mol = qml.qchem.Molecule(symbols, geometry)
        print(qml.qchem.hf_energy(mol)(geometry))
        hf_energies.append(qml.qchem.hf_energy(mol)(geometry))
    return np.array(hf_energies)


def ground_energy(hf_energies):
    """Finds the minimum energy of a molecule given its potential energy surface.
    
    Args: 
        hf_energies (numpy.tensor): 

    Returns:
        (float): The minumum energy in units of hartrees.
    """

    ind = np.argmin(hf_energies)
    return hf_energies[ind]

def reaction():
    """Calculates the energy of the reactants, the activation energy, and the energy of 
    the products in that order.

    Returns:
        (numpy.tensor): [E_reactants, E_activation, E_products]
    """
    molecules = {
        "H2": 
            {"symbols": ["H", "H"], "E0": 0, "E_dissociation": 0, "bond lengths": np.arange(0.5, 9.3, 0.3)}, 
        "Li2": 
            {"symbols": ["Li", "Li"], "E0": 0, "E_dissociation": 0, "bond lengths": np.arange(3.5, 8.3, 0.3)}, 
        "LiH": 
            {"symbols": ["Li", "H"], "E0": 0, "E_dissociation": 0, "bond lengths": np.arange(2.0, 6.6, 0.3)}
    }


    for molecule in molecules.keys():
        # Put your code here #
        print(molecules[molecule])
        hf_energies = potential_energy_surface(molecules[molecule]["symbols"], molecules[molecule]["bond lengths"])
        # populate each molecule's E0 and E_dissociation values
        molecules[molecule]["E0"] = ground_energy(hf_energies)
        molecules[molecule]["E_dissociation"] = np.abs(hf_energies[-1] - molecules[molecule]["E0"])


    # Calculate the following and don't forget to balance the chemical reaction!
    # calculate the energy of the reactants 
    E_reactants = molecules["H2"]["E0"] + molecules["Li2"]["E0"]
    # calculate the activation energy
    E_activation = E_reactants + (molecules["Li2"]["E_dissociation"] + molecules["H2"]["E_dissociation"])
    # calculate the energy of the products
    E_products = molecules["LiH"]["E0"] + molecules["LiH"]["E_dissociation"]

    return np.array([E_reactants, E_activation, E_products])


# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    output = reaction().tolist()
    return str(output)

def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    print(solution_output, expected_output)
    assert np.allclose(solution_output, expected_output, rtol=1e-3)

test = run('')
print(test)
#check (test)