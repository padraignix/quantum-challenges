from qiskit.quantum_info import SparsePauliOp
import numpy as np
from qiskit import transpile
from qiskit import QuantumCircuit
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager




def linear_model(x, a, b):
    return a * x + b


def quadratic_model(x, a, b, c):
    return a * x**2 + b * x + c


def exponential_model(x, a, b, c):
    return a * np.exp(-b * x) + c


def zne_method(method="linear", xdata=[], ydata=[]):

    if method == "linear":
        popt, _ = curve_fit(linear_model, xdata, ydata)
        zero_val = linear_model(0, *popt)
        fit_fn = linear_model
    elif method == "quadratic":
        popt, _ = curve_fit(quadratic_model, xdata, ydata)
        zero_val = quadratic_model(0, *popt)
        fit_fn = quadratic_model
    elif method == "exponential":
        popt, _ = curve_fit(exponential_model, xdata, ydata, p0=(1, 0.1, 0), maxfev=5000)
        zero_val = exponential_model(0, *popt)
        fit_fn = exponential_model
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'linear', 'quadratic', or 'exponential'.")

    return zero_val, ydata, popt, fit_fn


def plot_zne(scales, values, zero_val, fit_fn, fit_params, method):
    x_plot = np.linspace(0, max(scales), 200)
    y_plot = fit_fn(x_plot, *fit_params)

    plt.figure(figsize=(8, 5))
    plt.plot(scales, values, "o", label="Noisy measurements")
    plt.plot(x_plot, y_plot, "-", label=f"{method.capitalize()} fit")
    plt.axvline(0, linestyle="--", color="gray")
    plt.axhline(zero_val, linestyle="--", color="red", label="Zero-noise estimate")
    plt.xlabel("Noise scaling factor")
    plt.ylabel("⟨Z⟩ Expectation Value")
    plt.title(f"Zero-Noise Extrapolation ({method})")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_backend_errors_and_counts(backends, errors_and_counts_list):

    # Unpack errors and counts from the list
    (
        acc_total_errors,
        acc_two_qubit_errors,
        acc_single_qubit_errors,
        acc_readout_errors,
        single_qubit_gate_counts,
        two_qubit_gate_counts,
    ) = np.array(errors_and_counts_list).T.tolist()

    errors = np.array(
        [
            acc_total_errors,
            acc_two_qubit_errors,
            acc_single_qubit_errors,
            acc_readout_errors,
        ]
    )
    error_labels = [
        "Total Error",
        "Two-Qubit Error",
        "Single-Qubit Error",
        "Readout Error",
    ]
    counts = np.array([single_qubit_gate_counts, two_qubit_gate_counts])
    count_labels = ["Single-Qubit Gate Count", "Two-Qubit Gate Count"]

    # Transpose errors and counts to align with plotting requirements
    errors = errors.T
    counts = counts.T
    # Plot for accumulated errors
    x = np.arange(len(error_labels))  # the label locations
    width = 0.2  # the width of the bars
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(len(backends)):
        ax.bar(x + i * width, errors[i], width, label=backends[i].name)
    ax.set_xlabel("Error Type")
    ax.set_ylabel("Accumulated Error")
    ax.set_title("Accumulated Errors by Backend")
    ax.set_xticks(x + width)
    ax.set_xticklabels(error_labels)
    ax.legend()
    plt.show()

    # Plot for gate counts
    x = np.arange(len(count_labels))  # the label locations

    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(len(backends)):
        ax.bar(x + i * width, counts[i], width, label=backends[i].name)

    ax.set_xlabel("Gate Type")
    ax.set_ylabel("Gate Count")
    ax.set_title("Gate Counts by Backend")
    ax.set_xticks(x + width)
    ax.set_xticklabels(count_labels)
    ax.legend()
    plt.show()
