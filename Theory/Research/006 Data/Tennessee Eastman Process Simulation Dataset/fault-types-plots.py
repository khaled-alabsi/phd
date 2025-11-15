import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple

# Constants
N: int = 300  # total time steps
FAULT_TIME: int = 160  # fault onset


def generate_step_fault(normal_value: float = 1.0,
                        fault_value: float = 2.0) -> np.ndarray:
    series = np.full(N, normal_value)
    series[FAULT_TIME:] = fault_value
    return series


def generate_random_fault(normal_value: float = 1.0,
                          noise_std: float = 0.1) -> np.ndarray:
    np.random.seed(0)
    series = np.full(N, normal_value)
    noise = np.random.normal(0, noise_std, N - FAULT_TIME)
    series[FAULT_TIME:] += noise
    return series


def generate_drift_fault(normal_value: float = 1.0,
                         final_value: float = 2.0) -> np.ndarray:
    series = np.full(N, normal_value)
    drift_length = N - FAULT_TIME
    drift = np.linspace(normal_value, final_value, drift_length)
    series[FAULT_TIME:] = drift
    return series


def generate_stuck_fault(
        normal_value: float = 1.0,
        stuck_value: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
    actual = np.full(N, normal_value)
    control = np.full(N, normal_value)
    actual[FAULT_TIME:] = stuck_value
    control[FAULT_TIME:] = np.linspace(
        normal_value, 2.0, N - FAULT_TIME)  # controller tries to correct
    return actual, control


def plot_faults():
    t = np.arange(N)

    # Step Fault
    step_series = generate_step_fault()

    # Random Fault
    random_series = generate_random_fault()

    # Drift Fault
    drift_series = generate_drift_fault()

    # Stuck Fault
    stuck_actual, stuck_control = generate_stuck_fault()

    # Plotting
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    axs[0].plot(t, step_series, label="Step Fault", color='blue')
    axs[0].axvline(FAULT_TIME, color='gray', linestyle='--')
    axs[0].set_ylabel("Value")
    axs[0].set_title("Step Fault")

    axs[1].plot(t, random_series, label="Random Fault", color='orange')
    axs[1].axvline(FAULT_TIME, color='gray', linestyle='--')
    axs[1].set_ylabel("Value")
    axs[1].set_title("Random Fault")

    axs[2].plot(t, drift_series, label="Drift Fault", color='green')
    axs[2].axvline(FAULT_TIME, color='gray', linestyle='--')
    axs[2].set_ylabel("Value")
    axs[2].set_title("Drift Fault")

    axs[3].plot(t, stuck_actual, label="Actual (Stuck)", color='red')
    axs[3].plot(t,
                stuck_control,
                label="Controller Output",
                color='black',
                linestyle='--')
    axs[3].axvline(FAULT_TIME, color='gray', linestyle='--')
    axs[3].set_ylabel("Value")
    axs[3].set_title("Stuck Fault")
    axs[3].legend()

    plt.xlabel("Time")
    plt.tight_layout()
    plt.show()


plot_faults()
