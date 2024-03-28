import torch
import numpy as np
import os
import random
import pandas as pd
from scipy.stats import weibull_min


def seed_everything(seed: int):
    """Set random seeds for reproducibility.

    This function should be called only once per python process, preferably at the beginning of the main script.
    It has global effects on the random state of the python process, so it should be used with care.

    Args:
        seed: random seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def reset_wandb_env():
    """Reset the wandb environment variables.

    This is useful when running multiple sweeps in parallel, as wandb
    will otherwise try to use the same directory for all the runs.
    """
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


def load_ark_data(file_path):
    """
    Loads data from a CSV file and returns specific columns related to quay wall maintenance.

    This function reads the first row of a CSV file specified by the file path and extracts
    information related to the condition and damage scores of wooden poles, kesps, and connections
    between poles and kesps or poles and floors. The scores are adjusted by subtracting 1 to align
    with a zero-based indexing.

    Args:
        file_path (str): The path to the CSV file containing the maintenance data.

    Returns:
        tuple: A tuple containing the extracted data from the CSV file, including:
            - Number of deteriorated wooden poles.
            - Score for the deterioration of wooden poles, adjusted to zero-based indexing.
            - Number of damaged kesps.
            - Score for the damage of kesps, adjusted to zero-based indexing.
            - Number of damaged pole-kesp/pole-floor connections.
            - Score for the damage of pole-kesp/pole-floor connections, adjusted to zero-based indexing.

    Notes:
        - The CSV file is expected to have specific column names that match the elements of interest
          for quay wall maintenance analysis.
        - This function is designed to work with data where the first row contains relevant information,
          making it suitable for files formatted to contain summary or aggregated maintenance data.
    """
    df = pd.read_csv(file_path, index_col=False, nrows=1)
    return (
        df['aantasting houten palen'].iloc[0],
        int(df['aantasting houten palen score1'].iloc[0])-1,
        df['beschadigde kesp'].iloc[0],
        int(df['beschadigde kesp score3'].iloc[0])-1,
        df['beschadigde paal-kesp/paal-vloer'].iloc[0],
        int(df['paal-kesp/paal-vloer score4'].iloc[0])-1
            )


def generate_weighted_states(n_components, start_states, probs, risk):
    """
    Generate a weighted list of start states for wooden components.

    This function assigns the highest probability (specified by `probs`) to the risk state.
    The remaining states are assigned the complementary probability, which is equally divided
    among them.

    Args:
        n_components (int): The number of components.
        start_states (ndarray): A list of possible start states.
        probs (int): The probability for the risk state expressed as a percentage (0-100).
        risk (int): The start state that is considered at risk and should receive the highest probability.

    Returns:
        numpy.ndarray: An array of weighted start states for the components. The risk state has the
        probability specified by `probs`, and each of the other states has an equal share of the
        remaining probability.
    """
    weights = [(100 - probs) / (n_components - 1 if num != risk else 0) if num != risk else probs for num in start_states]
    return np.array(random.choices(start_states, weights=weights, k=n_components))


def generate_proportionally_weighted_states(n_components, start_states, probs, risk):
    """
    Generate a list of start states for wooden components with weights decreasing proportionally
    to their distance from the risk state.

    Args:
        n_components (int): The number of components.
        start_states (ndarray): A list of possible start states.
        probs (int): The probability for the risk state.
        risk (int): The start state that is considered at risk.

    Returns:
        numpy.ndarray: An array of start states for the components with proportional weights.
    """
    # Calculate base weights inversely proportional to the distance from the risk state
    base_weights = [1 / (abs(num - risk) + 0.1) for num in start_states]  # +1e-5 to avoid division by zero

    # Adjust the weights so that the sum is 100, maintaining the proportionality
    sum_base_weights = sum(base_weights)
    adjusted_weights = [probs * (w / base_weights[risk]) if num == risk else ((100 - probs) * (w / (sum_base_weights - base_weights[risk]))) for num, w in enumerate(base_weights)]

    return np.array(random.choices(start_states, weights=adjusted_weights, k=n_components))


def weibull_interpolation(P, P_start, ndeterioration, lambda_=92, kappa=2):
    """
    Adjust the transition matrix over time using a Weibull distribution, where the probability
    of moving to a higher state increases, and the probability of staying in the same state decreases.

    This function creates ndeterioration + 1 steps of matrices starting from P_start,
    where the values in P_start increase over time influenced by a Weibull distribution with kappa > 1.
    Formula = (F = 1 - np.exp(-((x / lambda_) ** kappa))

    Args:
        P (numpy.ndarray): The initialized final matrix.
        P_start (numpy.ndarray): The starting matrix.
        ndeterioration (int): The number of interpolation steps.
        lambda_ (float, optional): The scale parameter of the Weibull distribution. Default is 1.0.
        kappa (float, optional): The shape parameter of the Weibull distribution. Default is 2.5.

    Returns:
        numpy.ndarray: A 3D matrix (ndeterioration+1 x ntypes x nstcomp x nstcomp) where each matrix
                       represents the transition probabilities at a given time step.

    Raises:
        ValueError: If ndeterioration is less than 0 or kappa is less than or equal to 1.
    """
    if ndeterioration < 1:
        raise ValueError("ndeterioration must be at least 1")
    if kappa <= 1:
        raise ValueError("kappa must be greater than 1 for the function to model increasing probabilities")

    P[0] = P_start

    for t in range(1, ndeterioration):
        P_copy = P_start.copy()
        # Calculate the Weibull factor for this time t
        weibull_factor = weibull_min.cdf(t, kappa, 0, lambda_)

        for i in range(P_start.shape[0]):  # Iterate through each matrix
            for j in range(P_start.shape[1] - 1):  # Last row is the absorbing state, no need to change
                # Increase the probability of transitioning to a more severe state based on the Weibull factor
                P_copy[i, j, j + 1] += (1 - P_copy[i, j, j + 1]) * weibull_factor  # Increase to the next state
                P_copy[i, j, j] -= P_copy[i, j, j + 1] - P_start[i, j, j + 1]  # Adjust the current state probability

                # Ensure that the probabilities sum to 1, considering the updated next state probability
                if j < P_start.shape[1] - 2:  # If not penultimate state
                    # Scale the remaining probabilities to sum to 1
                    sum_remaining = 1 - (P_copy[i, j, j] + P_copy[i, j, j + 1])
                    P_copy[i, j, j + 2:] = P_start[i, j, j + 2:] / np.sum(P_start[i, j, j + 2:]) * sum_remaining

                # For the penultimate state, assign all remaining probability to the final state
                else:
                    P_copy[i, j, -1] = 1 - P_copy[i, j, j]

        P[t] = P_copy

    print(list(P[-1].round(3)))
    return P


def linear_interpolation(P, P_start, P_end, ndeterioration):
    """
    Interpolate between two matrices using linear interpolation.

    This function linearly interpolates between P_start and P_end matrices over
    ndeterioration steps.

    Args:
        P: The initialized final matrix.
        P_start (numpy.ndarray): The starting matrix.
        P_end (numpy.ndarray): The ending matrix.
        ndeterioration (int): The number of interpolation steps.

    Returns:
        numpy.ndarray: A matrix of linearly interpolated values.

    Raises:
        ValueError: If ndeterioration is less than 1.
    """
    if ndeterioration < 1:
        raise ValueError("ndeterioration must be at least 1")

    for i in range(ndeterioration):
        # Calculate the linear interpolation factor
        factor = i / (ndeterioration - 1)
        # Apply the interpolation
        P[i, :, :] = P_start + (P_end - P_start) * factor

    return P
