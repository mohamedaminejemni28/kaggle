import numpy as np
from scipy.stats import skew, kurtosis


def bimodality_coeff(x):
    """
    Calculate the bimodality coefficient and flag for a dataset.

    Parameters
    ----------
    x : array-like
        Input data. Can be a 1D array (vector) or 2D array (matrix).
        If 2D, columns are treated as separate variables.

    Returns
    -------
    BF : np.ndarray
        Bimodality flag (True if BC > 1.05*5/9, else False) for each column.
    BC : np.ndarray
        Bimodality coefficient for each column.
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    N = x.shape[0]
    S = skew(x, axis=0, bias=False)
    K = kurtosis(x, axis=0, bias=False)  # normal distribution has kurtosis 0
    # Calculate the bimodality coefficient (unbiased)
    BC = (S ** 2 + 1) / (K + 3 * (N - 1) ** 2 / ((N - 2) * (N - 3)))
    # Determine the bimodality flag (using +5% margin)
    BF = BC > 1.05 * 5 / 9
    return BF, BC


if __name__ == "__main__":
    # Example usage
    data = np.random.normal(size=1000)
    bf, bc = bimodality_coeff(data)
    print(f"Bimodality flag: {bf}")
    print(f"Bimodality coefficient: {bc}") 