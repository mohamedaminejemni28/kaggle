import numpy as np
from scipy.stats import shapiro, kurtosis, normaltest


def swtest_normality(x, alpha=0.05):
    """
    Shapiro-Wilk (or Shapiro-Francia) normality test.
    If the sample is leptokurtic (kurtosis > 3), use Shapiro-Francia (approximate with normaltest),
    else use Shapiro-Wilk.

    Parameters
    ----------
    x : array-like
        Input data (1D array).
    alpha : float, optional
        Significance level (default: 0.05).

    Returns
    -------
    H : int
        0 if null hypothesis (normality) is not rejected, 1 if rejected.
    p_value : float
        p-value of the test.
    W : float
        Test statistic (W for Shapiro-Wilk, or statistic for Shapiro-Francia/normaltest).
    """
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    if x.ndim != 1:
        raise ValueError("Input sample 'x' must be a 1D array.")
    n = len(x)
    if n < 3:
        raise ValueError("Sample vector 'x' must have at least 3 valid observations.")
    if n > 5000:
        import warnings
        warnings.warn("Shapiro-Wilk test might be inaccurate for n > 5000.")
    k = kurtosis(x, bias=False, fisher=False)  # Pearson kurtosis
    if k > 3:
        # Leptokurtic: use Shapiro-Francia (approximate with normaltest)
        stat, p_value = normaltest(x)
        W = stat
    else:
        # Platykurtic: use Shapiro-Wilk
        W, p_value = shapiro(x)
    H = int(alpha >= p_value)
    return H, p_value, W


if __name__ == "__main__":
    # Example usage
    data = np.random.normal(size=100)
    H, p, W = swtest_normality(data)
    print(f"H: {H}, p-value: {p}, W/statistic: {W}") 