import numpy as np

def sample_lkj(d, eta=1.0, n=1, rng=None):
    """
    Sample n correlation matrices from LKJ(eta).

    Parameters
    ----------
    d : int
        Dimension.
    eta : float
        LKJ concentration, > 0.
    n : int
        Number of matrices.
    rng : np.random.Generator, optional

    Returns
    -------
    R : ndarray, shape (n, d, d)
    """
    if eta <= 0:
        raise ValueError("eta must be > 0")
    if d < 1:
        raise ValueError("d must be >= 1")

    rng = np.random.default_rng() if rng is None else rng

    L = np.zeros((n, d, d), dtype=np.float64)
    L[:, 0, 0] = 1.0

    for i in range(1, d):
        alpha = eta + 0.5 * (d - i - 1)

        # i independent partial correlations for row i
        u = rng.beta(alpha, alpha, size=(n, i))
        rho = 2.0 * u - 1.0

        # Convert partial correlations into Cholesky row.
        prod = np.ones(n)

        for j in range(i):
            L[:, i, j] = rho[:, j] * np.sqrt(prod)
            prod *= 1.0 - rho[:, j] ** 2

        L[:, i, i] = np.sqrt(prod)

    return L @ np.swapaxes(L, -1, -2)

def random_lkj_matrix(d, eta_min=0.1, eta_max=10.0):
    return sample_lkj(d, eta=np.random.uniform(eta_min, eta_max),n=1)[0]


if __name__=="__main__":
    print(random_correlation_matrix(10))   


