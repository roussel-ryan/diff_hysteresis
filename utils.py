import torch
import numpy as np


def generate_mesh(h_sat, n):
    xvalues = np.linspace(-h_sat, h_sat, n)
    yvalues = np.linspace(-h_sat, h_sat, n)
    xx, yy = np.meshgrid(xvalues, yvalues)
    return torch.tensor(xx), torch.tensor(yy)


def vector_to_tril(vector, n):
    """Returns a simulated hysterion density as an nxn tensor.
    Density means the number of hysterions given (alpha, beta).
    The lower triangle is zeroed out because the space
    where alpha<beta is non-physical.

    Parameters
    ----------
    vector : array,
        A vector of dummy values for the initial density. The
        size must be n**2/2 + n/2 to populate the upper triangle,
        including the diagonal.

    n : int,
        The number of points expected in the discretization.

    Raises
    ------
    RuntimeError
        If len(dens_i) does not equal n**2/2 + n/2.

    ValueError
        If n is negative.
    """
    assert vector.shape[0] == n**2 / 2 + n / 2
    # zeroed lower triangle
    dens = torch.zeros((n, n)).double()
    idx = torch.tril_indices(row=n, col=n, offset=0)
    dens[idx[0], idx[1]] = vector
    return dens


def tril_to_vector(tril, n):
    """Returns a simulated hysterion density as an nxn tensor.
    Density means the number of hysterions given (alpha, beta).
    The lower triangle is zeroed out because the space
    where alpha<beta is non-physical.

    Parameters
    ----------
    tril : array,
        A vector of dummy values for the initial density. The
        size must be n**2/2 + n/2 to populate the upper triangle,
        including the diagonal.

    n : int,
        The number of points expected in the discretization.

    Raises
    ------
    RuntimeError
        If len(dens_i) does not equal n**2/2 + n/2.

    ValueError
        If n is negative.
    """
    assert tril.shape[0] == n
    assert tril.shape[0] == tril.shape[1]

    # zeroed lower triangle
    idx = torch.tril_indices(row=n, col=n, offset=0)
    vector = tril[idx[0], idx[1]]
    return vector


def get_upper_trainagle_size(n):
    return n ** 2 / 2 + n / 2

