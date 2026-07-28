from nmrglue.analysis.lineshapes1d import location_2params, location_scale, sim_lorentz_fwhm, center_fwhm
import numpy as np

class split_lorentz_fwhm(location_2params):
    """
    Lorentzian lineshape class representing a split peak, with height at center of
    each multiplet, full-width half-maximum, and J-constant parameters.
    """
    name = "split-lorentz"

    def sim(self, M, p):
        x = np.arange(M)
        x0, fwhm, j = p
        return sim_lorentz_fwhm(x, x0 - int(j/2), fwhm) + sim_lorentz_fwhm(x, x0 + int(j/2), fwhm)

    def guessp(self, sig):
        """Probably won't work."""
        c, fwhm = center_fwhm(sig)
        return (c, fwhm, 250)

    def pnames(self, M):
        return ("x0", "fwhm", "J-constant")

def z(x, alpha, beta):
    """Helper for Gumbel function exponent."""
    return (x - alpha)/beta

def sim_gumbel_beta(x, alpha, beta):
    """
    Simulate a Gumbel lineshape with unit height at the mode (alpha).
    Simulate discrete points of a continuous Gumbel distribution with unit
    height at the center.  Beta (the horizontal scale of the distribution)
    is used as the distribution scale parameter.
    Functional form:
        f(x; alpha, beta) = exp( 1 - z - exp( -z ))
            where z = (x - alpha)/beta
    Parameters
    ----------
    x : ndarray
        Array of values at which to evaluate the distribution.
    alpha : float
        Center (mode) of Gumbel distribution.
    beta : float
        Scale of the Gumbel distribution.
    Returns
    -------
    f : ndarray
        Distribution evaluated at points in x.
    """
    z = (x - alpha)/beta
    return np.exp(1 - z - np.exp(-z))


class gumbel_hb(location_scale):
    """
    Gumbel lineshape class with unit height at alpha (the mode) and beta scale 
    parameter.
    """
    name = "gumbel"

    def sim(self, M, p):
        x = np.arange(M)
        alpha, beta = p
        return sim_gumbel_beta(x, alpha, beta)

    def guessp(self, sig):
        """Probably won't work."""
        c, fwhm = center_fwhm(sig)
        return (c, fwhm / np.e)

    def pnames(self, M):
        return ("alpha", "beta")