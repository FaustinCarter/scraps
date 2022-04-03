import lmfit
import numpy as np
from scraps.fitsS21 import utils


def offset(freqs, re0, im0):
    """Complex offset re + j*im.

    Freqs vector is ignored, but required for lmfit Model."""
    return re0 + 1j * im0


def mag(freqs, g0, g1, g2):
    """2nd order polynomial.

    References the freqs-array midpoint, which is important because the baseline coefs shouldn't drift
    around with changes in f0 due to power or temperature."""
    x = utils.reduce_by_midpoint(freqs)
    return g0 + g1 * x + g2 * x ** 2


def phase(freqs, p0, p1, p2):
    """Angle in complex plane parameterized by 2nd order polynomial.

    References the freqs-array midpoint, which is important because the baseline coefs shouldn't drift
    around with changes in f0 due to power or temperature."""
    x = utils.reduce_by_midpoint(freqs)
    phi = p0 + p1 * x + p2 * x ** 2
    return np.exp(1j * phi)


class ModelMagBaseline(lmfit.Model):
    __doc__ = (
        "lmfit model that fits a 2nd order polynomial as a function of reduced frequency."
        + lmfit.models.COMMON_INIT_DOC
    )

    def __init__(self, *args, **kwargs):
        super().__init__(mag, *args, **kwargs)

    def guess(self, data, freqs=None, mask=None, **kwargs):
        """Mask is an number for how many points to keep on each end. If int, then assumes
        number of points, if float then assumes percent of data. If None, keeps everything."""

        if freqs is None:
            raise ValueError("Must pass a frequencies vector")

        mag_data = np.abs(data)
        xvals = utils.reduce_by_midpoint(freqs)

        masked_data = utils.mask_array_ends(mag_data, mask)
        masked_xvals = utils.mask_array_ends(xvals, mask)

        g2, g1, g0 = np.polyfit(masked_xvals, masked_data, 2)

        params = self.make_params(g2=g2, g1=g1, g0=g0)

        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)


class ModelPhaseBaseline(lmfit.Model):
    __doc__ = (
        "lmfit model that fits a 2nd order polynomial to phase angle as a function of reduced frequency."
        + lmfit.models.COMMON_INIT_DOC
    )

    def __init__(self, *args, **kwargs):
        super().__init__(phase, *args, **kwargs)

    def guess(self, data, freqs=None, mask=None, phase_step=None, **kwargs):
        """Treats phase angle as either a 1st or 2nd order polynomial.

        Passing a float to phase_step will cause that amount to be subtracted
        from the first half of the masked data prior to fitting a polynomial.
        This is useful if it's clear that the phase rolls sharply
        through either pi or 2pi, but it's also obvious there is some strong
        frequency-dependent behavior.

        Passing fit_quadratic_phase will cause the baseline to be modeled as
        a second-order polynomial. The default value for this is False."""

        if freqs is None:
            raise ValueError("Must pass a frequencies vector")

        fit_quadratic_phase = kwargs.pop("fit_quadratic_phase", False)
        unwrap_phase = kwargs.pop("unwrap_phase", True)

        phase_data = np.angle(data)
        xvals = utils.reduce_by_midpoint(freqs)

        if unwrap_phase:
            phase_data = np.unwrap(phase_data)

        masked_data = utils.mask_array_ends(phase_data, mask)
        masked_xvals = utils.mask_array_ends(xvals, mask)

        if phase_step:
            step_vector = np.ones_like(masked_data)
            step_vector[-len(step_vector) // 2 :] = 0
            step_vector *= phase_step
            masked_data -= step_vector

        if fit_quadratic_phase:
            p2, p1, p0 = np.polyfit(masked_xvals, masked_data, 2)
        else:
            p1, p0 = np.polyfit(masked_xvals, masked_data, 1)
            p2 = 0

        params = self.make_params(p2=p2, p1=p1, p0=p0)

        if not fit_quadratic_phase:
            params[f"{self.prefix}p2"].set(vary=False)

        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)


class ModelComplexOffset(lmfit.Model):
    __doc__ = "lmfit model that fits a complex offset." + lmfit.models.COMMON_INIT_DOC

    def __init__(self, *args, **kwargs):
        super().__init__(offset, *args, **kwargs)

    def guess(self, data, freqs=None, mask=None, **kwargs):
        """Guesses the real and imaginary offsets as the mean of the
        masked real and imaginary parts of data, respectively."""

        masked_data = utils.mask_array_ends(data, mask)

        re = np.real(masked_data).mean()
        im = np.imag(masked_data).mean()

        params = self.make_params(re=re, im=im)

        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)
