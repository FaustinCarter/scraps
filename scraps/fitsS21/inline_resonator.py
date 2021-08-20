"""Functions for fitting inline/transmission resonators.

These are resonators with a traditional Lorentzian lineshape (i.e. 1 on resonance, 0 off resonance).
"""

import types
import warnings

import lmfit
import numpy as np
from scraps.fitsS21 import baselines, utils


def cmplx_inline(freqs, f0, df, qc, q0):
    """An asymmetric lineshape vs frequency.

    Pass df = 0 to recover the standard complex Lorentzian shape."""

    ft = f0 + df
    x = (freqs - ft) / ft
    dx = df / ft

    s21 = (q0 / qc) * (1 - 2j * qc * dx) / (1 + 2j * q0 * x)

    return s21


class ModelInlineResonator(lmfit.model.Model):
    __doc__ = (
        "lmfit model that fits the complex transmission (S_21) of an inline/transmission style resonator."
        + lmfit.models.COMMON_DOC
    )

    def __init__(self, *args, **kwargs):
        super().__init__(cmplx_inline, *args, **kwargs)

        self.set_param_hint("qc", min=0)
        self.set_param_hint("q0", min=0)

    def guess(self, data, freqs=None, **kwargs):
        """Guess some reasonable initial values for fit parameters.

        Can override any of the guessed values by passing param_name=value
        as keyword argument.

        Pass fit_df = False to assume df is zero and magnitude
        is set by q0/qc. In this case, the value returned for qc will
        also contain any uncalibrated gain."""

        if freqs is None:
            raise ValueError("Must pass a frequencies vector")

        fit_df = kwargs.pop("fit_df", False)

        # Calculate magnitude and phase
        mag_s21 = np.abs(data)
        phase_s21 = np.angle(data)

        # Resonant frequency is probably where the peak is
        max_s21 = mag_s21.max()
        argmax_s21 = np.argmax(mag_s21)
        ft = f0 = freqs[argmax_s21]
        phase0 = phase_s21[argmax_s21]

        # Calculate the linewidth to guess the Q
        fwhm = np.sqrt(max_s21 ** 2 / 2)
        fwhm_mask = mag_s21 > fwhm
        bandwidth = freqs[fwhm_mask].max() - freqs[fwhm_mask].min()

        # A reasonable minimum bound for Q0 is center frequency divided by
        # the bandwidth of the full dataset
        q0_min = f0 / (freqs.max() - freqs.min())

        # A reasonable maximum bound is the center frequency divided by
        # the minimum frequency spacing of the dataset
        q0_max = f0 / np.abs(np.ma.masked_equal(np.diff(freqs), 0)).min()

        q0 = f0 / bandwidth

        if not q0_max > q0 > q0_min:
            q0 = np.sqrt(q0_min * q0_max)
            warnings.warn(
                "q0 = f0/fwhm_bandwith results in impossible value."
                + " Falling back on sqrt(q0_min*q0_max) and setting guess_qc = False."
            )
            fit_df = False

        if fit_df:
            # Self-consistently calculate f0, df, and qc
            # In practice this doesn't seem to need more than one or two passes
            # so allowing up to 5 seems like a good worst-case scenario
            f0_new = f0
            for ix in range(5):
                q0 = f0_new / bandwidth

                qc = q0 / max_s21 * np.sqrt(1 + np.tan(phase0) ** 2)

                dx = -1 / (2 * qc) * np.tan(phase0)

                df = f0_new * dx / (1 - dx)

                f0 = ft - df

                # Check whether the new guess is good to within a fraction of a linewidth
                if np.abs(f0 - f0_new) < 0.1 * bandwidth:
                    break
                else:
                    f0_new = f0

                if ix == 4:
                    warnings.warn("Failure to estimate df and qc self-consistently.")

        else:
            q0 = f0 / bandwidth
            qc = q0 / max_s21
            df = 0

        params = self.make_params(f0=f0, df=df, qc=qc, q0=q0)
        params[f"{self.prefix}q0"].set(min=q0_min, max=q0_max)

        if not fit_df:
            # In this case, the f0 parameter will track the resonant peak
            params[f"{self.prefix}df"].set(vary=False)
            params[f"{self.prefix}f0"].set(min=freqs.min(), max=freqs.max())
        else:
            # If allowing df to vary, then it's actually the sum f0+df that
            # tracks the resonant peak
            params.add(f"{self.prefix}ft", expr="f0+df", min=freqs.min(), max=freqs.max())

        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)


# Make up a composite model for an inline resonator with arbitrary baseline and offset
complex_baseline = baselines.ModelMagBaseline() * baselines.ModelPhaseBaseline()
inline_resonator_full = ModelInlineResonator() * complex_baseline + baselines.ModelComplexOffset()


def inline_resonator_full_guess(self, data, freqs, mask=0.05, **kwargs):

    mag_mask = kwargs.pop("mag_mask", mask)
    phase_mask = kwargs.pop("phase_mask", mask)
    offset_mask = kwargs.pop("offset_maks", mask)

    fit_baseline = kwargs.pop("fit_baseline", False)
    fit_offset = kwargs.pop("fit_offset", False)

    use_filter = kwargs.pop("use_filter", False)

    if use_filter:
        re_filt = utils.filter_data(np.real(data))
        im_filt = utils.filter_data(np.imag(data))
        data = re_filt + 1j * im_filt

    fit_baseline_vals = [True, False, "mag", "phase", "both"]

    if fit_baseline not in fit_baseline_vals:
        raise ValueError(
            f"Invalid value for fit_baseline. Allowed values are {', '.join(fit_baseline_vals)}."
        )

    if fit_offset not in [True, False]:
        raise ValueError("Invalid value for fit_offset. Allowed values are True or False.")

    # Grab best guess for baseline params
    # Left = inline resonator
    # Right = mag gain * phase gain

    if fit_baseline in [True, "mag", "both"]:
        gain_params = self.left.right.left.guess(data, freqs, mag_mask, **kwargs)
    else:
        gain_params = self.left.right.left.make_params(g0=1, g1=0, g2=0)
        gain_params[f"{self.prefix}g0"].set(vary=False)
        gain_params[f"{self.prefix}g1"].set(vary=False)
        gain_params[f"{self.prefix}g2"].set(vary=False)

    if fit_baseline in [True, "phase", "both"]:
        phase_params = self.left.right.right.guess(data, freqs, phase_mask, **kwargs)
    else:
        phase_params = self.left.right.right.make_params(p0=0, p1=0, p2=0)
        phase_params[f"{self.prefix}p0"].set(vary=False)
        phase_params[f"{self.prefix}p1"].set(vary=False)
        phase_params[f"{self.prefix}p2"].set(vary=False)

    if fit_offset:
        offset_params = self.right.guess(data, freqs, offset_mask, **kwargs)
    else:
        offset_params = self.right.make_params(re0=0, im0=0)
        offset_params[f"{self.prefix}re0"].set(vary=False)
        offset_params[f"{self.prefix}im0"].set(vary=False)

    # Calculate the baselines
    mag_baseline_guess = self.left.right.left.eval(gain_params, freqs=freqs)
    phase_baseline_guess = self.left.right.right.eval(phase_params, freqs=freqs)
    offset_guess = self.right.eval(offset_params, freqs=freqs)

    # Try and make a best guess for clean data
    reduced_data = (data - offset_guess) / (mag_baseline_guess * phase_baseline_guess)

    # Calculate baseline params from best-guess clean data
    inline_params = self.left.left.guess(reduced_data, freqs, **kwargs)

    if not kwargs.get("fit_df", False):
        if fit_baseline in [True, "both", "mag"]:
            inline_params[f"{self.prefix}qc"].set(value=1, vary=False)

    # Final params is all of the params concatenated
    params = inline_params + gain_params + phase_params + offset_params

    return lmfit.models.update_param_vals(params, "", **kwargs)


# Attach this to the model instance so it behaves like the non-composite models
inline_resonator_full.guess = types.MethodType(inline_resonator_full_guess, inline_resonator_full)


def inline_fit(paramsVec, res, residual=True, **kwargs):
    """Wrapper function to make compatible with scraps interface.

    Will deprecate in next major version of scraps."""

    s21_concat = np.concatenate([res.I, res.Q], axis=0)
    freqs = res.freq

    if res.sigmaI is not None and res.sigmaQ is not None:
        sigma_i = res.sigmaI
        sigma_q = res.sigmaQ
    else:
        sigma_i = np.ones_like(freqs)
        sigma_q = np.ones_like(freqs)

    sigma_concat = np.concatenate([sigma_i, sigma_q], axis=0)

    model = inline_resonator_full.eval(params=paramsVec, freqs=freqs, **kwargs)

    model_concat = np.concatenate([np.real(model), np.imag(model)], axis=0)

    if residual:
        return (model_concat - s21_concat) / sigma_concat
    else:
        return model_concat


def inline_params(res, **kwargs):
    """Wrapper function to make compatible with scraps interface.

    Will deprecate in next major version of scraps."""
    s21 = res.I + 1j * res.Q
    freqs = res.freq

    return inline_resonator_full.guess(data=s21, freqs=freqs, **kwargs)
