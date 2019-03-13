import types
import numpy as np
import lmfit as lf
from ..utils import mask_array, reduce_by_midpoint


def inverted_assymmetric_lorentzian(freqs, df, f0, qc, qi):
    #Reference a unitless reduced frequency
    fs = f0+df
    ff = (freqs-fs)/fs

    #Calculate the total Q_0
    q0 = 1./(1./qi+1./qc)

    return (1./qi+1j*2.0*(ff+df/fs))/(1./q0+1j*2.0*ff)

def phase_gain(freqs, pgain0, pgain1, pgain2):
    ffm = reduce_by_midpoint(freqs)

    return np.exp(1j*(pgain0 + pgain1*ffm + pgain2*ffm**2))


def magnitude_gain(freqs, gain0, gain1, gain2):
    ffm = reduce_by_midpoint(freqs)

    return gain0 + gain1*ffm + gain2*ffm**2


class Model_inverted_assymmetric_lorentzian(lf.Model):
    def __init__(self, independent_vars=['freqs'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        super(Model_inverted_assymmetric_lorentzian, self).__init__(inverted_assymmetric_lorentzian, **kwargs)

    def guess(self, data, freqs, **kwargs):
        pre_filter = kwargs.pop('pre_filter', False)
        #Assuming data is complex valued vector of I+1j*Q
        magnitude = np.abs(data)

        ix_min = np.argmin(magnitude)
        f0_guess = freqs[ix_min]
        mag_min = magnitude[ix_min]
        mag_max = 1 #assuming baseline has been removed

        #Guess the Q values:
        #1/Q0 = 1/Qc + 1/Qi
        #Q0 = f0/fwhm bandwidth
        #Q0/Qi = min(mag)/max(mag)
        fwhm = np.sqrt((mag_max**2 + mag_min**2)/2.)
        fwhm_mask = magnitude < fwhm
        bandwidth = freqs[fwhm_mask][-1]-freqs[fwhm_mask][0]
        q0_guess = f0_guess/bandwidth

        qi_guess = q0_guess*mag_max/mag_min

        qc_guess = 1./(1./q0_guess-1./qi_guess)

        self.set_param_hint('f0', min=freqs[0], max=freqs[-1])
        self.set_param_hint('qi', min=0)
        self.set_param_hint('qc', min=0)

        params = self.make_params(df=0, f0=f0_guess, qc=qc_guess, qi=qi_guess)

        return lf.models.update_param_vals(params, self.prefix, **kwargs)


class Model_magnitude_gain(lf.Model):
    def __init__(self, independent_vars=['freqs'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        super(Model_magnitude_gain, self).__init__(magnitude_gain, **kwargs)

    def guess(self, data, freqs, mask=0.05, **kwargs):
        """Mask is an number for how many points to keep on each end. If int, then assumes
        number of points, if float then assumes percent of data. If None, keeps everything."""

        ffm = reduce_by_midpoint(freqs)

        masked_data = mask_array(data, mask)
        masked_ffm = mask_array(ffm, mask)

        gain2_guess, gain1_guess, gain0_guess = np.polyfit(masked_ffm, np.abs(masked_data), 2)

        params = self.make_params(gain2=gain2_guess, gain1=gain1_guess, gain0=gain0_guess)

        return lf.models.update_param_vals(params, self.prefix, **kwargs)


class Model_phase_gain(lf.Model):
    def __init__(self, independent_vars=['freqs'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        super(Model_phase_gain, self).__init__(phase_gain, **kwargs)

    def guess(self, data, freqs, mask=0.05, **kwargs):
        ffm = reduce_by_midpoint(freqs)

        masked_data = mask_array(data, mask)
        masked_ffm = mask_array(ffm, mask)

        pgain2_guess, pgain1_guess, pgain0_guess = np.polyfit(masked_ffm, np.unwrap(np.angle(masked_data)), 2)

        params = self.make_params(pgain2=pgain2_guess, pgain1=pgain1_guess, pgain0=pgain0_guess)

        return lf.models.update_param_vals(params, self.prefix, **kwargs)

#Make up a composite model for a typical hanger resonator
model_complex_baseline = Model_magnitude_gain()*Model_phase_gain()
model_hanger_resonator = Model_inverted_assymmetric_lorentzian()*model_complex_baseline

def hanger_resonator_guess(self, data, freqs, mag_mask=0.05, phase_mask=0.05, **kwargs):
    #Grab best guess for baseline params
    #Left = inv assym lorentzian
    #Right = mag gain * phase gain
    gain_params = self.right.left.guess(data, freqs, mag_mask, **kwargs)
    phase_params = self.right.right.guess(data, freqs, phase_mask, **kwargs)

    #Calculate the baseline
    mag_baseline_guess = self.right.left.eval(gain_params, freqs=freqs)
    phase_baseline_guess = self.right.right.eval(phase_params, freqs=freqs)

    #Try and make a best guess for clean data
    reduced_data = data/(mag_baseline_guess*phase_baseline_guess)

    #Calculate baseline params from best-guess clean data
    ial_params = self.left.guess(reduced_data, freqs, **kwargs)

    #Final params is all of the params concatenated
    params = gain_params + phase_params + ial_params

    return lf.models.update_param_vals(params, '', **kwargs)

#Attach this to the model instance so it behaves like the non-composite models
model_hanger_resonator.guess = types.MethodType(hanger_resonator_guess, model_hanger_resonator)