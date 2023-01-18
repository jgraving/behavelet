# -*- coding: utf-8 -*-
"""
Copyright 2021 Jacob M. Graving <jgraving@gmail.com>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np

LOG_PI = np.log(np.pi)
LOG2 = np.log(2)


def logsumexp(x, axis=None, keepdims=False):
    """
    Calculates the log of the sum of the exponentiated elements of an input array.

    This function is useful for avoiding numerical underflow or overflow when calculating the sum of a large number of exponential values.

    Parameters:
    -----------
    x : numpy.ndarray
        Input array.
    axis : int or None (default = None)
        The axis along which to perform the sum. If None, the sum is taken over all elements of the array.
    keepdims : bool (default = False)
        If True, the output will have the same number of dimensions as the input, with the specified axis having size 1. If False, the output will have one fewer dimensions than the input.

    Returns:
    --------
    logsumexp : float
        The log of the sum of the exponentiated elements of the input array.
    """
    max_x = np.max(x, axis=axis, keepdims=True)
    result = max_x + np.log(np.sum(np.exp(x - max_x), axis=axis, keepdims=True))
    if not keepdims:
        result = np.squeeze(result, axis=axis)
    return result


def morlet_conj_ft(omegas, omega0=5.0):
    """
    Returns a Fourier conjugate Morlet wavelet.

    The time-frequency trade-off in the complex Morlet wavelet transform is controlled by the omega0 and omegas parameters. A larger value of omega0 corresponds to a higher time resolution and a lower frequency resolution, while a larger value of omegas corresponds to a lower time resolution and a higher frequency resolution.

    The wavelet is defined as:

        pi**(-0.25) * exp(-0.5 * (omegas - omega0)**2)
        = exp((-0.25 * log(pi)) + (-0.5 * (omegas - omega0)**2))

    Parameters:
    ===========
    omegas : numpy.ndarray
        Dimensionless parameter that controls the width of the frequency band of the wavelet. It is related to the standard deviation of the wavelet's frequency distribution. A larger value of omegas corresponds to a wavelet with a wider frequency band and a lower central frequency.
    omega0 : float (default=5.0)
        Dimensionless parameter that determines the central frequency of the wavelet. It is related to the number of oscillations that the wavelet undergoes within a fixed number of standard deviations from its center. A larger value of omega0 corresponds to a wavelet with a higher central frequency and a narrower frequency band.

    Returns:
    ========
    ft_wavelet : numpy.ndarray
        Array of Fourier conjugate wavelets.
    """
    # (
    #    (np.pi**-0.25)
    #    * torch.exp(-0.5 * (omegas - omega0)**2)
    # )
    return np.exp((-0.25 * LOG_PI) + (-0.5 * (omegas - omega0) ** 2))


def morlet_fft_convolution(
    x, log_scales, sample_interval, unpadding, density=True, omega0=5.0
):
    """
    Calculates a Morlet continuous wavelet transform for a given signal across a range of frequencies.

    The time-frequency trade-off in the complex Morlet wavelet transform is controlled by the omega0 and omegas parameters. A larger value of omega0 corresponds to a higher time resolution and a lower frequency resolution, while a larger value of omegas corresponds to a lower time resolution and a higher frequency resolution.

    This implementation adjusts the output for disproportionately large wavelet response at low frequencies and removes amplitude fluctuations by normalizing the power spectrum.

    Parameters:
    ===========
    x : numpy.ndarray, shape (batch, channels, sequence)
        A batch of multichannel sequences.
    log_scales : numpy.ndarray, shape (1, 1, freqs, 1)
        A tensor of logarithmic scales.
    sample_interval : float
        Change in time per sample. The inverse of the sampling frequency.
    unpadding : int
        Amount of extra padding to remove, i.e. to return the valid part of the spectrogram without edge effects.
    density : bool (default = True)
        Whether to normalize so the power spectrum sums to one. This effectively removes amplitude fluctuations.
    omega0 : float
        Dimensionless parameter that determines the central frequency of the wavelet. It is related to the number of oscillations that the wavelet undergoes within a fixed number of standard deviations from its center. A larger value of omega0 corresponds to a wavelet with a higher central frequency and a narrower frequency band.

    Returns:
    ========
    out : numpy.ndarray, shape (batch, channels, freqs, sequence)
        The transformed signal.
    """
    n_sequence = x.shape[-1]

    # Pad with extra zero if needed
    pad_sequence = n_sequence % 2 != 0
    x = np.pad(x, ((0, 0), (0, 0), (0, 1))) if pad_sequence else x

    # Set index to remove the extra zero if added
    idx0 = (n_sequence // 2) + unpadding
    idx1 = (
        (n_sequence // 2) + n_sequence - unpadding - 1
        if pad_sequence
        else (n_sequence // 2) + n_sequence - unpadding
    )
    x = np.pad(x, ((0, 0), (0, 0), (n_sequence // 2, n_sequence // 2)))

    # (batch, channels, sequence) -> (batch, channels, freqs, sequence)
    x = x[:, :, np.newaxis]

    # Calculate the omega values
    n_padded = x.shape[-1]
    omegas = (
        -2
        * np.pi
        * np.arange(-n_padded // 2, n_padded // 2)
        / (n_padded * sample_interval)
    )[
        None, None, None
    ]  # (sequence,) -> (batch, channels, freqs, sequence)

    # Fourier transform the padded signal
    x_hat = np.fft.fftshift(np.fft.fft(x, axis=-1))

    # Calculate the wavelets
    morlet = morlet_conj_ft(omegas * np.exp(log_scales), omega0)

    # Perform the wavelet transform
    convolved = np.fft.ifft(morlet * x_hat, axis=-1)[..., idx0:idx1] * np.exp(
        log_scales * 0.5
    )

    power = np.abs(convolved)

    # scale power to account for disproportionally
    # large response at low frequencies
    # power_scale = (
    #    np.pi ** -0.25
    #    * np.exp(0.25 * (omega0 - np.sqrt(omega0 ** 2 + 2)) ** 2)
    #    / scales.mul(2).sqrt()
    # )
    log_power_scale = (
        -0.25 * np.log(np.pi)
        + 0.25 * (omega0 - np.sqrt(omega0 ** 2 + 2)) ** 2
        - log_scales
        + np.log(2) / 2
    )
    log_power_scaled = np.log(power) + log_power_scale
    log_total_power = logsumexp(log_power_scaled, axis=(1, 2), keepdims=True)
    log_density = log_power_scaled - log_total_power
    return np.exp(log_density)


class BermanWavelet:
    """
    Applies a Morlet continuous wavelet transform to a batch of multi-channel sequences across a range of frequencies.

    The time-frequency trade-off in the complex Morlet wavelet transform is controlled by the omega0 and omegas parameters. A larger value of omega0 corresponds to a higher time resolution and a lower frequency resolution, while a larger value of omegas corresponds to a lower time resolution and a higher frequency resolution. The value of omegas is derived from the values of fmin, fmax, and n_freqs, which control the range and resolution of the frequencies considered in the transform.

    This implementation from Berman et al. [1] adjusts the output for disproportionately large wavelet response at low frequencies and removes amplitude fluctuations by normalizing the power spectrum.

    Parameters:
    ===========
    fsample : float
        Sampling frequency of the data (in Hz).
    fmin : float
        Minimum frequency of interest for the wavelet transform (in Hz).
    fmax : float
        Maximum frequency of interest for the wavelet transform (in Hz). Typically the Nyquist frequency of the signal (fsample/2).
    n_freqs : int
        Number of frequencies to consider from fmin to fmax (inclusive).
    density : bool (default = True)
        Whether to normalize the power so the power spectrum sums to one. This effectively removes amplitude fluctuations.
    padding : str (default = "same")
        Controls the padding applied to the input. Must be "valid" or "same" which returns the valid part of the signal or full length signal respectively.
    omega0 : float (default = 5.0)
        Dimensionless parameter that determines the central frequency of the wavelet. It is related to the number of oscillations that the wavelet undergoes within a fixed number of standard deviations from its center. A larger value of omega0 corresponds to a wavelet with a higher central frequency and a narrower frequency band.

    Inputs:
    ===========
    x : numpy.ndarray, shape (batch, channels, sequence)

    Outputs:
    ========
    out : numpy.ndarray, shape (batch, channels, freqs, sequence)

    References:
    ===========
    [1] Berman, G. J., Choi, D. M., Bialek, W., & Shaevitz, J. W. (2014). Mapping the stereotyped behaviour of freely moving fruit flies. Journal of The Royal Society Interface, 11(99), 20140672.

    Notes:
    ======
    Based on code from Gordon J. Berman et al. (https://github.com/gordonberman/MotionMapper)
    """

    def __init__(
        self,
        fsample,
        fmin,
        fmax,
        n_freqs,
        padding="same",
        density=True,
        omega0=5.0,
    ):
        super().__init__()
        self.fsample = fsample
        self.sample_interval = 1 / fsample
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else fsample / 2
        self.n_freqs = n_freqs
        self.padding = padding
        self.density = density
        self.omega0 = omega0
        self.output_sequence = output_sequence
        self.freqs = np.logspace(
            np.log(self.fmin) / LOG2,
            np.log(self.fmax) / LOG2,
            self.n_freqs,
            base=2,
        )[None, None, :, None]
        # (freqs,) -> (batch, channels, freqs, sequence)
        # scales = (omega0 + np.sqrt(2 + omega0 ** 2)) / (4 * np.pi * self.freqs)
        self.log_scales = (
            np.log(self.omega0 + np.sqrt(2 + self.omega0 ** 2))
            - (2 * LOG2)
            - LOG_PI
            - np.log(self.freqs)
        )
        self.unpadding = 0
        if self.padding == "valid":
            # Get the largest scale
            largest_scale = np.max(self.log_scales)
            # Calculate the size of the wavelet function
            # L = 2 * np.pi / np.exp(largest_scale)
            L = np.exp(LOG_PI - largest_scale)
            # Calculate the size of the valid region
            # self.unpadding = int(np.ceil(L / 2))
            self.unpadding = int(np.ceil(L))

    def __call__(self, x):
        return morlet_fft_convolution(
            x,
            self.log_scales,
            self.sample_interval,
            self.unpadding,
            self.density,
            self.omega0,
        )
