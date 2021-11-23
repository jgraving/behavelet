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
import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

LOG_PI = np.log(np.pi)
LOG2 = np.log(2)


def split(x, num_splits=2, dim=-1):
    return torch.split(x, x.shape[dim] // num_splits, dim=dim)


def fftshift1d(x):
    """
    fft shift along last dim
    """
    n_sequence = x.shape[-1]
    pad_sequence = n_sequence % 2 != 0
    x = F.pad(x, (0, 1)) if pad_sequence else x
    x1, x2 = split(x, 2, dim=-1)
    x2 = x2[..., :-1] if pad_sequence else x2
    return torch.cat((x2, x1), dim=-1)


def morlet_conj_ft(omegas, omega0=5.0):
    """
    Returns a Fourier conjugate Morlet wavelet
    given by the formula:

        pi**(-0.25) * exp(-0.5 * (omegas - omega0)**2)
        = exp((-0.25 * log(pi)) + (-0.5 * (omegas - omega0)**2))

    Parameters:
    ===========
    omegas : torch.Tensor
        omegas to calculate wavelets
    omega0 : float (default=5.0)
        Dimensionless omega0 parameter for wavelet transform

    Returns:
    ========
    ft_wavelet : ndarray
        array of Fourier conjugate wavelets
    """
    # (
    #    (np.pi**-0.25)
    #    * torch.exp(-0.5 * (omegas - omega0)**2)
    # )
    return ((-0.25 * LOG_PI) + (-0.5 * (omegas - omega0) ** 2)).exp()


def morlet_fft_convolution(x, log_scales, dtime, unpadding=0, density=True, omega0=6.0):
    """
    Calculates a Morlet continuous wavelet transform
    for a given signal across a range of frequencies

    Parameters:
    ===========
    x : torch.Tensor, shape (batch, channels, sequence)
        A batch of multichannel sequences
    log_scales : torch.Tensor, shape (1, 1, freqs, 1)
        A tensor of logarithmic scales
    dtime : float
        Change in time per sample. The inverse of the sampling frequency.
    unpadding : int
        The amount of padding to remove from each side of the sequence
    density : bool (default = True)
        Whether to normalize so the power spectrum sums to one.
        This effectively removes amplitude fluctuations.
    omega0 : float
        Dimensionless omega0 parameter for wavelet transform
    Returns:
    ========
    out : torch.Tensor, shape (batch, channels, freqs, sequence)
        The transformed signal.
    """

    n_sequence = x.shape[-1]

    # Pad with extra zero if needed
    pad_sequence = n_sequence % 2 != 0
    x = F.pad(x, (0, 1)) if pad_sequence else x

    # Set index to remove the extra zero if added
    idx0 = (n_sequence // 2) + unpadding
    idx1 = (
        (n_sequence // 2) + n_sequence - unpadding - 1
        if pad_sequence
        else (n_sequence // 2) + n_sequence - unpadding
    )
    x = F.pad(x, (n_sequence // 2, n_sequence // 2))

    # (batch, channels, sequence) -> (batch, channels, freqs, sequence)
    x = x.unsqueeze(-2)

    # Calculate the omega values
    n_padded = x.shape[-1]
    omegas = (
        -2
        * np.pi
        * torch.arange(start=-n_padded // 2, end=n_padded // 2, device=x.device)
        / (n_padded * dtime)
    )[
        None, None, None
    ]  # (sequence,) -> (batch, channels, freqs, sequence)

    # Fourier transform the padded signal
    x_hat = fftshift1d(fft.fft(x, dim=-1))

    # Calculate the wavelets
    morlet = morlet_conj_ft(omegas * log_scales.exp(), omega0)

    # Perform the wavelet transform
    convolved = (
        fft.ifft(morlet * x_hat, dim=-1)[..., idx0:idx1] * log_scales.mul(0.5).exp()
    )

    power = convolved.abs()

    # scale power to account for disproportionally
    # large response at low frequencies
    # power_scale = (
    #    np.pi ** -0.25
    #    * np.exp(0.25 * (omega0 - np.sqrt(omega0 ** 2 + 2)) ** 2)
    #    / scales.mul(2).sqrt()
    # )
    log_power_scale = (
        -0.25 * LOG_PI
        + 0.25 * (omega0 - np.sqrt(omega0 ** 2 + 2)) ** 2
        - log_scales.add(LOG2).mul(0.5)
    )
    log_power_scaled = power.log() + log_power_scale
    log_total_power = log_power_scaled.logsumexp(
        (1, 2), keepdims=True
    )  # (channels, freqs)
    log_density = log_power_scaled - log_total_power
    return log_density.exp()


class BermanWavelet(nn.Module):
    """
    Applies a Morlet continuous wavelet transform to a batch of
    multi-channel sequences across a range of frequencies.

    This is an implementation of the continuous wavelet transform
    described in Berman et al. 2014 [1].

    The output is adjusted for disproportionately large wavelet response
    at low frequencies by scaling the response amplitude to a sine wave
    of the same frequency. Amplitude fluctuations are removed by
    normalizing the power spectrum at each sample.

    Parameters:
    ===========
    fsample : float
        Sampling frequency of the data (in Hz)
    fmin : float
        Minimum frequency of interest for the wavelet transform (in Hz)
    fmax : float
        Maximum frequency of interest for the wavelet transform (in Hz)
        Typically the Nyquist frequency of the signal (0.5 * fsample).
    n_freqs : int
        Number of frequencies to consider from fmin to fmax (inclusive)
    density : bool (default = True)
        Whether to normalize the power so the power spectrum sums to one.
        This effectively removes amplitude fluctuations.
    padding : str (default = "same")
        Controls the padding applied to the input.
        Must be "valid" or "same" which returns the valid part of the
        signal or full length signal respectively
    omega0 : float (default = 5.0)
        Dimensionless omega0 parameter for wavelet transform.

    Inputs:
    ===========
    x : torch.Tensor, shape (batch, channels, sequence)

    Outputs:
    ========
    out : torch.Tensor, shape (batch, channels, freqs, sequence)

    References:
    ===========
    [1] Berman, G. J., Choi, D. M., Bialek, W., & Shaevitz, J. W. (2014).
        Mapping the stereotyped behaviour of freely moving fruit flies.
        Journal of The Royal Society Interface, 11(99), 20140672.

    Notes:
    ======
    Based on code from Gordon J. Berman et al.
    (https://github.com/gordonberman/MotionMapper)
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
        output_sequence=True,
    ):
        super().__init__()
        self.fsample = fsample
        self.dtime = 1 / fsample
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else fsample / 2
        self.n_freqs = n_freqs
        self.padding = padding
        self.valid_padding = int((fsample / fmin) * omega0)
        self.unpadding = self.valid_padding if self.padding == "valid" else 0
        self.density = density
        self.omega0 = omega0
        self.output_sequence = output_sequence
        self.freqs = torch.logspace(
            np.log(self.fmin) / LOG2,
            np.log(self.fmax) / LOG2,
            self.n_freqs,
            base=2,
        )[None, None, :, None]
        # (freqs,) -> (batch, channels, freqs, sequence)
        # scales = (omega0 + np.sqrt(2 + omega0 ** 2)) / (4 * np.pi * self.freqs)
        log_scales = (
            np.log(self.omega0 + np.sqrt(2 + self.omega0 ** 2))
            - (2 * LOG2)
            - LOG_PI
            - self.freqs.log()
        )
        self.register_buffer("log_scales", log_scales)

    @torch.no_grad()
    def forward(self, x):
        out = morlet_fft_convolution(
            x, self.log_scales, self.dtime, self.unpadding, self.density, self.omega0
        )
        return (
            out
            if not self.output_sequence
            else out.reshape(out.shape[0], out.shape[1] * out.shape[2], out.shape[3])
        )
