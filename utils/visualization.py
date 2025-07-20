
### Visualization ###

# todo: make plotting big signals faster, improve complex signal handling
# todo: make master function like visualize(signal, plots=("time", "fft", "constellation")) for quick subplots
# todo: Also could make plot_together(plot1, plot2) type shi

import numpy as np
import matplotlib.pyplot as plt
import scipy


def plot_signal(*signals, n_samps=100, ylabel=None, xlabel="n", title='Signal', label=[],
                xlim=None, ylim=None, ax=None, x=None):
    if len(signals) == 0:
        raise ValueError("At least one signal must be provided.")
    
    # User must input one ylabel for every inputted signal
    label = np.asarray(label)
    if len(label) != len(signals) and len(label) != 0:
        raise ValueError("Must have equal number of signals and ylabels.")

    # Determine the default x-axis (independent variable)
    if x is not None:
        x = np.asarray(x)
        if xlim is None:
            xlim = [x[0], x[min(n_samps, len(x)) - 1]]
        # Apply xlim to find corresponding indices
        start_idx = np.searchsorted(x, xlim[0], side='left')
        stop_idx = np.searchsorted(x, xlim[1], side='right')
        x = x[start_idx:stop_idx]
        signals = [np.asarray(s)[start_idx:stop_idx] for s in signals]
    else:
        # Use default index-based x
        if xlim is None:
            if n_samps > len(signals[0]):
                n_samps = len(signals[0])
            xlim = [0, n_samps - 1]
        start, stop = xlim[0], xlim[1] + 1
        x = np.arange(start, stop)
        signals = [np.asarray(s)[start:stop] for s in signals]

    # Flatten signals into vectors
    signals = [np.asarray(s).flatten() for s in signals]

    # Axes object can be passed in, which allows figures to be plotted in subplots
    if ax is None:
        fig, ax = plt.subplots()
        show = True
    else:
        show = False

    # Add all signals to plot
    if len(label) == 0:
        label = [f'Signal {i+1}' for i in range(len(signals))]
    for i, s in enumerate(signals):
        ax.plot(x, s, '.-', label=label[i])

    # Decorate plot
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(True)
    if len(signals) > 1:
        ax.legend()
    if show:
        fig.show()

    # if fig is not None: return fig

# Plot signal constellation diagram
def plot_constellation(signal, n_samples=1000, ax=None, title="Constellation Plot"):
    """
    Display the constellation diagram of a signal
    """

    # axes object can be passed in, which allows figures to be plotted in subplots
    if ax is None:
        fig, ax = plt.subplots()
        show = True
    else:
        show = False

    # truncate to just visible signal values
    signal = signal[:n_samples]
    
    ax.set_title(title)
    ax.set_xlabel("In Phase")
    ax.set_ylabel("Quadrature")
    ax.margins(x=0.5, y=0.5)
    ax.plot(np.real(signal), np.imag(signal), '.')
    ax.grid(True)
    if show: fig.show()

# Plot frequency spectrum of signal
def plot_spectrum(signal, size=None, n_samples=None, Fs=1.0, window='hann', db=True, title='Spectrum', xlim=None, ylim=None, ax=None, dec_factor=None):
    """
    Plot the frequecy spectrum of a signal1

    signal: np array signal vector
    Fs: sampling rate for frequency axis values
    size: size of fft. Must be a power of 2. default: minimum pow of 2 greater than len(signal)
    window: 'hann', 'blackman', or rectangular (default) window
    db: use db scale (default linear)
    title, xlim, ylim: corresponding property of matplotlib plot
    """

    # axes object can be passed in, which allows figures to be plotted in subplots
    if ax is None:
        fig, ax = plt.subplots()
        show = True
    else:
        show = False
    
    if size is None:
        size = 2 ** int(np.ceil(np.log2(len(signal))))

    if n_samples:
        signal = signal[:n_samples]
    
    if window == 'hann':
        win = np.hanning(len(signal))
    elif window == 'blackman':
        win = np.blackman(len(signal))
    else:
        win = np.ones(len(signal)) # Rectangular window

    # apply windowing
    sig_win = signal * win

    # compute fft
    spec = np.fft.fftshift(np.fft.fft(sig_win, n=size))
    freqs = np.fft.fftshift(np.fft.fftfreq(size, 1/Fs))

    # decimate
    if dec_factor:
        spec = scipy.signal.decimate(spec, dec_factor)
        freqs = scipy.signal.decimate(freqs, dec_factor)
    
    # convert from linear to dB scale
    spec = np.abs(spec)
    if db:
        spec = 20 * np.log10(spec + 1e-12)
    
    # Display spectrum
    ax.plot(freqs, spec)
    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(True)
    if show: fig.show()


