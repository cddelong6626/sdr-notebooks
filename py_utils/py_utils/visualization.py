import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

### Time Domain ###

def plot_signal(*signals, n_samps=None, ylabel=None, xlabel="n", title="Signal",
                label=None, xlim=None, ylim=None, ax=None, x=None, show_parts=True, db=False):
    """
    Plot one or more signals in the time domain.
    If a signal is complex and show_parts=True, the real and imaginary
    parts are plotted as separate traces.
    Optionally plot magnitude in dB.

    Parameters
    ----------
    *signals : array_like
        One or more input signals (real or complex).
    n_samps : int, optional
        Number of samples to display. If None or larger than the signal,
        the full signal is shown. Default is 300.
    ylabel : str, optional
        Label for the y-axis. Default is None.
    xlabel : str, optional
        Label for the x-axis. Default is "n".
    title : str, optional
        Plot title. Default is "Signal".
    label : list of str, optional
        Labels for each signal. If None, defaults to "Signal 1", "Signal 2", etc.
    xlim : tuple of (float, float), optional
        Limits for the x-axis. If None, inferred from sample range.
    ylim : tuple of (float, float), optional
        Limits for the y-axis. If None, autoscaled.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on. If None, a new figure and axes are created.
    x : array_like, optional
        Custom x-axis values corresponding to the signal samples. If None,
        sample indices are used.
    show_parts : bool, optional
        If True, complex signals are split into real and imaginary parts.
        Default is True.
    db : bool, optional
        If True, plot magnitude in dB (20*log10(abs(signal))). Default is False.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot.
    ax : matplotlib.axes.Axes
        The axes with the plotted signals.
    """
    if len(signals) == 0:
        raise ValueError("At least one signal must be provided.")

    signals = [np.asarray(s).flatten() for s in signals]

    # Expand complex signals if requested
    expanded_signals = []
    expanded_labels = []
    for idx, s in enumerate(signals):
        lbl = label[idx] if label and idx < len(label) else f"Signal {idx+1}"
        if db:
            s_db = 20 * np.log10(np.abs(s) + 1e-12)
            expanded_signals.append(s_db)
            expanded_labels.append(lbl + " (dB)")
        elif np.iscomplexobj(s) and show_parts:
            expanded_signals.append(s.real)
            expanded_labels.append(lbl + " (Re)")
            expanded_signals.append(s.imag)
            expanded_labels.append(lbl + " (Im)")
        else:
            expanded_signals.append(s)
            expanded_labels.append(lbl)
    signals = expanded_signals
    label = expanded_labels

    if n_samps is None or n_samps > len(signals[0]):
        n_samps = len(signals[0])

    if x is not None:
        x = np.asarray(x)
        if xlim is None:
            xlim = [x[0], x[min(n_samps, len(x))-1]]
        start_idx = np.searchsorted(x, xlim[0], side="left")
        stop_idx = np.searchsorted(x, xlim[1], side="right")
        x = x[start_idx:stop_idx]
        signals = [s[start_idx:stop_idx] for s in signals]
    else:
        if xlim is None:
            xlim = [0, n_samps-1]
        start, stop = xlim[0], xlim[1]+1
        x = np.arange(start, stop)
        signals = [s[start:stop] for s in signals]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    for s, lbl in zip(signals, label):
        ax.plot(x, s, '.-', label=lbl)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel if ylabel else ("Magnitude (dB)" if db else None))
    ax.grid(True)
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    if len(signals) > 1:
        ax.legend()

    return fig, ax


### Frequency Domain ###

def plot_spectrum(signal, size=None, n_samples=None, Fs=None,
                  window='hann', db=True, title='Spectrum',
                  xlim=None, ylim=None, ax=None, dec_factor=None):
    """
    Plot the frequency spectrum of a signal using the FFT.
    Handles both real and complex signals, with optional windowing
    and decimation.

    Parameters
    ----------
    signal : array_like
        Input signal to transform.
    size : int, optional
        FFT size. If None, next power of two >= len(signal) is used.
    n_samples : int, optional
        Number of samples from the signal to use. If None, full length.
    Fs : float, optional
        Sampling frequency in Hz. If None, frequency axis is normalized
        (cycles/sample).
    window : {"hann", "blackman", None}, optional
        Window function to apply before FFT. Default is 'hann'.
    db : bool, optional
        If True, plot magnitude in dB. Otherwise, plot linear magnitude.
        Default is True.
    title : str, optional
        Plot title. Default is "Spectrum".
    xlim : tuple of (float, float), optional
        Limits for the frequency axis.
    ylim : tuple of (float, float), optional
        Limits for the magnitude axis.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on. If None, a new figure and axes are created.
    dec_factor : int, optional
        If provided, decimate spectrum and frequency axis by this factor.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot.
    ax : matplotlib.axes.Axes
        The axes with the plotted spectrum.
    """
    signal = np.asarray(signal)

    if n_samples:
        signal = signal[:n_samples]

    if size is None:
        size = 2 ** int(np.ceil(np.log2(len(signal))))

    if window == 'hann':
        win = np.hanning(len(signal))
    elif window == 'blackman':
        win = np.blackman(len(signal))
    else:
        win = np.ones(len(signal))

    sig_win = signal * win

    spec = np.fft.fftshift(np.fft.fft(sig_win, n=size))

    if Fs is not None:
        freqs = np.fft.fftshift(np.fft.fftfreq(size, 1/Fs))
        freq_label = "Frequency (Hz)"
    else:
        freqs = np.fft.fftshift(np.fft.fftfreq(size))
        freq_label = "Normalized Frequency (cycles/sample)"

    if dec_factor:
        spec = scipy.signal.decimate(spec, dec_factor)
        freqs = scipy.signal.decimate(freqs, dec_factor)

    spec_mag = np.abs(spec)

    if db:
        spec_mag = 20*np.log10(spec_mag + 1e-12)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.plot(freqs, spec_mag)
    ax.set_title(title)
    ax.set_xlabel(freq_label)
    ax.set_ylabel("Magnitude (dB)" if db else "Magnitude")
    ax.grid(True)
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)

    return fig, ax


### Modulation ###

def plot_constellation(signal, n_samples=1000, offset=0, ax=None, title="Constellation Plot", xlim=None):
    """
    Plot a constellation diagram from a complex-valued signal.

    Parameters
    ----------
    signal : array_like
        Complex input signal representing modulated symbols.
    n_samples : int, optional
        Maximum number of points to display. Default is 1000.
    offset : int, optional
        Starting index for the signal. Default is 0.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on. If None, a new figure and axes are created.
    title : str, optional
        Plot title. Default is "Constellation Plot".
    xlim : tuple of (int, int), optional
        Indices of the signal to plot: (start, stop). If None, uses offset and n_samples.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot.
    ax : matplotlib.axes.Axes
        The axes with the plotted constellation diagram.
    """
    signal = np.asarray(signal)
    if not np.iscomplexobj(signal):
        raise ValueError("Input signal must be complex for constellation plot.")

    if xlim is not None:
        start, stop = xlim
        signal = signal[start:stop]
    else:
        signal = signal[offset:offset+n_samples]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.plot(np.real(signal), np.imag(signal), '.')
    ax.set_title(title)
    ax.set_xlabel("In Phase")
    ax.set_ylabel("Quadrature")
    ax.margins(x=0.5, y=0.5)
    ax.grid(True)

    return fig, ax


def visualize(signal, 
              plots=("time", "fft", "constellation"), 
              Fs=None, 
              plot_kwargs=None):
    """
    Convenience wrapper to quickly visualize a signal in multiple domains.
    Handles complex signals appropriately.

    Parameters
    ----------
    signal : array_like
        The input signal (real or complex).
    plots : tuple of {"time", "fft", "constellation"}
        Which plots to include.
    Fs : float, optional
        Sampling frequency, required for FFT axis scaling.
    plot_kwargs : dict, optional
        Dictionary of keyword argument dictionaries for each plot type.
        Example:
            {
                "time": {"xlabel": "Sample Index", "title": "My Time Plot"},
                "fft": {"window": "blackman", "title": "Spectrum"},
                "constellation": {"n_samples": 500}
            }

    Returns
    -------
    fig : matplotlib.figure.Figure
        The combined figure containing the subplots.
    axes : dict
        Dictionary mapping plot names to their Axes objects.
    """
    if plot_kwargs is None:
        plot_kwargs = {}

    nplots = len(plots)
    fig, axs = plt.subplots(1, nplots, figsize=(5 * nplots, 4))
    if nplots == 1:
        axs = [axs]  # ensure iterable

    axes = {}
    for ax, name in zip(axs, plots):
        kwargs = plot_kwargs.get(name, {})
        if name == "time":
            _, axes[name] = plot_signal(signal, ax=ax, **kwargs)
        elif name == "frequency" or name == "fft" or name == "spectrum":
            _, axes[name] = plot_spectrum(signal, Fs=Fs, ax=ax, **kwargs)
        elif name == "constellation" and np.iscomplexobj(signal):
            _, axes[name] = plot_constellation(signal, ax=ax, **kwargs)

    fig.tight_layout()
    return fig, axes