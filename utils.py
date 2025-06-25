
import numpy as np
import matplotlib.pyplot as plt
import scipy

# visualization (spectrum, constillation, BER vs SNR) and RRC filtering
# todo: make plotting big signals faster, improve complex signal handling

### Visualization ###


def plot_signal(*signals, n=None, ylabel=None, xlabel="n", title='Signal', xlim=None, ylim=None, ax=None):
    if len(signals) == 0:
        raise ValueError("At least one signal must be provided.")

    # truncate signals to just visible values
    if xlim:
        start, stop = xlim[0], xlim[1]+1
        signals = [np.asarray(s[start:stop]) for s in signals]
        n = np.arange(start, stop)
        
    # flatten signals into vectors
    signals = [np.asarray(s).flatten() for s in signals]

    # axes object can be passed in, which allows figures to be plotted in subplots
    if ax is None:
        fig, ax = plt.subplots()
        show = True
    else:
        show = False
    
    # add all signals to plot
    for i, s in enumerate(signals): 
        if n is not None: ax.plot(n, np.asarray(s).flatten(), '.-', label=f'Signal {i+1}')
        else: ax.plot(np.asarray(s).flatten(), '.-', label=f'Signal {i+1}')

    # decorate plot
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(True)
    if len(signals) > 1:
        ax.legend([f"Signal {i+1}" for i in range(len(signals))])
    if show: fig.show()

# Plot signal constellation diagram
def plot_constellation(signal, n_samples=10, ax=None, title="Constellation Plot"):
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
    Plot the frequecy spectrum of a signal

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




### Filtering ###


# Generate root-raise cosine filter coefficients
def rrc_coef(n_taps=101, beta=0.35, Ts=1.0):

    # initialize vectors
    h = np.zeros(n_taps, dtype=complex)
    t_vec = np.arange(n_taps)  - (n_taps-1)//2 # -50, -49, ..., 49, 50
    
    for i, t in enumerate(t_vec):  
        # Piecewise definition from https://en.wikipedia.org/wiki/Root-raised-cosine_filter
        
        # t = 0:
        if t == 0:
            h[i] = 1/Ts * (1 + beta*(4/np.pi - 1))
            continue
    
        # t = Ts/(4*beta): 
        if abs(t) == Ts/(4*beta):
            h[i] = beta/(Ts*np.sqrt(2)) * ( (1 + 2/np.pi)*np.sin(np.pi/(4*beta)) + \
                                           (1 - 2/np.pi)*np.cos(np.pi/(4*beta)) )
            continue
    
        # otherwise
        h[i] = 1/Ts * (np.sin(np.pi*(t/Ts)*(1-beta)) + 4*beta*(t/Ts)*np.cos(np.pi*(t/Ts)*(1+beta))) / \
                      (np.pi*(t/Ts)*(1 - (4*beta*(t/Ts))**2))
    return h


