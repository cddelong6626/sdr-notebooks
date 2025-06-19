
import numpy as np
import matplotlib.pyplot as plt

# visualization (spectrum, constillation, BER vs SNR) and RRC filtering
# todo: make plotting big signals faster, improve complex signal handling

### Visualization ###


def plot_signal(*signals, n=None, ylabel=None, xlabel="n", title='Signal', xlim=None, ylim=None):
    signals = [np.asarray(s).flatten() for s in signals]
        
    plt.figure()
    for i, s in enumerate(signals): 
        if n: plt.plot(n, np.asarray(s).flatten(), '.-', label=f'Signal {i+1}')
        else: plt.plot(np.asarray(s).flatten(), '.-', label=f'Signal {i+1}')

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.grid(True)
    if len(signals) > 1:
        plt.legend([f"Signal {i+1}" for i in range(len(signals))])
    plt.show()

# Plot frequency spectrum of signal
def plot_spectrum(signal, size=None, Fs=1.0, window='hann', db=True, title='Spectrum', xlim=None, ylim=None):
    """
    Plot the frequecy spectrum of a signal

    signal: np array signal vector
    Fs: sampling rate for frequency axis values
    size: size of fft. Must be a power of 2. default: minimum pow of 2 greater than len(signal)
    window: 'hann', 'blackman', or rectangular (default) window
    db: use db scale (default linear)
    title, xlim, ylim: corresponding property of matplotlib plot
    """
    
    if size is None:
        size = 2 ** int(np.ceil(np.log2(len(signal))))

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

    # convert from linear to dB scale
    spec = np.abs(spec)
    if db:
        spec = 20 * np.log10(spec + 1e-12)
    
    # Display spectrum
    plt.figure()
    plt.plot(freqs, spec)
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.grid(True)
    plt.show()



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


