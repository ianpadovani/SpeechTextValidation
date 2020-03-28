from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np




if __name__ == "__main__":

# WITHOUT PARSELMOUTH OR SEABORN
    # Reads wav and gets sample rate and frames that will make up waveform.
    sample_rate, signal = wav.read("testing.wav")
    # For plotting, uses the sample rate to generate time values in seconds.
    Time = np.linspace(0, len(signal) / sample_rate, num=len(signal))

    plt.figure(1)
    # Plotting waveform
    waveform = plt.subplot(311)
    waveform.set_xlabel("Time [s]")
    waveform.set_ylabel("Amplitude")
    plt.plot(Time, signal)

    # Plotting Spectrogram
    spectrogram = plt.subplot(312)
    Pxx, freqs, bins, im = spectrogram.specgram(signal, NFFT=1024, Fs=sample_rate, noverlap=1023)
    spectrogram.set_xlabel('Time [s]')
    spectrogram.set_ylabel('Frequency [Hz]')

    # Plotting MFCCs
    mfcc_features = mfcc(signal, sample_rate, winfunc=np.hamming)

    mfcc = plt.subplot(313)
    # Swapping axes so that frame number is on the x-axis and cepstrum is on the y-axis.
    mfcc_data = np.swapaxes(mfcc_features, 0, 1)
    # Applying colour map to MFCC features.
    cax = mfcc.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
    mfcc.set_title('MFCC')
    plt.colorbar(cax)
    plt.tight_layout()
    plt.show()

    fbank_features = logfbank(signal, sample_rate)
