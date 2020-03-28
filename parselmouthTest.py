import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import parselmouth
import seaborn as sns

def draw_spectrogram(spectrogram, dynamic_range=70):
    X, Y = spectrogram.x_grid(), spectrogram.y_grid()
    sg_db = 10 * np.log10(spectrogram.values)
    plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap='afmhot')
    plt.ylim([spectrogram.ymin, spectrogram.ymax])
    plt.xlabel("time [s]")
    plt.ylabel("frequency [Hz]")

def draw_intensity(intensity):
    plt.plot(intensity.xs(), intensity.values.T, linewidth=3, color='w')
    plt.plot(intensity.xs(), intensity.values.T, linewidth=1)
    plt.grid(False)
    plt.ylim(0)
    plt.ylabel("intensity [dB]")

def draw_mfcc(mfcc_values):
    mfcc_sub = plt.subplot()
    cax = mfcc_sub.imshow(mfcc_values, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
    mfcc_sub.set_title('MFCC')
    plt.colorbar(cax)


if __name__ == "__main__":

    sns.set() # Uses seaborn's default style

    snd = parselmouth.Sound("testing.wav")

    # Plotting waveform.
    plt.figure()
    waveform = plt.subplot(211)
    plt.plot(snd.xs(), snd.values.T)
    plt.xlim([snd.xmin, snd.xmax])
    plt.xlabel("time [s]")
    plt.ylabel("amplitude")

    # Plotting spectrogram
    spectro = plt.subplot(212)
    intensity = snd.to_intensity()
    spectrogram = snd.to_spectrogram()
    draw_spectrogram(spectrogram)
    plt.twinx()
    draw_intensity(intensity)
    plt.xlim([snd.xmin, snd.xmax])
    plt.show()  # or plt.savefig("spectrogram.pdf")

    # Plotting MFCCS
    mfcc = snd.to_mfcc()
    mfcc_values = mfcc.to_array()
    print(mfcc_values)
    draw_mfcc(mfcc_values)
    plt.show()

