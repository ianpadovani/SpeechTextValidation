import pandas as pd
import numpy as np
import math
import csv
import scipy.io.wavfile as wav
# from scipy.spatial.distance import pdist, squareform
from scipy.signal import find_peaks
# from python_speech_features import mfcc
from numpy import load
from numpy import matlib
from matplotlib import pyplot as plt
# from matplotlib import cm
import os


def phoneme_boundaries(phn_path, wav_path, frames=True):
    """Takes the path to a .phn file and extracts the initial boundary for each phoneme."""
    phonemes = []
    sample_rate, signal = wav.read(wav_path)

    with open(phn_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            line = line.split("\t")
            phonemes.append(tuple(line))
            if frames:
                # Convert time to equivalent index in MFCC frames.
                boundaries = [round((int(item[1]) / sample_rate) * 100) for item in phonemes]
            else:
                boundaries = [int(item[1]) / sample_rate for item in phonemes]
    return boundaries


def gaussian_kernel_similarity(v1, v2, h=40):
    """Calculates gaussian kernel similarity for two vectors."""
    # h is Gaussian Kernel Width
    euclidean_norm = np.linalg.norm(v1-v2, 2)
    weighted = -(euclidean_norm**2/h**2)
    similarity = np.exp(weighted)
    return similarity
    # return np.exp(-np.linalg.norm(v1 - v2, 2) ** 2 / (2. * 1 ** 2))


def epsilon_neighbourhood(v1, v2, epsilon, h=40):
    """Returns True if the two frames are in the same neighbourhood and False otherwise."""
    if 1-gaussian_kernel_similarity(v1, v2, h) < epsilon:
        return True
    else:
        return False


def reachable(matrix, i, epsilon, h=40, k=5, w=50):
    """Receives a matrix, the current frame and the epsilon threshold for the current frame.
    Returns the index of the last reachable frame for the current frame."""
    count = 0
    p_boundary = i
    for j in range(i+1, min(i+w, len(frames))):
        # Check each vector with the current frame's.
        # If not neighbours add to count, set p_boundary to the index of last neighbour.
        # If neighbours, reset count and p_boundary.
        if epsilon_neighbourhood(matrix[i], matrix[j], epsilon, h):
            count = 0
            p_boundary = j
        else:
            count += 1

        # If count is k, set boundary.
        if count == k:
            return p_boundary

    # If no boundary found by end, set last possible frame as boundary.
    return min(i+w, len(frames)-1)


# ------------------- PLOTTING ------------------- #
def plot_kernel_gram_boundaries(matrix, boundaries, minima):
    """Plots a side-by-side comparison of the real boundaries and predicted boundaries over the kernel-gram matrix."""
    # Sets axes to start at 0.
    plt.rcParams['axes.xmargin'] = 0
    plt.rcParams['axes.ymargin'] = 0
    plt.figure(1)

    # Set up left plot with real boundaries.
    real_boundaries = plt.subplot(121)
    real_boundaries.imshow(matrix, cmap="Greys")

    for line in boundaries:
        real_boundaries.axvline(line, c="r")

    # Set up right plot with predicted boundaries.
    predicted_boundaries = plt.subplot(122)
    predicted_boundaries.imshow(matrix, cmap="Greys")

    for minimum in minima:
        predicted_boundaries.axvline(minimum, c="r")
    plt.show()


def plot_spectrogram_comparison(wav_path, boundaries, minima):
    """Plots a side-by-side comparison of the real boundaries and predicted boundaries over the kernel-gram matrix."""
    # Reads wav and gets sample rate and frames that will make up waveform.
    sample_rate, signal = wav.read(wav_path)

    # Sets axes to start at 0.
    plt.rcParams['axes.xmargin'] = 0
    plt.rcParams['axes.ymargin'] = 0
    plt.figure(1)

    # Set up top plot with real boundaries.
    real_boundaries = plt.subplot(211)
    real_boundaries.specgram(signal, NFFT=1024, Fs=sample_rate, noverlap=1023)
    real_boundaries.set_xlabel('Time [s]')
    real_boundaries.set_ylabel('Frequency [Hz]')

    for line in boundaries:
        real_boundaries.axvline(line, c="r")

    # Set up bottom plot with predicted boundaries.
    predicted_boundaries = plt.subplot(212)
    predicted_boundaries.specgram(signal, NFFT=1024, Fs=sample_rate, noverlap=1023)
    predicted_boundaries.set_xlabel('Time [s]')
    predicted_boundaries.set_ylabel('Frequency [Hz]')

    for minimum in minima:
        predicted_boundaries.axvline(minimum, c="r")
    plt.show()


def plot_wavespec_boundaries(wav_path, boundaries):
    """Plots the given boundaries over the waveform and spectrogram."""
    # Reads wav and gets sample rate and frames that will make up waveform.
    sample_rate, signal = wav.read(wav_path)
    # For plotting, uses the sample rate to generate time values in seconds.
    time = np.linspace(0, len(signal) / sample_rate, num=len(signal))

    # Sets axes to start at 0.
    plt.rcParams['axes.xmargin'] = 0
    plt.rcParams['axes.ymargin'] = 0
    plt.figure(1)

    # Plotting waveform
    waveform = plt.subplot(211)
    waveform.set_xlabel("Time [s]")
    waveform.set_ylabel("Amplitude")
    plt.plot(time, signal)

    # Plotting Spectrogram
    spectrogram = plt.subplot(212)
    spectrogram.specgram(signal, NFFT=1024, Fs=sample_rate, noverlap=1023)
    spectrogram.set_xlabel('Time [s]')
    spectrogram.set_ylabel('Frequency [Hz]')

    # Plotting phoneme boundary lines.
    for line in boundaries:
        waveform.axvline(line, c="r")
        spectrogram.axvline(line, c="r")
    plt.show()


# ------------------- EVALUATION ------------------- #
def correct_boundaries(manual_b, predicted_b, margin=10):
    """Checks every boundary index in predicted to see if there is a manual index within one frame of it."""
    correct = []
    used = []
    for index in predicted_b:
        # Checks exact boundary first
        if index in manual_b:
            correct.append(index)
            used.append(index)
        else:
            # Then checks the next closest boundaries until one is found
            for i in range(0, margin+1):
                if index+i in manual_b and index+i not in used:
                    correct.append(index)
                    used.append(index+i)
                    break
                elif index-i in manual_b and index-i not in used:
                    correct.append(index)
                    used.append(index+i)
                    break
    return correct


def hit_rate(manual, predicted, margin=10):
    # No of correct boundaries / No of manual boundaries
    try:
        return len(correct_boundaries(manual, predicted, margin)) / len(manual)
    except ZeroDivisionError:
        return 0


def false_alarm_rate(manual, predicted, margin=10):
    # (Total predicted boundaries - Correct) / Total predicted
    try:
        return (len(predicted) - len(correct_boundaries(manual, predicted, margin))) / len(predicted)
    except ZeroDivisionError:
        return 0


def over_segmentation(manual, predicted):
    try:
        return (len(predicted) - len(manual)) / len(manual)
    except ZeroDivisionError:
        return 0


def f_score(manual, predicted, margin=10):
    fa = false_alarm_rate(manual, predicted, margin)
    hr = hit_rate(manual, predicted, margin)
    # print(fa)
    # print(hr)

    num = 2*(1-fa)*hr
    den = 1-fa+hr

    try:
        return num/den
    except ZeroDivisionError:
        return 0


def r_score(manual, predicted, margin=10):
    ovs = over_segmentation(manual, predicted)
    hr = hit_rate(manual, predicted, margin)

    r1 = math.sqrt((1-hr)**2 + ovs**2)
    r2 = -ovs+hr-1/math.sqrt(2)
    return 1-((abs(r1)+abs(r2))/2)


if __name__ == "__main__":

    # For writing CSV file at the end.
    dict_data = []

    # CSV Headers
    score_names = ["hit_rate", "false_alarm", "over_segmentation", "F-score", "R-score", "num"]
    file_types = ["wa1", "wa2", "combined"]  # For looking at head-mounted and desk-mounted mic recordings. WSJCAM0.

    # HYPER PARAMETERS
    PEAK_HEIGHT = 0.2  # The value below which local minima are considered.
    K_REACH = 5  # The number of consecutive frames that are not neighbours in order to consider a frame reachable.
    MIN_WIDTH = 20  # The minimum kernel width of the range that you would like to test.
    MAX_WIDTH = 60  # The maximum kernel width of the range that you would like to test.
    WINDOW_CONSTRAINT = 50  # The number of frames that will be considered when calculating similarity.
    CORRECT_MARGIN = 10

    # FILE PATHS
    rootdir = r"C:\Users\Ian\Desktop\Corpus\WSJCAM0_Corpus_Full\output\disc3"
    path_file = os.path.join(rootdir, "test_paths.csv")

    for width in range(MIN_WIDTH, MAX_WIDTH+1, 10):
        print("CURRENT WIDTH: "+str(width))
        # Initialising scores dictionary.
        scores = {
            "wa1": {"hit_rate": 0, "false_alarm": 0, "over_segmentation": 0, "F-score": 0, "R-score": 0, "num": 0},
            "wa2": {"hit_rate": 0, "false_alarm": 0, "over_segmentation": 0, "F-score": 0, "R-score": 0, "num": 0},
            "combined": {"hit_rate": 0, "false_alarm": 0, "over_segmentation": 0, "F-score": 0, "R-score": 0, "num": 0}
        }

        # Open CSV file with the recordings' file paths in it.
        with open(path_file) as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                # Initialising file paths for data corresponding to current recording.
                npy_path = row["NPY_path"]
                wav_path = row["WAV_path"]
                phn_path = wav_path[:-8] + ".phn"

                # Load in feature vectors.
                frames = load(npy_path)

                # Set up Data Frame to record all info on current recording.
                frame_info = pd.DataFrame(columns=["mfccs", "start", "end", "boundary", "neighbourhood",
                                                   "n-graph", "epsilon", "reachable"],
                                          index=[i for i in range(0, len(frames))])

                # Sets up empty matrix for kernel-gram similarity and neighbourhood
                kernel_gram_matrix = matlib.zeros((len(frames), len(frames)))
                n_graph = []

                # Iterating through pairs of frames.
                for i in range(0, len(frames)):
                    epsilon_threshold = 0

                    # Filling Kernel-Gram matrix with similarities between the frame and itself.
                    kernel_gram_matrix[i, i] = gaussian_kernel_similarity(frames[i], frames[i], width)

                    for j in range(i+1, min(i+WINDOW_CONSTRAINT, len(frames))):
                        # print(str(i)+", "+str(j)+": "+str(gaussian_kernel_similarity(frames[i], frames[j])))

                        # Filling Kernel-Gram matrix with the rest of the similarity pairs.
                        # Also calculates epsilon threshold.
                        pair_similarity = gaussian_kernel_similarity(frames[i], frames[j], width)
                        kernel_gram_matrix[i, j] = pair_similarity
                        epsilon_threshold += pair_similarity

                    # Finds the last reachable frame for the current frame.
                    epsilon_threshold /= WINDOW_CONSTRAINT
                    last_reached = reachable(frames, i, epsilon_threshold, h=width, k=K_REACH, w=WINDOW_CONSTRAINT)

                    current_n_graph = 0
                    for j in range(i+1, last_reached+1):
                        current_n_graph += kernel_gram_matrix[i, j]

                    n_graph.append(current_n_graph)

                    frame_info.iloc[i] = pd.Series({
                        "mfccs": frames[i],
                        "start": i*10,
                        "end": (i*10)+25,
                        "boundary": False,
                        "neighbourhood": {},
                        "n-graph": current_n_graph,
                        "epsilon": epsilon_threshold,
                        "reachable": last_reached
                    })

                # Retrieving manual boundaries
                manual = phoneme_boundaries(phn_path, wav_path)  # Frame indices

                # Finding predicted boundaries.
                n_graph = np.asarray(n_graph)
                predicted = find_peaks(np.negative(n_graph), height=-PEAK_HEIGHT)[0]  # Frame indices

                # FOR SPEC COMPARISON
                # predicted_time = [item/100 for item in predicted]  # Time indices
                # manual = phoneme_boundaries(phn_path, wav_path, frames=False)  # Time indices
                # print(wav_path)
                # plot_spectrogram_comparison(wav_path, manual, predicted_time)

                # EVALUATION
                # Updating scores dictionary for final CSV
                if wav_path.endswith("wa1.wav"):
                    scores["wa1"]["hit_rate"] += hit_rate(manual, predicted, margin=CORRECT_MARGIN)
                    scores["wa1"]["false_alarm"] += false_alarm_rate(manual, predicted, margin=CORRECT_MARGIN)
                    scores["wa1"]["over_segmentation"] += over_segmentation(manual, predicted)
                    scores["wa1"]["F-score"] += f_score(manual, predicted, margin=CORRECT_MARGIN)
                    scores["wa1"]["R-score"] += r_score(manual, predicted, margin=CORRECT_MARGIN)
                    scores["wa1"]["num"] += 1

                elif wav_path.endswith("wa2.wav"):
                    scores["wa2"]["hit_rate"] += hit_rate(manual, predicted, margin=CORRECT_MARGIN)
                    scores["wa2"]["false_alarm"] += false_alarm_rate(manual, predicted, margin=CORRECT_MARGIN)
                    scores["wa2"]["over_segmentation"] += over_segmentation(manual, predicted)
                    scores["wa2"]["F-score"] += f_score(manual, predicted, margin=CORRECT_MARGIN)
                    scores["wa2"]["R-score"] += r_score(manual, predicted, margin=CORRECT_MARGIN)
                    scores["wa2"]["num"] += 1

                print(wav_path[-16:]+" complete.")

        # EVALUATION continued
        for item in score_names:
            scores["combined"][item] = scores["wa1"][item] + scores["wa2"][item]

        for quality in file_types:
            for score in score_names[:-1]:
                scores[quality][score] /= scores[quality]["num"]

        for quality in file_types:
            current_row = {
                "kernel-width": width,
                "type": quality,
            }
            for score in score_names:
                current_row.update({score: scores[quality][score]})

            dict_data.append(current_row)

    # Writing results into CSV
    csv_columns = ["type", "kernel-width"]
    csv_columns.extend(score_names)
    with open(os.path.join(rootdir, "results.csv"), "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(dict_data)

        # for quality in file_types:
        #     print("\n\n-----------------------------------------"+quality+"-------------------------------------------")
        #     for score in score_names:
        #         print(score+": "+str("%.3f" % scores[quality][score]))




