import pandas as pd
import numpy as np
import math
import csv
import scipy.io.wavfile as wav
from scipy.spatial.distance import pdist, squareform
from scipy.signal import find_peaks
from python_speech_features import mfcc
from numpy import load
from numpy import matlib
from matplotlib import pyplot as plt
from matplotlib import cm
import os


def phoneme_boundaries(phn_path, wav_path):
    """Takes the path to a .phn file and extracts the initial boundary for each phoneme."""
    phonemes = []
    sample_rate, signal = wav.read(wav_path)

    with open(phn_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            line = line.split("\t")
            phonemes.append(tuple(line))
            boundaries = [round((int(item[1]) / sample_rate) * 100) for item in phonemes]
    return boundaries


def gaussian_kernel_similarity(v1, v2, h=35):
    """Calculates gaussian kernel similarity for two vectors."""
     # h is Gaussian Kernel Width
    euclidean_norm = np.linalg.norm(v1-v2, 2)
    weighted = -(euclidean_norm**2/h**2)
    similarity = np.exp(weighted)
    return similarity
    # return np.exp(-np.linalg.norm(v1 - v2, 2) ** 2 / (2. * 1 ** 2))


def epsilon_neighbourhood(v1, v2, epsilon, h=35):
    """Returns True if the two frames are in the same neighbourhood and False otherwise."""
    if 1-gaussian_kernel_similarity(v1, v2, h) < epsilon:
        return True
    else:
        return False


def reachable(matrix, i, epsilon, h=35, k=5):
    """Receives a matrix, the current frame and the epsilon threshold for the current frame.
    Returns the index of the last reachable frame for the current frame."""
    count = 0
    p_boundary = i
    for j in range(i+1, min(i+50, len(frames))):
        # Check each vector with the current frame's.
        # If not neighbours add to count, set p_boundary to the index of last neighbour.
        # If neighbours, reset count and p_boundary.
        if epsilon_neighbourhood(matrix[i], matrix[j], epsilon, h):
            count = 0
            p_boundary = j
        else:
            count += 1

        # Check if k is 4.
        if count == k:
            return p_boundary

    # If no boundary found by end, set last possible frame as boundary.
    return min(i+50, len(frames)-1)


# ------------------- PLOTTING ------------------- #
def plot_kernel_gram_boundaries(matrix, boundaries, minima):
    plt.rcParams['axes.xmargin'] = 0
    plt.rcParams['axes.ymargin'] = 0
    plt.figure(1)

    real_boundaries = plt.subplot(121)
    real_boundaries.imshow(matrix, cmap="Greys")

    for line in boundaries:
        real_boundaries.axvline(line, c="r")

    predicted_boundaries = plt.subplot(122)
    predicted_boundaries.imshow(matrix, cmap="Greys")

    for item in minima:
        predicted_boundaries.axvline(item, c="r")
    plt.show()


def plot_wavespec_boundaries(wav_path, boundaries):
    # Reads wav and gets sample rate and frames that will make up waveform.
    sample_rate, signal = wav.read(wav_path)
    # For plotting, uses the sample rate to generate time values in seconds.
    time = np.linspace(0, len(signal) / sample_rate, num=len(signal))

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


# ------------------- SCORING ------------------- #
def correct_boundaries(manual, predicted, margin=1):
    """Checks every boundary index in predicted to see if there is a manual index within one frame of it."""
    correct = []
    for index in predicted:
        for i in range(index - margin, index + margin + 1):
            if i in manual:
                correct.append(i)
                break
    return correct


def hit_rate(manual, predicted):
    # No of correct boundaries / No of manual boundaries
    try:
        return len(correct_boundaries(manual, predicted)) / len(manual)
    except ZeroDivisionError:
        return 0


def false_alarm_rate(manual, predicted):
    # (Total predicted boundaries - Correct) / Total predicted
    try:
        return (len(predicted) - len(correct_boundaries(manual, predicted))) / len(predicted)
    except ZeroDivisionError:
        return 0


def over_segmentation(manual, predicted):
    try:
        return (len(predicted) - len(manual)) / len(manual)
    except ZeroDivisionError:
        return 0


def f_score(manual, predicted):
    fa = false_alarm_rate(manual, predicted)
    hr = hit_rate(manual, predicted)
    # print(fa)
    # print(hr)

    num = 2*(1-fa)*hr
    den = 1-fa+hr

    try:
        return num/den
    except ZeroDivisionError:
        return 0


def r_score(manual, predicted):
    os = over_segmentation(manual, predicted)
    hr = hit_rate(manual, predicted)

    r1 = math.sqrt((1-hr)**2 + os**2)
    r2 = -os+hr-1/math.sqrt(2)
    return 1-((abs(r1)+abs(r2))/2)

if __name__ == "__main__":

    dict_data = []
    score_names = ["hit_rate", "false_alarm", "over_segmentation", "F-score", "R-score", "num"]
    file_types = ["wa1", "wa2", "combined"]

    # CHANGE AS NECESSARY
    rootdir = r"C:\Users\Ian\Desktop\Corpus\WSJCAM0_Corpus_Full\output\disc3"
    path_file = os.path.join(rootdir, "paths.csv")

    for width in range(10, 101, 10):
        print("CURRENT WIDTH: "+str(width))
        scores = {
            "wa1": {"hit_rate": 0, "false_alarm": 0, "over_segmentation": 0, "F-score": 0, "R-score": 0, "num": 0},
            "wa2": {"hit_rate": 0, "false_alarm": 0, "over_segmentation": 0, "F-score": 0, "R-score": 0, "num": 0},
            "combined": {"hit_rate": 0, "false_alarm": 0, "over_segmentation": 0, "F-score": 0, "R-score": 0, "num": 0}
        }

        with open(path_file) as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:

                npy_path = row["NPY_path"]
                wav_path = row["WAV_path"]
                phn_path = wav_path[:-8] + ".phn"

                frames = load(npy_path)

                frame_info = pd.DataFrame(columns=["mfccs", "start", "end", "boundary", "neighbourhood",
                                                   "n-graph", "epsilon", "reachable"],
                                          index=[i for i in range(0, len(frames))])

                kernel_gram_matrix = matlib.zeros((len(frames), len(frames)))
                n_graph = []

                # Iterating through frame pairs.
                for i in range(0, len(frames)):
                    epsilon_threshold = 0

                    # Filling Kernel-Gram matrix with similarities between the frame and itself.
                    kernel_gram_matrix[i, i] = gaussian_kernel_similarity(frames[i], frames[i], width)

                    for j in range(i+1, min(i+50, len(frames))):
                        # print(str(i)+", "+str(j)+": "+str(gaussian_kernel_similarity(frames[i], frames[j])))

                        # Filling Kernel-Gram matrix with the rest of the similarity pairs.
                        # Also calculates epsilon threshold.
                        pair_similarity = gaussian_kernel_similarity(frames[i], frames[j], width)
                        kernel_gram_matrix[i, j] = pair_similarity
                        epsilon_threshold += pair_similarity

                    # Finds the last reachable frame for the current frame.
                    epsilon_threshold /= 50
                    last_reached = reachable(frames, i, epsilon_threshold, h=width)

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
                manual = phoneme_boundaries(phn_path, wav_path)

                # Finding predicted boundaries.
                n_graph = np.asarray(n_graph)
                predicted = find_peaks(np.negative(n_graph), height=-0.5)[0]

                if wav_path.endswith("wa1.wav"):
                    scores["wa1"]["hit_rate"] += hit_rate(manual, predicted)
                    scores["wa1"]["false_alarm"] += false_alarm_rate(manual, predicted)
                    scores["wa1"]["over_segmentation"] += over_segmentation(manual, predicted)
                    scores["wa1"]["F-score"] += f_score(manual, predicted)
                    scores["wa1"]["R-score"] += r_score(manual, predicted)
                    scores["wa1"]["num"] += 1

                elif wav_path.endswith("wa2.wav"):
                    scores["wa2"]["hit_rate"] += hit_rate(manual, predicted)
                    scores["wa2"]["false_alarm"] += false_alarm_rate(manual, predicted)
                    scores["wa2"]["over_segmentation"] += over_segmentation(manual, predicted)
                    scores["wa2"]["F-score"] += f_score(manual, predicted)
                    scores["wa2"]["R-score"] += r_score(manual, predicted)
                    scores["wa2"]["num"] += 1

                print(wav_path[-16:]+" complete.")

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




