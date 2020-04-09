from python_speech_features import mfcc, delta
from numpy import save
import numpy as np
import scipy.io.wavfile as wav
import os
import csv


def mfcc_extract(wav_path, numpy_path):
    """Takes a sound file, extracts MFCC features and saves them as a numpy file."""
    (rate, sig) = wav.read(wav_path)
    mfcc_feat = mfcc(sig, rate, nfft=2048, winlen=0.02).tolist()
    deltas = delta(mfcc_feat, 2).tolist()
    double_deltas = delta(deltas, 2).tolist()

    for i in range(0, len(mfcc_feat)):
        mfcc_feat[i].extend(deltas[i])
        mfcc_feat[i].extend(double_deltas[i])

    mfcc_feat = np.array([np.array(item) for item in mfcc_feat])
    save(numpy_path, mfcc_feat)
    print(numpy_path+" generated...")


if __name__ == "__main__":
    directory = r"C:\Users\Ian\Desktop\OUTPUT"
    section = "TEST"
    csv_dir = os.path.join(directory, section+"_paths.csv")
    data_dir = os.path.join(directory, section)

    # WSJCAM0
    # csv_columns = ["Speaker", "Utterance", "WAV_path", "NPY_path"]

    # TIMIT
    csv_columns = ["Dialect", "Speaker", "Utterance", "WAV_path", "NPY_path"]
    dict_data = []

    print("Generating CSV file and MFCC feature vectors for this disc.")

    dialects = os.listdir(data_dir)
    for code in dialects:
        for root, subdirectory, file in os.walk(os.path.join(data_dir, code)):
            for item in file:
                # if item.endswith(".wa1") or item.endswith(".wv1") or item.endswith(".wa2"):
                if item.endswith(".waV"):
                    # wv_path = os.path.join(root, item)
                    # wav_path = os.path.join(root, item+".wav")
                    wav_path = os.path.join(root, item)
                    numpy_path = os.path.join(root, item+".npy")

                    # os.rename(wv_path, wav_path)

                    mfcc_extract(wav_path, numpy_path)

                    current_row = {
                        "Dialect": code,
                        "Speaker": root[-5:],
                        "Utterance": item[:-4],
                        "WAV_path": wav_path,
                        "NPY_path": numpy_path
                    }

                    # WSJCAM0
                    # current_row = {"Speaker": item[:3],
                    #                "Utterance": item[3:8],
                    #                # "WV_path": wv_path,
                    #                "WAV_path": wav_path,
                    #                "NPY_path": numpy_path}
                    dict_data.append(current_row)

    with open(csv_dir, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(dict_data)
    print("CSV Compiled. MFCCs Extracted.")


