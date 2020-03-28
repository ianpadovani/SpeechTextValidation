from python_speech_features import mfcc
from numpy import save
import scipy.io.wavfile as wav
import os
import csv


def mfcc_extract(wav_path, numpy_path):
    """Takes a sound file, extracts MFCC features and saves them as a numpy file."""
    (rate, sig) = wav.read(wav_path)
    mfcc_feat = mfcc(sig, rate, nfft=2048)
    save(numpy_path, mfcc_feat)
    print(numpy_path[-16:]+" generated...")


if __name__ == "__main__":
    directory = r"C:\Users\Ian\Desktop\Corpus\WSJCAM0_Corpus_Full\output\disc6"
    csv_dir = os.path.join(directory, "paths.csv")
    data_dir = os.path.join(directory, "wsjcam0", "si_et_2")

    csv_columns = ["Speaker", "Utterance",
                   "WV_path", "WAV_path", "NPY_path"]
    dict_data = []

    print("Generating CSV file and MFCC feature vectors for this disc.")
    for root, subdirectory, filenames in os.walk(data_dir):
        for item in filenames:
            if item.endswith(".wa1") or item.endswith(".wv1") or item.endswith(".wa2"):

                wv_path = os.path.join(root, item)
                wav_path = os.path.join(root, item+".wav")
                numpy_path = os.path.join(root, item+".npy")

                os.rename(wv_path, wav_path)

                mfcc_extract(wav_path, numpy_path)

                current_row = {"Speaker": item[:3],
                               "Utterance": item[3:8],
                               "WV_path": wv_path,
                               "WAV_path": wav_path,
                               "NPY_path": numpy_path}
                dict_data.append(current_row)

    with open(csv_dir, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(dict_data)
    print("CSV Compiled. MFCCs Extracted.")


