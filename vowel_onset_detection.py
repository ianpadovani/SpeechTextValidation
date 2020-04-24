import parselmouth
from parselmouth.praat import run_file
from kernel_gram import phoneme_boundaries
from numpy import load
from numpy import matlib

if __name__ == "__main__":
    # directory_path = r"C:\Users\Ian\Desktop\OUTPUT\TEST\DR1\FAKS0"
    # script_path = r"C:\Users\Ian\OneDrive\Documents\UNIVERSITY\DISSERTATION\PraatIO Practice\syllable_count.praat"
    #
    # objects = run_file(script_path, -25, 2, 0.3, True, directory_path)
    # print(objects)

    npy_path = r"C:\Users\Ian\Desktop\OUTPUT\TEST\DR1\FAKS0\SA1.waV.npy"

    frames = load(npy_path)
    kernel_gram_matrix = matlib.zeros((len(frames), len(frames)))

    print(len(frames))
    print(len(kernel_gram_matrix))