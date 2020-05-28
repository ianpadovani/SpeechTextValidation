import kernel_gram
import os
import re
import pandas as pd
from syllable_nuclei import speech_rate
from mfcc_extract import mfcc_extract
from textmt.tokenise.tokenise import MTTokenizer
from G2PTool.g2p_cw_rules import g2p_cw_rules

ALL_VOWELS = ['ɐː', 'ɛː', 'iː', 'ɪː', 'ɔː', 'ʊː', 'ɛɪ', 'ɐɪ', 'ɔɪ', 'ɛʊ', 'ɐʊ', 'ɔʊ', 'ɐ', 'ɛ', 'ɪ', 'ɔ', 'ʊ', ]
ALL_AFFRICATES = ['ʦ', 'ʧ', 'ǳ', 'ʤ', 'ts', 'tʃ', 'dz', 'dʒ']


def remove_duplicates(word):
    # If length is less than 2, no possible duplicates.
    if len(word) < 2:
        return word

    # Convert to list for traversal and replacement.
    listword = list(word)

    # Index to compare to
    j = 0

    # i remains ahead of j, when j is on a repeated letter, it replaces that letter with the one at i.
    for i in range(len(listword)):
        if listword[j] != listword[i]:
            j += 1
            listword[j] = listword[i]

    j += 1
    listword = listword[:j]

    # Convert back to string.
    finalword = ""
    for character in listword:
        finalword += character

    return str(finalword)


if __name__ == "__main__":

    root_path = r"C:\Users\Ian\Desktop\FYP RECORDINGS\Experiment 2 - Altered"

    sentences = {
        "S01": {"text": "Dawn iż-żewġ sejħiet kienu għal għoxrin liċenzja kull waħda",
                "pred_nsyll": 0,
                "pred_nseg": 0},
        "S02": {"text": "Hekk se nagħmlu bil-vot tagħna dwar din il-liġi",
                "pred_nsyll": 0,
                "pred_nseg": 0},
        "S03": {"text": "Ħafna nies għadhom mhux qed jifhmu li kull qatra tgħodd",
                "pred_nsyll": 0,
                "pred_nseg": 0},
        "S04": {"text": "Ir-raġel ittieħed l-Isptar Mater Dei permezz ta' ambulanza",
                "pred_nsyll": 0,
                "pred_nseg": 0},
        "S05": {"text": "U inti minn hemm trid tibda",
                "pred_nsyll": 0,
                "pred_nseg": 0},
        "S06": {"text": "Jiġifieri anke dan il-konċett inbidel",
                "pred_nsyll": 0,
                "pred_nseg": 0},
        "S07": {"text": "Għandna wkoll bżonn ta' entitajiet oħra",
                "pred_nsyll": 0,
                "pred_nseg": 0},
        "S08": {"text": "Meta qabel wieħed kien jiddependi fuq il-hard copy dan ma setax isir",
                "pred_nsyll": 0,
                "pred_nseg": 0},
        "S09": {"text": "Dan l-att se jkun qed jiġi mressaq mis-Segretarju Parlamentari",
                "pred_nsyll": 0,
                "pred_nseg": 0},
        "S10": {"text": "Liema huma dawn l-operazzjonijiet u kemm hu n-numru ta' kull kategorija",
                "pred_nsyll": 0,
                "pred_nseg": 0}
    }

    tok = MTTokenizer()

    # Predicting number of syllables and segments from each sentence.
    for key in sentences.keys():
        tokens = tok.tokenize(sentences[key]["text"])
        for word in tokens:
            # Transcribe word
            trans = g2p_cw_rules(word)

            # Replace all kinds of vowels with V. Normalises monophthongs, diphthongs and long vowels.
            for vowel in ALL_VOWELS:
                if re.search(vowel, trans):
                    trans = re.sub(vowel, "V", trans)

            # Replaces affricates with single character.
            for affricate in ALL_AFFRICATES:
                if re.search(affricate, trans):
                    trans = re.sub(affricate, "A", trans)

            # Remove duplicate segments, as kernel-gram does not detect gemination.
            trans = remove_duplicates(trans)
            print(trans)

            sentences[key]["pred_nseg"] += len(trans)
            for char in trans:
                if char == "V":
                    sentences[key]["pred_nsyll"] += 1

    # Iterating through recordings.

    datalist = []
    for root, subdirectory, files in os.walk(root_path):
        for file in files:
            if file.endswith(".wav"):
                wav_path = os.path.join(root, file)
                npy_path = os.path.join(root, file[:-4] + ".npy")

                # Extracting MFCCs
                mfcc_extract(wav_path, npy_path)

                # Counting number of syllables in file.
                nsyll_dict = speech_rate(wav_path)

                # Counting number of segments in file.
                pred_boundaries, matrix = kernel_gram.predict_boundaries(npy_path)

                info_dict = {"speaker": root[-3:],
                             "utterance": file[:3],
                             "type": file[4],
                             "pred_nsyll": sentences[file[:3]]["pred_nsyll"],
                             "nsyll": nsyll_dict["nsyll"],
                             "npause": nsyll_dict["npause"],
                             "pred_nseg": sentences[file[:3]]["pred_nseg"],
                             "nseg": len(pred_boundaries) - 1,
                             "speech_rate": nsyll_dict["speechrate(nsyll / dur)"]
                             }

                datalist.append(info_dict)

        df = pd.DataFrame(datalist)
        df.to_csv(r'C:\Users\Ian\Desktop\FYP RECORDINGS\exp2tool_results.csv')
