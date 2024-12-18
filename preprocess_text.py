# This file needs to be run in the main folder
# %%
import text
from utils import read_lines_from_file


def write_lines_to_file(path, lines, mode='w', encoding='utf-8'):
    with open(path, mode, encoding=encoding) as f:
        for i, line in enumerate(lines):
            if i == len(lines)-1:
                f.write(line)
                break
            f.write(line + '\n')

# %%


lines = read_lines_from_file('/content/drive/MyDrive/final_speech_dataset/Final Transcript/transcripts_train.txt')
lines2 = read_lines_from_file('/content/drive/MyDrive/final_speech_dataset/Final Transcript/transcripts_test.txt')

new_lines_arabic = []
new_lines_phonetic = []
new_lines_buckw = []

for line in lines:
    wav_name, utterance = line.split('" "')
    print(wav_name)
    wav_name, utterance = wav_name[1:], utterance[:-1]
    utterance = utterance.replace("a~", "~a") \
                         .replace("i~", "~i") \
                         .replace("u~", "~u") \
                         .replace(" - ", " ") \
                         .replace("?"," ")    \
                         .replace("."," ")
    print(utterance)

    utterance_arab = text.arabic_to_buckwalter(utterance)
    print(utterance_arab)
    utterance_phon = text.buckwalter_to_phonemes(utterance_arab)

    line_new_ara = f'"{wav_name}" "{utterance_arab}"'
    new_lines_arabic.append(line_new_ara)

    line_new_pho = f'"{wav_name}" "{utterance_phon}"'
    new_lines_phonetic.append(line_new_pho)

    line_new_buckw = f'"{wav_name}" "{utterance}"'
    new_lines_buckw.append(line_new_buckw)



new_lines_arabic2 = []
new_lines_phonetic2 = []
new_lines_buckw2 = []

for line in lines2:
    wav_name, utterance = line.split('" "')
    print(wav_name)
    wav_name, utterance = wav_name[1:], utterance[:-1]
    utterance = utterance.replace("a~", "~a") \
                         .replace("i~", "~i") \
                         .replace("u~", "~u") \
                         .replace(" - ", " ") \
                         .replace("?"," ")    \
                         .replace("."," ")
    print(utterance)

    utterance_arab = text.arabic_to_buckwalter(utterance)
    print(utterance_arab)
    utterance_phon = text.buckwalter_to_phonemes(utterance_arab)

    line_new_ara2 = f'"{wav_name}" "{utterance_arab}"'
    new_lines_arabic2.append(line_new_ara2)

    line_new_pho2 = f'"{wav_name}" "{utterance_phon}"'
    new_lines_phonetic2.append(line_new_pho2)

    line_new_buckw2 = f'"{wav_name}" "{utterance}"'
    new_lines_buckw2.append(line_new_buckw2)


# %% train

write_lines_to_file('./data/train_arab.txt', new_lines_buckw)
write_lines_to_file('./data/train_phon.txt', new_lines_phonetic)
write_lines_to_file('./data/train_buckw.txt', new_lines_arabic)

# %% test

write_lines_to_file('./data/test_arab.txt', new_lines_buckw2)
write_lines_to_file('./data/test_phon.txt', new_lines_phonetic2)
write_lines_to_file('./data/test_buckw.txt', new_lines_arabic2)