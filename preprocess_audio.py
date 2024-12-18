# %%
import os
import torch
import torchaudio
import librosa
import numpy as np
from tqdm import tqdm

# %%

wav_path = '/content/drive/MyDrive/final_speech_dataset/audio_5sec_mo2atan'
wav_new_path = './data/arabic-speech-corpus/train/wav_new'

sr_target = 22050
silence_audio_size = 256 * 3

device = 'cuda'

wav_fpaths = [f.path for f in os.scandir(wav_path) if f.path.endswith('.wav')]

if not os.path.exists(wav_new_path):
    os.makedirs(wav_new_path)
    print(f"Created folder @ {wav_new_path}")

# %%

for wav_fpath in tqdm(wav_fpaths):
    fname = os.path.basename(wav_fpath)
       
    fpath = os.path.join(wav_path, fname)
    wave, sr = torchaudio.load(fpath, normalize=True)
    wav, sr = librosa.load(fpath, sr=sr_target)
    original_shape = np.shape(wav)[0]

    if sr != sr_target:
        wave = wave.to(device)
        wave = torchaudio.functional.resample(wave, sr, sr_target, 
                                              lowpass_filter_width=1024)

    if wave.shape[1] < original_shape:
      pad_amount = original_shape - wave.shape[1]
      wave = torch.nn.functional.pad(wave, (0, pad_amount))  # Pad at the end
    elif wave.shape[1] > original_shape:
      wave = wave[:, :original_shape]  # Truncate the extra samples
    
    wave_ = wave[0].cpu().numpy()
    #wave_ = wave_ / np.abs(wave_).max() * 0.999
    #wave_ = librosa.effects.trim(wave_, top_db=23, frame_length=1024, hop_length=256)[0]
    #wave_ = np.append(wave_, [0.]*silence_audio_size)
    print(np.shape(wave_))
    x=fname.split('.')
    n=str(x[0]).zfill(4)
    fname="ARA NORM  "+n+".wav"

    torchaudio.save(f'{wav_new_path}/{fname}',
                    torch.Tensor(wave_).unsqueeze(0), sr_target)
      



test_wav_path = '/content/drive/MyDrive/final_speech_dataset/Final_wav_audio/test'
tes_wav_new_path = './data/arabic-speech-corpus/test/wav_new'

wav_fpaths_test = [f.path for f in os.scandir(test_wav_path) if f.path.endswith('.wav')]

if not os.path.exists(tes_wav_new_path):
    os.makedirs(tes_wav_new_path)
    print(f"Created folder @ {tes_wav_new_path}")

for wav_fpath in tqdm(wav_fpaths_test):
    fname = os.path.basename(wav_fpath)

    fpath = os.path.join(test_wav_path, fname)
    wave, sr = torchaudio.load(fpath)
    wav, sr = librosa.load(fpath, sr=sr_target)
    original_shape = np.shape(wav)[0]
    
    if sr != sr_target:
        wave = wave.to(device)
        wave = torchaudio.functional.resample(wave, sr, sr_target, 
                                              lowpass_filter_width=1024)

    if wave.shape[1] < original_shape:
      pad_amount = original_shape - wave.shape[1]
      wave = torch.nn.functional.pad(wave, (0, pad_amount))  # Pad at the end
    elif wave.shape[1] > original_shape:
      wave = wave[:, :original_shape]  # Truncate the extra samples

    wave_ = wave[0].cpu().numpy()
    #wave_ = wave_ / np.abs(wave_).max() * 0.999
    #wave_ = librosa.effects.trim(wave_, top_db=23, frame_length=1024, hop_length=256)[0]
    #wave_ = np.append(wave_, [0.]*silence_audio_size)
    x=fname.split('.')
    n=str(x[0]).zfill(4)
    fname="ARA NORM  "+n+".wav"
    torchaudio.save(f'{tes_wav_new_path}/{fname}',
                    torch.Tensor(wave_).unsqueeze(0), sr_target)


# %%