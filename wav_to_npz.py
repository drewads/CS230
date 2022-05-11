
import numpy as np
from scipy.io.wavfile import read, write

wav_in_filepath = '../../../Downloads/CS230Piano.wav'  # Need to change this
npz_filepath = 'CS230Piano.npz'
wav_out_filepath = 'CS230PianoTransformed.wav'
sample_rate_out = 0 # Need to change this

def wav_to_arr(filepath):
  a = read(filepath)

  print("Sample rate:", a[0])

  arr = np.array(a[1], dtype=float)
  return arr

def arr_to_npz(filepath, arr):
  np.savez(filepath, arr)

def wav_to_npz():
  arr = wav_to_arr(wav_in_filepath)
  arr_to_npz(npz_filepath, arr)

def npz_to_wav():
  arr = np.load(npz_filepath)
  arr = arr['arr_0']
  write(wav_out_filepath, sample_rate_out, arr)

# wav_to_npz()
# npz_to_wav()
