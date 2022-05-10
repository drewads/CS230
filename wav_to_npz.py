
import numpy as np
from scipy.io.wavfile import read

wav_filepath = '../../../Downloads/CS230Piano.wav'  # Need to change this
npz_filepath = 'CS230Piano.npz'

def wav_to_arr(filepath):
  a = read(filepath)

  print("Sample rate:", a[0])

  arr = np.array(a[1], dtype=float)
  return arr

def arr_to_npz(filepath, arr):
  np.savez(filepath, arr)

arr = wav_to_arr(wav_filepath)
arr_to_npz(npz_filepath, arr)
