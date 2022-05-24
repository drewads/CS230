from music21 import * # need to have music21 installed

filename = 'CS230PianoShortMidi.mid'

mf = midi.MidiFile()
mf.open(filename)
mf.read() 
mf.close()

#print(mf.ticksPerQuarterNote)
#print('tracks size', len(mf.tracks))

events = mf.tracks[0].events
#print('events size', len(events))
target_t = 0 # in ticks
t = 0 # in ticks
num_steps = 0
t_inc = (mf.ticksPerQuarterNote * 120 / 60) / 16000. # ticks per sample = ticks per second / fs
freq = 0
result = []
for e in events:
	if e.isDeltaTime() and (e.time is not None):
		# ticks / (ticks per quarter * quarters per second) (120BPM)
		target_t += e.time
		while t <= target_t:
			result.append(freq)
			t += t_inc
			num_steps += 1
	elif e.isNoteOn():
		freq = 2**((e.pitch - 69) / 12) * 440
	elif e.isNoteOff():
		freq = 0

import numpy as np
np_result = np.array(result, dtype=np.float16)
np.savez('CS230PianoShortFreqs.npz', np_result[:5856000]) # match length to audio