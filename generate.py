import os
import numpy as np
import pandas as pd
import pretty_midi

# from training
M = 30

C_1 = 12
C_2 = 9
C_3 = 14

# import optimized weights
Theta_1 = np.load('./out/optimization/big_data_with/Theta_1.npy')
Theta_2 = np.load('./out/optimization/big_data_with/Theta_2.npy')
Theta_3 = np.load('./out/optimization/big_data_with/Theta_3.npy')

# slightly tweak the model
def model(x, Theta, m): 
  if np.array_equal(Theta, Theta_1): C = C_1
  if np.array_equal(Theta, Theta_2): C = C_2
  if np.array_equal(Theta, Theta_3): C = C_3
      
  if m == 0:
    return np.full(C, Theta[:, 0, 0])
  else:
    w0 = Theta[:, m, 0]
    w1 = Theta[:, m, 1]
    u0 = Theta[:, m, 2]
    u = Theta[:, m, 3:]
    f = np.tanh(u0 + np.dot(x, u.T))
    prev = model(x, Theta, m - 1)
    return prev + w0 + w1 * f

# get features from three notes
# each with pv, octave, and canonical time value label
def is_in_key(pv):
  return 1.0 if pv in [0, 2, 4, 5, 7, 9, 11] else 0.0 

def is_ascending(av1, av2):
  d = av2 - av1
  if d > 0: return 1.0
  if d == 0: return 0.0
  if d < 0: return -1.0

def is_longer(ctvl1, ctvl2):
  if ctvl2 > ctvl1: return 1.0
  if ctvl2 == ctvl1: return 0.0
  if ctvl2 < ctvl1: return -1.0

def are_consonant(av1, av2):
  d = (av2 - av1) % 12
  return 1.0 if d in [0, 3, 4, 7] else 0.0 

def get_features(x1, x2, x3):
  x_11 = x1[0] / (C_1 - 1)
  x_12 = x1[1] / (C_2 - 1)
  x_13 = x1[2] / (C_3 - 1)

  x_21 = x2[0] / (C_1 - 1)
  x_22 = x2[1] / (C_2 - 1)
  x_23 = x2[2] / (C_3 - 1)

  x_31 = x3[0] / (C_1 - 1)
  x_32 = x3[1] / (C_2 - 1)
  x_33 = x3[2] / (C_3 - 1)

  d_12 = ((x_21 + x_22 * 12) - (x_11 + x_12 * 12)) / 87
  d_13 = ((x_31 + x_32 * 12) - (x_11 + x_12 * 12)) / 87
  d_23 = ((x_31 + x_32 * 12) - (x_21 + x_22 * 12)) / 87

  k_1 = is_in_key(x_11)
  k_2 = is_in_key(x_21)
  k_3 = is_in_key(x_31)

  dir_12 = is_ascending(x_11 + x_12 * 12, x_21 + x_22 * 12)
  dir_13 = is_ascending(x_11 + x_12 * 12, x_31 + x_32 * 12)
  dir_23 = is_ascending(x_21 + x_22 * 12, x_31 + x_32 * 12)

  c_12 = are_consonant(x_11 + x_12 * 12, x_21 + x_22 * 12)
  c_13 = are_consonant(x_11 + x_12 * 12, x_31 + x_32 * 12)
  c_23 = are_consonant(x_21 + x_22 * 12, x_31 + x_32 * 12)

  return x_11, x_12, x_13, x_21, x_22, x_23, x_31, x_32, x_33, d_12, d_13, d_23, k_1, k_2, k_3, dir_12, dir_13, dir_23, c_12, c_13, c_23

dt_sample = pd.read_csv('./data/processed/sample.csv', header = None)
P_sample = len(dt_sample)

def generate(x):
  probs_1 = np.exp(model(x, Theta_1, M)) / np.sum(np.exp(model(x, Theta_1, M)))
  probs_2 = np.exp(model(x, Theta_2, M)) / np.sum(np.exp(model(x, Theta_2, M)))
  probs_3 = np.exp(model(x, Theta_3, M)) / np.sum(np.exp(model(x, Theta_3, M)))

  top_two_indices_1 = np.argsort(probs_1)[-2:]
  top_two_indices_2 = np.argsort(probs_2)[-1:]
  top_two_indices_3 = np.argsort(probs_3)[-1:]

  top_two_probs_1 = probs_1[top_two_indices_1]
  top_two_probs_2 = probs_2[top_two_indices_2]
  top_two_probs_3 = probs_3[top_two_indices_3]

  normalized_probs_1 = top_two_probs_1 / np.sum(top_two_probs_1)
  normalized_probs_2 = top_two_probs_2 / np.sum(top_two_probs_2)
  normalized_probs_3 = top_two_probs_3 / np.sum(top_two_probs_3)

  select_1 = np.random.choice(top_two_indices_1, p = normalized_probs_1)
  select_2 = np.random.choice(top_two_indices_2, p = normalized_probs_2)
  select_3 = np.random.choice(top_two_indices_3, p = normalized_probs_3)

  return select_1, select_2, select_3

# add the existing notes to the melody
melody = []

x_11 = np.array(dt_sample.T.iloc[1].values)
x_12 = np.array(dt_sample.T.iloc[2].values)
x_13 = np.array(dt_sample.T.iloc[3].values)

x_21 = np.array(dt_sample.T.iloc[4].values)
x_22 = np.array(dt_sample.T.iloc[5].values)
x_23 = np.array(dt_sample.T.iloc[6].values)

x_31 = np.array(dt_sample.T.iloc[7].values)
x_32 = np.array(dt_sample.T.iloc[8].values)
x_33 = np.array(dt_sample.T.iloc[9].values)

for p in range(len(x_11)):
  melody.append((x_11[p], x_12[p], x_13[p]))

melody.append((x_21[-1], x_22[-1], x_23[-1]))
melody.append((x_31[-1], x_32[-1], x_33[-1]))

# generate notes until a predetermined length is achieved
# allowed time values
quarter_note = 1.0
    
whole = 4.0 * quarter_note
half = 2.0 * quarter_note
quarter = 1.0 * quarter_note
eighth = 1.0 / 2.0 * quarter_note
sixteenth = 1.0 / 4.0 * quarter_note
thirtysecond = 1.0 / 8.0 * quarter_note

half_dot = 1.5 * half
quarter_dot = 1.5 * quarter
eighth_dot = 1.5 * eighth
sixteenth_dot = 1.5 * sixteenth

half_3let = 2.0 / 3.0 * half
quarter_3let = 2.0 / 3.0 * quarter
eighth_3let = 2.0 / 3.0 * eighth
sixteenth_3let = 2.0 / 3.0 * sixteenth

# time values
TVs = np.array([whole, half_dot, half, quarter_dot, half_3let, quarter, eighth_dot, quarter_3let, eighth, sixteenth_dot, eighth_3let, sixteenth, sixteenth_3let, thirtysecond])

length = np.sum([TVs[melody[r][-1]] for r in range(len(melody))])
stop = 60.0
while length < stop:
  y_1, y_2, y_3 = generate(get_features(melody[-3], melody[-2], melody[-1]))
  melody.append((y_1, y_2, y_3))
  length += TVs[y_3]

bpm = 60.0
melody_durs = []
for row in melody:
  melody_durs.append((row[0], row[1], 60.0 / bpm * TVs[row[2]]))
melody = melody_durs

# generate MIDI
midi_object = pretty_midi.PrettyMIDI(initial_tempo=60)
instrument = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program('Acoustic Grand Piano'))
start_time = 0.0
    
for note, octave, duration in melody:
  pitch = note + 12 * (octave + 1) 
  adjusted_duration = duration
  midi_note = pretty_midi.Note(velocity=100, pitch=pitch, start=start_time, end=start_time + adjusted_duration)
  instrument.notes.append(midi_note)
  start_time += adjusted_duration
    
midi_object.instruments.append(instrument)
midi_object.write('./out/mid/from_sample.mid')
print('MIDI generated at')
print('  \'./out/mid/from_sample.mid\'')

