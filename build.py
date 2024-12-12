# the data-table builder assumes there is only a monophonic melody line and the piece has already been transposed to the key of C major. 

import os
import numpy as np
import pandas as pd
import pretty_midi

# numerify the twelve notes of the western music theory
# C, C#, D, D#, ..., A#, B
# principal value of a note
def PV(note): 
  if note == 'C': 
    return 0
  if note == 'C#' or note == 'Db': 
    return 1 
  if note == 'D': 
    return 2
  if note == 'D#' or note == 'Eb': 
    return 3
  if note == 'E': 
    return 4
  if note == 'F': 
    return 5
  if note == 'F#' or note == 'Gb': 
    return 6
  if note == 'G': 
    return 7
  if note == 'G#' or note == 'Ab': 
    return 8
  if note == 'A': 
    return 9
  if note == 'A#' or note == 'Bb': 
    return 10
  if note == 'B': 
    return 11

# absolute value of a note
def AV(note, octave):
  return octave * 12 + PV(note)

# canonical duration of a note in seconds when played with 60 BPM
def CD(duration, BPM):
  return float(BPM) * duration / 60.0

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

# time value labels
TVLs = ['whole', 'half_dot', 'half', 'quarter_dot', 'half_3let', 'quarter', 'eighth_dot', 'quarter_3let', 'eighth', 'sixteenth_dot', 'eighth_3let', 'sixteenth', 'sixteenth_3let', 'thirtysecond']

# canonical time value label
def CTVL(duration):
  return np.argmin(np.abs(TVs - duration)) # to be used as the index of TVLs later

# nocturne BPMs
# with correction factor to fix mismatches in MIDI encoding
def BPMs(mid):
  if 'sample' in mid: return 80.0 * 1.0 / 0.9979166666666668
  if '/1.mid' in mid: return 90.0 * 0.5 / 0.56015625
  if '/2.mid' in mid: return 60.0 * 0.5 / 0.47291666666666665
  if '/6.mid' in mid: return 100.0 * 1.0 / 0.9479166666666666
  if '/11.mid' in mid: return 66.0 * 2.0 / 1.9123992891666668
  if '/20.mid' in mid: return 60.0 * 1.0 / 0.9479166666666666
  if '/21.mid' in mid: return 60.0 * 1.0 / 0.9979166666666667

# midi files
midi_dir = './data/midi/'
midi_files = [f for f in os.listdir(midi_dir) if os.path.isfile(os.path.join(midi_dir, f))]
midis = [os.path.join(midi_dir, file) for file in midi_files]

# data files
raw_data = ['./data/raw/' + os.path.basename(f) for f in midis]
raw_data = [os.path.splitext(path)[0] + ".csv" for path in raw_data]
processed_data = ['./data/processed/' + os.path.basename(f) for f in midis]
processed_data = [os.path.splitext(path)[0] + ".csv" for path in processed_data]

# prepare raw data table for midi file i
def export_raw_data(i):
  midi = midis[i]
  midi_data = pretty_midi.PrettyMIDI(midi)

  raw = [(note_name[:-1], int(note_name[-1]), note.end - note.start) for note in midi_data.instruments[0].notes if (note_name := pretty_midi.note_number_to_name(note.pitch))]

  melody = []

  # raw data table
  # 0:note
  # 1:octave
  # 2:duration (s)
  # 3:principal value
  # 4:absolute value
  # 5:canonical duration (s)
  # 6:canonical time value label
  # 7:weight 

  for row in raw:
    note = row[0]
    octave = row[1]
    duration = row[2]

    BPM = BPMs(midi)
    
    pv = PV(note)
    av = AV(note, octave)
    cd = CD(duration, BPM)
    ctvl = CTVL(cd)

    w = 10.0 if 'sample' in midi else 1.0 # sample is the composition to be completed

    melody.append((note, octave, duration, pv, av, cd, ctvl, w)) 
    
    export_dir = raw_data[i]
    with open(raw_data[i], 'w') as file:
      head = ('0:note', '1:octave', '2:duration (s)', '3:principal value', '4:absolute value', '5:canonical duration (s)', '6:canonical time value label', '7:weight')
      file.write(','.join(map(str, head)) + "\n")
      for row in melody:
        file.write(','.join(map(str, row)) + "\n")

for i in range(len(midis)):
  export_raw_data(i)

# processing tools
def is_in_key(pv):
  return 1.0 if pv in [0, 2, 4, 5, 7, 9, 11] else 0.0 # 1 if principal value is in the major scale

def is_ascending(av1, av2):
  d = av2 - av1
  if d > 0: return 1.0
  if d == 0: return 0.0
  if d < 0: return -1.0
  
def are_consonant(av1, av2):
  d = (av2 - av1) % 12
  return 1.0 if d in [0, 3, 4, 7] else 0.0 # 1 if unison, minor third, major third, perfect fifth

# prepare processed data table for midi file i
def export_processed_data(i):
  raw_dt = pd.read_csv(raw_data[i])

  P = len(raw_dt)  

  notes = np.array(raw_dt.T.iloc[0].values, str)
  octaves = np.array(raw_dt.T.iloc[1].values, int)
  pvs = np.array(raw_dt.T.iloc[3].values, int)
  avs = np.array(raw_dt.T.iloc[4].values, int)
  ctvls = np.array(raw_dt.T.iloc[6].values, int)
  weights = np.array(raw_dt.T.iloc[7].values, int)

  # natural features
  x1_1 = pvs[0:-3]
  x1_2 = octaves[0:-3]
  x1_3 = ctvls[0:-3]

  x2_1 = pvs[1:-2]
  x2_2 = octaves[1:-2]
  x2_3 = ctvls[1:-2]

  x3_1 = pvs[2:-1]
  x3_2 = octaves[2:-1]
  x3_3 = ctvls[2:-1]

  # targets
  y_1 = pvs[3:]
  y_2 = octaves[3:]
  y_3 = ctvls[3:]

  # engineered features
  d_12 = avs[1:-2] - avs[0:-3]
  d_13 = avs[2:-1] - avs[0:-3]
  d_23 = avs[2:-1] - avs[1:-2]

  k_1 = np.array([is_in_key(x1_1[p]) for p in range(P - 3)], int)
  k_2 = np.array([is_in_key(x2_1[p]) for p in range(P - 3)], int)
  k_3 = np.array([is_in_key(x3_1[p]) for p in range(P - 3)], int)

  dir_12 = np.array([is_ascending(avs[0:-3][p], avs[1:-2][p]) for p in range(P - 3)], int)
  dir_13 = np.array([is_ascending(avs[0:-3][p], avs[2:-1][p]) for p in range(P - 3)], int)
  dir_23 = np.array([is_ascending(avs[1:-2][p], avs[2:-1][p]) for p in range(P - 3)], int)
  
  c_12 = np.array([are_consonant(avs[0:-3][p], avs[1:-2][p]) for p in range(P - 3)], int)
  c_13 = np.array([are_consonant(avs[0:-3][p], avs[2:-1][p]) for p in range(P - 3)], int)
  c_23 = np.array([are_consonant(avs[1:-2][p], avs[2:-1][p]) for p in range(P - 3)], int)

  dt = []

  for p in range(P - 3):
    dt.append((weights[p],
               x1_1[p], x1_2[p], x1_3[p],
               x2_1[p], x2_2[p], x2_3[p],
               x3_1[p], x3_2[p], x3_3[p],
               y_1[p], y_2[p], y_3[p],
               d_12[p], d_13[p], d_23[p],
               k_1[p], k_2[p], k_3[p],
               dir_12[p], dir_13[p], dir_23[p],
               c_12[p], c_13[p], c_23[p]))

  with open(processed_data[i], 'w') as file:
    for row in dt:
      file.write(','.join(map(str, row)) + "\n")

for i in range(len(midis)):
  export_processed_data(i)

# big data table with or without sample
data_tables_with = [pd.read_csv(f, header = None) for f in processed_data]
combined_dt = pd.concat(data_tables_with, ignore_index = True)
combined_dt.to_csv('./data/big_data_with.csv', index = False, header = None)

data_tables_without = [pd.read_csv(f, header = None) for f in processed_data[1:]]
combined_dt = pd.concat(data_tables_without, ignore_index = True)
combined_dt.to_csv('./data/big_data_without.csv', index = False, header = None)

