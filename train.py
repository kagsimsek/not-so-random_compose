import os
import time
import pandas as pd
from autograd import numpy as np
from autograd import grad
import matplotlib.pyplot as plt

# import the data table
name = 'big_data_with'
dtf = './data/' + name + '.csv'
dt = pd.read_csv(dtf, header = None)

# number of data points
P = len(dt)

# weights in optimization
# 1 for training data
# 10 for composition sample
b = np.array(dt.T.iloc[0].values, float)

# three vector natural features x1, x2, x3
# one vector target y

# target note: cv, o, cd
y_1 = np.array(dt.T.iloc[10].values, int)
y_2 = np.array(dt.T.iloc[11].values, int)
y_3 = np.array(dt.T.iloc[12].values, int)

# number of classes for each target
C_1 = 12
C_2 = 9
C_3 = 14

# natural features
# cv: canonical value [0=C, 11=B] scaled to the C major scale
# o: octave [0, 8]
# cd: canonical duration [0, 13], sorted list of 'whole', 'half,' 'quarter', etc.

# first note: cv, o, cd
x1_1 = np.array(dt.T.iloc[1].values, float) / float(C_1 - 1)
x1_2 = np.array(dt.T.iloc[2].values, float) / float(C_2 - 1)
x1_3 = np.array(dt.T.iloc[3].values, float) / float(C_3 - 1)

# second note: cv, o, cd
x2_1 = np.array(dt.T.iloc[4].values, float) / float(C_1 - 1)
x2_2 = np.array(dt.T.iloc[5].values, float) / float(C_2 - 1)
x2_3 = np.array(dt.T.iloc[6].values, float) / float(C_3 - 1)

# third note: cv, o, cd
x3_1 = np.array(dt.T.iloc[7].values, float) / float(C_1 - 1)
x3_2 = np.array(dt.T.iloc[8].values, float) / float(C_2 - 1)
x3_3 = np.array(dt.T.iloc[9].values, float) / float(C_3 - 1)

# engineered features
# d_ij: absolute difference between xi and xj on piano [-87, 87]
# k_i: xi in key [0=no, 1=yes]
# dir_ij: direction from xi to xj [-1=descend, 0=repeat, 1=ascend]
# c_ij: xi and xj are consonant [0=no, 1=yes]
d_12 = np.array(dt.T.iloc[13].values, float) / 87.0
d_13 = np.array(dt.T.iloc[14].values, float) / 87.0
d_23 = np.array(dt.T.iloc[15].values, float) / 87.0

k_1 = np.array(dt.T.iloc[16].values, float) 
k_2 = np.array(dt.T.iloc[17].values, float)
k_3 = np.array(dt.T.iloc[18].values, float)

dir_12 = np.array(dt.T.iloc[19].values, float)
dir_13 = np.array(dt.T.iloc[20].values, float)
dir_23 = np.array(dt.T.iloc[21].values, float)

c_12 = np.array(dt.T.iloc[22].values, float)
c_13 = np.array(dt.T.iloc[23].values, float)
c_23 = np.array(dt.T.iloc[24].values, float)

# set of all features
x = np.column_stack((x1_1, x1_2, x1_3,
                     x2_1, x2_2, x2_3,
                     x3_1, x3_2, x3_3,
                     d_12, d_13, d_23,
                     k_1, k_2, k_3,
                     dir_12, dir_13, dir_23,
                     c_12, c_13, c_23))

# number of features
N = x.shape[1]

# model
# boosting with neural network units
# model(x, Theta_m) = model(x, Theta_{m-1}) + w_{0,m} + w_{1,m} f(x, u)
# f(x, u) = tanh(xo . u)
# xo = (1, x)
def model(x, Theta, m, C): 
  if m == 0:
    return np.full((P, C), Theta[:, 0, 0])
  else:
    w0 = Theta[:, m, 0]
    w1 = Theta[:, m, 1]
    u0 = Theta[:, m, 2]
    u = Theta[:, m, 3:]
    f = np.tanh(u0 + np.dot(x, u.T))
    prev = model(x, Theta, m - 1, C)
    return prev + w0 + w1 * f

# multiclass softmax cost
def g(Theta, y, m, C):
  den = np.sum(np.exp(model(x, Theta, m, C)), axis = 1)
  num = np.exp(model(x, Theta, m, C))
  one_hot = np.eye(num.shape[1])[y]
  num = np.sum(num * one_hot, axis = 1)
  return -np.sum(b * np.log(num / den)) / np.sum(b)

# gradient
dg = grad(g, 0)

# training with gradient descent
def train(y, M, K):
  if np.array_equal(y, y_1): 
    C = C_1
    target = '1'
  if np.array_equal(y, y_2): 
    C = C_2
    target = '2'
  if np.array_equal(y, y_3): 
    C = C_3
    target = '3'
  
  Theta = np.zeros((C, M + 1, N + 3))

  for m in range(1, M + 1):
    Theta[:, m, 2:] = 4.0 * np.random.randn(C, N + 1)

  prog = 0
  total = (M + 1) * K

  # zeroth round
  start = time.time()
  for k in range(K):
    grads = dg(Theta, y, 0, C)
    Theta[:, 0, 0] -= 0.01 * grads[:, 0, 0]
    prog += 1
    percent_prog = round(float(prog) / float(total) * 100.0)
    elapsed_time = round(time.time() - start)
    print(f"\rtraining {target}: {percent_prog}% [{elapsed_time} s]", end = ' ', flush = True)

  # mth round
  for m in range(1, M + 1):
    for k in range(K):
      grads = dg(Theta, y, m, C)
      Theta[:, m, :] -= 0.01 * grads[:, m, :]
      prog += 1
      percent_prog = round(float(prog) / float(total) * 100.0)
      elapsed_time = round(time.time() - start)
      print(f"\rtraining target {target}: {percent_prog}% [{elapsed_time} s]", end = ' ', flush = True)

  print()

  return Theta

M = 30
K = 1000

Theta_1 = train(y_1, M, K)
Theta_2 = train(y_2, M, K)
Theta_3 = train(y_3, M, K)

def summary(Theta):
  if np.array_equal(Theta, Theta_1): 
    y = y_1
    C = C_1
    target = '1'
  if np.array_equal(Theta, Theta_2):
    y = y_2 
    C = C_2
    target = '2'
  if np.array_equal(Theta, Theta_3):
    y = y_3 
    C = C_3
    target = '3'

  predictions = np.zeros((M + 1, P))
  N_mis = np.zeros(M + 1)
  percent_error = np.zeros(M + 1)

  for m in range(M + 1):
    predictions[m, :] = np.array([np.argmax(model(x, Theta, m, C)[p]) for p in range(P)])
    N_mis[m] = np.sum(y != predictions[m, :])
    percent_error[m] = round(float(N_mis[m]) / float(P) * 100.0, 2)

  os.makedirs('./out/figures', exist_ok = True)
  plt.plot(range(M + 1), percent_error)
  plt.xlabel('$m$')
  plt.ylabel('percent error')
  figure_dir = './out/figures/' + name
  figure_location = figure_dir + '/percent_error_target_' + target + '.pdf'
  os.makedirs(figure_dir, exist_ok = True)
  plt.savefig(figure_location)
  print(f"exporting the percent error of target {target} vs. boosting round plot at")
  print(f"  \'{figure_location}\'")
  plt.clf()

  return 0

summary(Theta_1)
summary(Theta_2)
summary(Theta_3)

export_dir = './out/optimization/' + name
os.makedirs(export_dir, exist_ok = True)
np.save(export_dir + '/Theta_1.npy', Theta_1)
np.save(export_dir + '/Theta_2.npy', Theta_2)
np.save(export_dir + '/Theta_3.npy', Theta_3)
print('exporting weights at')
print(f"  {export_dir}/Theta_1.npy")
print(f"  {export_dir}/Theta_2.npy")
print(f"  {export_dir}/Theta_3.npy")

