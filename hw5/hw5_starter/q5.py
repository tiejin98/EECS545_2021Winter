import numpy as np
from scipy.io.wavfile import write
Fs = 11025

def sigmoid(x):
    return 1/(1+np.exp(-x))

def normalize(dat):
    return 0.99 * dat / np.max(np.abs(dat))

def load_data():
    mix = np.loadtxt('q5data/q5.dat')
    return mix

def unmixer(X):
    M, N = X.shape
    W = np.eye(N)

    anneal = [0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.02, 0.02, 0.01, 0.01,
              0.005, 0.005, 0.002, 0.002, 0.001, 0.001]
    print('Separating tracks ...')
    for alpha in anneal:
        for i in range(M):
            temp_x = X[i]
            temp_x = temp_x.reshape(-1,1)
            temp = np.dot(W,temp_x)
            temp = 1-2*sigmoid(temp)
            temp = np.dot(temp,temp_x.T)
            temp += np.linalg.inv(W.T)
            W += alpha*temp
    ###################################
    return W

def unmix(X, W):
    # cov_mat = np.cov(X.T)
    # eigen_vals, eigen_vectors = np.linalg.eig(cov_mat)
    # eigen_val_index = np.argsort(eigen_vals)[::-1]
    # eigen_vals = eigen_vals[eigen_val_index]
    # eigen_vectors = eigen_vectors[:, eigen_val_index]
    # Lambda = np.diag(eigen_vals**(-0.5))
    # temp = np.dot(X,eigen_vectors)
    # X_tilt = np.dot(Lambda,temp.T)
    S = np.dot(X,W.T)
    return S

X = normalize(load_data())
print(X.shape)
print('Saving mixed track 1')
write('q5_mixed_track_1.wav', Fs, X[:, 0])

import time
t0 = time.time()
W = unmixer(X) # This will take around 2min
print('time=', time.time()-t0)
S = normalize(unmix(X, W))

for track in range(5):
    print(f'Saving unmixed track {track}')
    write(f'q5_unmixed_track_{track}.wav', Fs, S[:, track])

print('W solution:')
print(W)
