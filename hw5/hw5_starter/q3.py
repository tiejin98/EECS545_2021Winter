import numpy as np
import matplotlib.pyplot as plt
import time

def validate_PCA(states, train_data):
  from sklearn.decomposition import PCA
  pca = PCA()
  pca.fit(train_data)
  true_matrix = pca.components_.T
  true_ev = pca.explained_variance_
  output_matrix = states['transform_matrix']
  error = np.mean(np.abs(np.abs(true_matrix) - np.abs(output_matrix)) / np.abs(true_matrix))
  if error > 0.01:
    print('Matrix is wrong! Error=',error)
  else:
    print('Matrix is correct! Error=', error)

  output_ev = states['eigen_vals']
  error = np.mean(np.abs(true_ev - output_ev) / true_ev)
  if error > 0.01:
    print('Variance is wrong! Error=', error)
  else:
    print('Variance is correct! Error=', error)

def train_PCA(train_data):
  ##### TODO: Implement here!! #####
  # Note: do NOT use sklearn here!
  # Hint: np.linalg.eig() might be useful
  states = {
      'transform_matrix': np.identity(train_data.shape[-1]),
      'eigen_vals': np.ones(train_data.shape[-1])
  }
  cov_mat = np.cov(train_data.T)
  eigen_vals,eigen_vectors = np.linalg.eig(cov_mat)
  eigen_val_index = np.argsort(eigen_vals)[::-1]
  eigen_vals = eigen_vals[eigen_val_index]
  eigen_vectors = eigen_vectors[:, eigen_val_index]
  states['eigen_vals'] =eigen_vals
  #pca_score = np.dot(train_data,eigen_vectors)
  states['transform_matrix'] = eigen_vectors

  return states

def number_vals(vals, threshold):
  total = np.sum(vals)
  sum = 0
  for i in range(len(vals)):
    sum += vals[i]
    if sum/total >= threshold:
      break
  return i+1

# Load data
start = time.time()
images = np.load('q3data/q3.npy')
num_data = images.shape[0]
train_data = images.reshape(num_data, -1)

states = train_PCA(train_data)
print('training time = %.1f sec'%(time.time() - start))

validate_PCA(states, train_data)
x = list(range(len(states['eigen_vals'])))
y = states['eigen_vals']
eigen_vectors = states['transform_matrix']
plt.plot(x, y)
# plt.xticks(x)
plt.show()
for i in range(1, 11):
  a = plt.subplot(2, 5, i)
  if i > 1:
    image = eigen_vectors[:, i - 2].reshape(48, 42)
    a.imshow(image)
  else:
    image_mean = np.mean(train_data,axis=0).reshape(48,42)
    a.imshow(image_mean)
plt.show()
print(states['eigen_vals'][:10])
print(number_vals(states['eigen_vals'],0.95))
print(number_vals(states['eigen_vals'],0.99))

