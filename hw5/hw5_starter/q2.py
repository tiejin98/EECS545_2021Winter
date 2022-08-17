import numpy as np
import matplotlib.pyplot as plt
import imageio
from scipy.stats import multivariate_normal  # Don't use other functions in scipy

def train_gmm(train_data, init_pi, init_mu, init_sigma):
    ##### TODO: Implement here!! #####
    # Hint: multivariate_normal() might be useful
    states = {
      'pi': init_pi,
      'mu': init_mu,
      'sigma': init_sigma,
    }
    for _ in range(50):
        # E-step
        E_matrix_temp = np.zeros(shape=(train_data.shape[0],len(init_pi)))
        for i in range(len(init_pi)):
          E_matrix_temp[:,i] = states['pi'][i]*\
                               multivariate_normal.pdf(train_data,mean=states['mu'][i],cov=states['sigma'][i])
        E_matrix = E_matrix_temp/np.sum(E_matrix_temp,axis=1).reshape(-1,1)
        # M-step
        Nk = np.sum(E_matrix,axis=0)
        pi_new = Nk / train_data.shape[0]
        states['pi'] = pi_new
        mu_new = np.dot(E_matrix.T,train_data)
        mu_new /= Nk.reshape(-1,1)
        states['mu'] = mu_new
        sigma_new = np.zeros_like(init_sigma)
        for i in range(len(init_pi)):
            temp_cov = train_data-mu_new[i]
            temp_cov = np.dot(temp_cov.T*E_matrix[:,i],temp_cov)
            temp_cov /= Nk[i]
            sigma_new[i] = temp_cov
        states['sigma'] = sigma_new
        temp = np.log(np.sum(E_matrix_temp,axis=1))
        log_likelihood = np.sum(temp)
        print(_,"iter loglikelihood:",log_likelihood)
    E_matrix_temp = np.zeros(shape=(train_data.shape[0], len(init_pi)))
    for i in range(len(init_pi)):
        E_matrix_temp[:, i] = states['pi'][i] * multivariate_normal.pdf(train_data, mean=states['mu'][i],
                                                                        cov=states['sigma'][i])
    temp = np.log(np.sum(E_matrix_temp,axis=1))
    log_likelihood = np.sum(temp)
    print(_+1,"iter loglikelihood:",log_likelihood)
    return states

def test_gmm(states, test_data):
    result = {}
    E_matrix = np.zeros(shape=(test_data.shape[0], len(init_pi)))
    for i in range(len(init_pi)):
        E_matrix[:, i] = states['pi'][i] * multivariate_normal.pdf(test_data, mean=states['mu'][i],
                                                                   cov=states['sigma'][i])
    res = np.argmax(E_matrix,axis=1)
    compressed_data = np.zeros_like(test_data)
    for c in range(len(states['pi'])):
        compressed_data[res == c] = states['mu'][c]
    result['pixel-error'] = calculate_error(test_data, compressed_data)
    compressed_data /= 255
    plt.imshow(compressed_data.reshape(512,512,3))
    plt.show()
    return result

### DO NOT CHANGE ###
def calculate_error(data, compressed_data):
    assert data.shape == compressed_data.shape
    error = np.sqrt(np.mean(np.power(data - compressed_data, 2)))
    return error
### DO NOT CHANGE ###

# Load data
img_small = np.array(imageio.imread('q12data/mandrill-small.tiff')) # 128 x 128 x 3
img_large = np.array(imageio.imread('q12data/mandrill-large.tiff')) # 512 x 512 x 3

ndim = img_small.shape[-1]
train_data = img_small.reshape(-1, ndim).astype(float)
test_data = img_large.reshape(-1, ndim).astype(float)

# GMM
num_centroid = 5
initial_mu_indices = [16041, 15086, 15419,  3018,  5894]
init_pi = np.ones((num_centroid, 1)) / num_centroid
init_mu = train_data[initial_mu_indices, :]
init_sigma = np.tile(np.identity(ndim), [num_centroid, 1, 1])*1000.

states = train_gmm(train_data, init_pi, init_mu, init_sigma)
result_gmm = test_gmm(states, test_data)
print('GMM result=', result_gmm)
print("\mu_k:",states['mu'])
print("\sigma_k:",states['sigma'])

