import numpy as np
import matplotlib.pyplot as plt


import ELPH_dyn


def get_runs_gaussian_init(kmax, n_kmax, gaussian_paras):

  runs = []

  for p in gaussian_paras:

    init = ELPH_dyn.get_init_cond_gauss(kmax = kmax, n_kmax = n_kmax, max_pos = p[0], width = p[1], density = p[2])
    eldyn = ELPH_dyn.get_el_dynamics(init, n_kmax = n_kmax)

    runs.append( eldyn )

  return runs


def get_SVD_from_runs(runs):
  
  data_matrix = np.concatenate(runs,axis=1)

  U,S,V = np.linalg.svd(data_matrix, full_matrices=False)

  return U,S


def get_reduced_coef_matrix(runs, U, rdim):
  data_matrix = np.concatenate(runs,axis=1)
  return U[:,:rdim].T @ data_matrix
  
  
def standardize_data_matrix(matrix):
  mean = np.mean(matrix, axis = 1)
  std = np.std(matrix, axis = 1)

  new = ((matrix.T - mean.T)/std.T).T

  return new, mean, std

def destandardize_data_matrix(matrix, mean, std):    
  return ( (matrix.T * std) + mean ).T


def get_coef_runs(coef_data_matrix, n_splits):
  return np.asarray(np.split(coef_data_matrix, n_splits, axis=1))


def get_ridge_regression_weights(state, target, alpha):
  return np.linalg.inv(state @ state.T + alpha * np.identity(state.shape[0])) @ state @ target.T
