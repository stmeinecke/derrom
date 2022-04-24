import numpy as np
import matplotlib.pyplot as plt


import ELPH_dyn


def get_runs_gaussian_init(kmax, n_kmax, gaussian_paras, tmax = 5000.0, n_tmax = 1000):

  runs = []

  for k in range(gaussian_paras.shape[0]):
    
    print('run ', k, ' from ', gaussian_paras.shape[0])
 
    p = gaussian_paras[k]
    init = ELPH_dyn.get_init_cond_gauss(kmax = kmax, n_kmax = n_kmax, max_pos = p[0], width = p[1], density = p[2])
    eldyn = ELPH_dyn.get_el_dynamics(init, n_kmax = n_kmax, tmax = tmax, n_tmax = n_tmax)

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


def get_KFold_runs(runs, folds=5):
    KFold_runs = np.array_split(runs, folds) #numpy returns list of ndarrays
    KFold_runs = [list(array) for array in KFold_runs] #covert ndarrays back to list such that KFold_runs is a list of lists of ndarrays (the individual runs)
    return KFold_runs

def get_KFold_CV_scores(model, runs, folds=5, seed=817, norm='fro', **kwargs):
    
    #create shuffled copy of the runs
    rng = np.random.default_rng(seed=seed)
    sruns = runs.copy()
    rng.shuffle(sruns)
    
    #split runs into folds
    KFold_runs = get_KFold_runs(sruns, folds=folds)
    
    scores = [] #of the individual folds
    for k in range(folds):
        train_runs = KFold_runs.copy()
        test_runs = train_runs.pop(k) #test_runs = the kth fold, train_runs to remaining folds

        train_runs = [item for sublist in train_runs for item in sublist] #unpack the training folds into one flattened list

        #train the model on the training runs and get scores from the testing runs
        model.load_runs(train_runs) 
        model.train(**kwargs)
        mean_score = model.score_multiple_runs(test_runs, norm=norm)[0]
        
        scores.append(mean_score)
    
    return np.mean(scores), scores