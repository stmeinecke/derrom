import numpy as np
import matplotlib.pyplot as plt


#######################################
### get initial conditions and and calc trajectories
#######################################

import ELPH

def get_el_trajectories(kmax, n_kmax, inits, tmax = 2000.0, n_tmax = 401):
  
  system = ELPH.ELPH(kmax=kmax,n_kmax=n_kmax)
  
  trajectories = []
  for k in range(len(inits)):
    print('run ', k+1, ' from ', len(inits))
    el_trajectory = system.get_electron_trajectory(inits[k], tmax = tmax, n_tmax = n_tmax)
    trajectories.append( eldyn )
  return trajectories



def get_gaussian_inits(system, gaussian_paras):
        
    inits = []

    for k in range(gaussian_paras.shape[0]):

        p = gaussian_paras[k]
        inits.append( system.get_init_cond_gauss(max_pos = p[0], width = p[1], density = p[2]) )

    return inits



#######################################
### save and load trajectories
#######################################

def save_trajectories(trajectories, filename='../trajectories'):
    np.savez(filename, trajectories)
    
def load_trajectories(filename='../trajectories.npz'):
    
    from os.path import exists

    if not exists(filename):
        print('trajectories file ot found')
        return None
    else:
        npz_trajectories = np.load(filename)
        trajectories = np.split(npz_trajectories['arr_0'], npz_trajectories['arr_0'].shape[0], axis=0)

        for k in range(len(trajectories)):
            trajectories[k] = np.reshape(trajectories[k], trajectories[k].shape[1:])
        return trajectories
    

