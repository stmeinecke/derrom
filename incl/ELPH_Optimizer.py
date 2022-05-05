import numpy as np
import sys

sys.path.append("incl/")
import ELPH_utils 

from scipy.optimize import minimize


def scpmin(state,target):

  lambda1 = 1.0

  def loss(beta, state, target):
    
    state_ncols = state.shape[1]
    target_ncols = target.shape[1]
    
    beta = np.reshape(beta, (state_ncols,target_ncols))
    
    pred = state.T @ beta
    res = pred - target.T 

    #err_lstsqs = np.mean( np.sum( np.square(res), axis = 0 ) )
    err_lstsqs = np.linalg.norm(err, 'fro')
    error = err_lstsqs

    #err_reg = alpha * np.mean( np.sum( np.square(beta), axis = 0 ) )
    err_reg = alpha * np.linalg.norm(beta, 'fro')
    error +=  err_reg 
    
    #ones = jnp.ones(target.T.shape[1])
    #err_density = lambda1 * jnp.linalg.norm( pred @ ones  - target.T @ ones , ord=2, axis=0) 
    #err_density = lambda1 * np.sum( np.square( (pred - target.T) @ ones ) ) 
    #error += err_density

    return error
    

  beta_init = np.random.normal(loc=0.0, scale=1.0, size=(state.shape[1],target.shape[1]))
  #beta_init = w
  result = minimize(loss, beta_init, args=(state,target))
  
  return result.x
