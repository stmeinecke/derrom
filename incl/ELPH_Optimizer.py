import numpy as np
import sys

sys.path.append("incl/")
import ELPH_utils 

from scipy.optimize import minimize
def scpmin(state,target, alpha=0.0, lambda1=0.0):

  #lambda1 = 1.0
  #alpha=10.0**-2.8
  

  def loss(beta, state, target):
    
    state_ncols = state.shape[1]
    target_ncols = target.shape[1]
    
    beta = np.reshape(beta, (state_ncols,target_ncols))
    
    pred = state @ beta
    res = pred - target 

    #err_lstsqs = np.mean( np.sum( np.square(res), axis = 0 ) )
    err_lstsqs = np.linalg.norm(res, 'fro')
    error = err_lstsqs

    #err_reg = alpha * np.mean( np.sum( np.square(beta), axis = 0 ) )
    err_reg = alpha * np.linalg.norm(beta, 'fro')
    error +=  err_reg 
    
    #ones = jnp.ones(target.T.shape[1])
    #err_density = lambda1 * jnp.linalg.norm( pred @ ones  - target.T @ ones , ord=2, axis=0) 
    #err_density = lambda1 * np.sum( np.square( (pred - target.T) @ ones ) ) 
    #error += err_density

    return error
    

  #beta_init = np.random.normal(loc=0.0, scale=1.0, size=(state.shape[1],target.shape[1]))
  beta_init = ELPH_utils.get_ridge_regression_weights(state, target, alpha=alpha)
  result = minimize(loss, beta_init, args=(state.T,target.T))
  
  
  beta = np.reshape(result.x, (state.shape[0], target.shape[0]))
  return beta


import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

def grddcnt(state, target, alpha=0.0, lambda1=0.0):
  
  
  def loss(beta):
    
    pred = state.T @ beta
    res = pred - target.T 

    err_lstsqs = jnp.linalg.norm(res, 'fro')**2
    error = err_lstsqs

    err_reg = alpha * jnp.linalg.norm(beta, 'fro')**2
    error +=  err_reg 
    
    ones = jnp.ones(target.T.shape[1])
    #err_density = lambda1 * jnp.linalg.norm( pred @ ones  - target.T @ ones , ord=2, axis=0) 
    err_density = lambda1 * np.sum( jnp.square( (pred - target.T) @ ones ) ) 
    error += err_density

    return error
  
  PIML_grad = jax.jit(jax.grad(loss))
  
  beta = ELPH_utils.get_ridge_regression_weights(state, target, alpha=alpha)
  #beta += np.random.normal(loc=0.0, scale=1e-2, size=beta.shape)
  
  
  print('ridge regression loss: ' + str(loss(beta)))
  plt.plot(np.ravel(PIML_grad(beta)))
  plt.show()
  
  GD_nmax = 50000
  GD_nout = 10
  GD_learning_rate = 1e-6
  GD_momentum = 0.5
  GD_diff = 0.0

  for i in range(GD_nmax):
      if(i%(GD_nmax//GD_nout) == 0):
        print('loss: ' + str(loss(beta)))
      GD_diff = GD_momentum * GD_diff - GD_learning_rate * PIML_grad(beta)
      beta += GD_diff


  plt.plot(np.ravel(PIML_grad(beta)))
  plt.show()
    
  return beta
  
  
  
import optax as opt

#def sgd(state, target, alpha=0.0, lambda1=0.0):
  
  #def loss(beta):
    
    #pred = state.T @ beta
    #res = pred - target.T 

    #err_lstsqs = jnp.linalg.norm(res, 'fro')**2
    #error = err_lstsqs

    #err_reg = alpha * jnp.linalg.norm(beta, 'fro')**2
    #error +=  err_reg 
    
    #ones = jnp.ones(target.T.shape[1])
    #err_density = lambda1 * np.sum( jnp.square( (pred - target.T) @ ones ) ) 
    #error += err_density

    #return error
  
  #PIML_grad = jax.jit(jax.grad(loss))
  
  #beta = ELPH_utils.get_ridge_regression_weights(state, target, alpha=alpha)
  ##beta += np.random.normal(loc=0.0, scale=1e-2, size=beta.shape)
  
  
  #print('ridge regression loss: ' + str(loss(beta)))
  #plt.plot(np.ravel(PIML_grad(beta)))
  #plt.show()
  
  #start_learning_rate = 1e-5
  #optimizer = opt.adam(start_learning_rate)
  
  #opt_state = optimizer.init(beta)
  
  #sgd_nmax = 10000
  #sgd_nout = 10
  
  #for i in range(sgd_nmax):
    #if(i%(sgd_nmax//sgd_nout) == 0):
        #print('loss: ' + str(loss(beta)))
    #grads = PIML_grad(beta)
    #updates, opt_state = optimizer.update(grads, opt_state)
    #beta = opt.apply_updates(beta, updates)

  #plt.plot(np.ravel(PIML_grad(beta)))
  #plt.show()
    
  #return beta
  
  
def sgd(state, target, alpha=0.0, lambda1=0.0):
  
  def loss(beta, state, target):
    
    pred = state.T @ beta
    res = pred - target.T 

    err_lstsqs = jnp.linalg.norm(res, 'fro')**2
    error = err_lstsqs

    err_reg = alpha * jnp.linalg.norm(beta, 'fro')**2
    error +=  err_reg 
    
    ones = jnp.ones(target.T.shape[1])
    err_density = lambda1 * np.sum( jnp.square( (pred - target.T) @ ones ) ) 
    error += err_density

    return error
  
  PIML_grad = jax.jit(jax.grad(loss))
  
  beta = ELPH_utils.get_ridge_regression_weights(state, target, alpha=alpha)
  #beta += np.random.normal(loc=0.0, scale=1e-2, size=beta.shape)
  
  
  print('ridge regression loss: ' + str(loss(beta, state, target)))
  #plt.plot(np.ravel(PIML_grad(beta, state, target)))
  #plt.show()
  
  
  print(state.shape)
  print(target.shape)
  
  n_batches = 200
  
  rng = np.random.default_rng(42)
  #rng.shuffle(state, axis=1)
  #rng.shuffle(target, axis=1)
  
  TRAINING_DATA = np.array_split( state , n_batches, axis=1)
  LABELS = np.array_split( target , n_batches, axis=1)
  
  
  def fit(params, optimizer):
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, batch, labels):
      loss_value, grads = jax.value_and_grad(loss)(params, batch, labels)
      updates, opt_state = optimizer.update(grads, opt_state, params)
      params = opt.apply_updates(params, updates)
      return params, opt_state, loss_value

    for i, (batch, labels) in enumerate(zip(TRAINING_DATA, LABELS)):
      params, opt_state, loss_value = step(params, opt_state, batch, labels)
      if i % 10 == 0:
        print(f'step {i}, loss: {loss_value}')

    return params

  optimizer = opt.adam(learning_rate=1e-5)
  beta = fit(beta, optimizer)
  
  print('regression loss: ' + str(loss(beta, state, target)))
  
  return beta
  
  #start_learning_rate = 1e-5
  #optimizer = opt.adam(start_learning_rate)
  
  #opt_state = optimizer.init(beta)
  
  #sgd_nmax = 1000
  #sgd_nout = 10
  
  #for i in range(sgd_nmax):
    #if(i%(sgd_nmax//sgd_nout) == 0):
        #print('loss: ' + str(loss(beta, state, target)))
    #grads = PIML_grad(beta, state, target)
    #updates, opt_state = optimizer.update(grads, opt_state, beta)
    #beta = opt.apply_updates(beta, updates)

  #plt.plot(np.ravel(PIML_grad(beta, state, target)))
  #plt.show()
    
  #return beta
  
  
  
  
  
