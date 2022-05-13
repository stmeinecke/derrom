import numpy as np
import sys

sys.path.append("incl/")
import ELPH_utils 

import jax
import jax.numpy as jnp
import optax as opt

import matplotlib.pyplot as plt


class base_optimizer:
    def __init__(self):
        pass
    
    def solve(self):
        raise NotImplementedError


class lstsqrs(base_optimizer):
    def __init__(self):
        pass
    
    def solve(self, state, target):
        return np.asarray( np.linalg.lstsq(state.T, target.T, rcond = -1)[0] )

    
class ridge(base_optimizer):
    def __init__(self, alpha = 0.0):
        self.alpha = alpha
        
    def solve(self, state, target):
        return np.linalg.inv(state @ state.T + self.alpha * np.identity(state.shape[0])) @ state @ target.T
        
        
class PIML_adam(base_optimizer):
    def __init__(self, alpha = 0.0, lambda1=0.0, mini_batch_size = 50, epochs = 1):
        self.alpha = alpha
        self.lambda1 = lambda1
        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        
    def solve(self, state, target):
        
        
        def loss(beta, state, target):
    
            pred = beta.T @ state
            res = pred - target

            err_lstsqs = jnp.sum(jnp.square(res))
            #err_lstsqs = jnp.amax(jnp.abs(res))
            error = err_lstsqs

            err_reg = self.alpha * jnp.sum(jnp.square(beta))
            error +=  err_reg 

            ones = jnp.ones(target.shape[0])
            err_density = self.lambda1 * np.sum( jnp.square( ones @ (pred - target) ) ) 
            
            
            error += err_density

            return error
  
        PIML_grad = jax.jit(jax.grad(loss))

        beta = ELPH_utils.get_ridge_regression_weights(state, target, alpha=self.alpha)
#         beta += np.random.normal(loc=0.0, scale=1e-2, size=beta.shape)


        print('ridge regression loss: ' + str(loss(beta, state, target)))
        #plt.plot(np.ravel(PIML_grad(beta, state, target)))
        #plt.show()


#         print(state.shape)
#         print(target.shape)

#         n_batches = 1000
        n_batches = state.shape[1]/self.mini_batch_size

        rng = np.random.default_rng(42)
  
  
        def fit(params, optimizer):
            opt_state = optimizer.init(params)

            @jax.jit
            def step(params, opt_state, batch, labels):
                loss_value, grads = jax.value_and_grad(loss)(params, batch, labels)
                updates, opt_state = optimizer.update(grads, opt_state, params)
                params = opt.apply_updates(params, updates)
                return params, opt_state, loss_value

            for k in range(self.epochs):
                permuted_inds = rng.permutation(state.shape[1])
                TRAINING_DATA = np.array_split( np.array(state)[:,permuted_inds] , n_batches, axis=1)
                LABELS = np.array_split( np.array(target)[:,permuted_inds] , n_batches, axis=1)

                for i, (batch, labels) in enumerate(zip(TRAINING_DATA, LABELS)):
                    params, opt_state, loss_value = step(params, opt_state, batch, labels)
#                     if i % 100 == 0:
#                         print(f'step {i}, loss: {loss_value}')
                        
                print('epoch:', k+1, 'loss:', loss(params, state, target) )

            return params

        
        optimizer = opt.adam(learning_rate=1e-5)
        beta = fit(beta, optimizer)
  
        print('regression loss: ' + str(loss(beta, state, target)))

        return beta
  

#   #start_learning_rate = 1e-5
#   #optimizer = opt.adam(start_learning_rate)
  
#   #opt_state = optimizer.init(beta)
  
#   #sgd_nmax = 1000
#   #sgd_nout = 10
  
#   #for i in range(sgd_nmax):
#     #if(i%(sgd_nmax//sgd_nout) == 0):
#         #print('loss: ' + str(loss(beta, state, target)))
#     #grads = PIML_grad(beta, state, target)
#     #updates, opt_state = optimizer.update(grads, opt_state, beta)
#     #beta = opt.apply_updates(beta, updates)

#   #plt.plot(np.ravel(PIML_grad(beta, state, target)))
#   #plt.show()
    
#   #return beta
        
        
        
        

# def grddcnt(state, target, alpha=0.0, lambda1=0.0):
  
  
#   def loss(beta):
    
#     pred = state.T @ beta
#     res = pred - target.T 

#     err_lstsqs = jnp.linalg.norm(res, 'fro')**2
#     error = err_lstsqs

#     err_reg = alpha * jnp.linalg.norm(beta, 'fro')**2
#     error +=  err_reg 
    
#     ones = jnp.ones(target.T.shape[1])
#     #err_density = lambda1 * jnp.linalg.norm( pred @ ones  - target.T @ ones , ord=2, axis=0) 
#     err_density = lambda1 * np.sum( jnp.square( (pred - target.T) @ ones ) ) 
#     error += err_density

#     return error
  
#   PIML_grad = jax.jit(jax.grad(loss))
  
#   beta = ELPH_utils.get_ridge_regression_weights(state, target, alpha=alpha)
#   #beta += np.random.normal(loc=0.0, scale=1e-2, size=beta.shape)
  
  
#   print('ridge regression loss: ' + str(loss(beta)))
#   plt.plot(np.ravel(PIML_grad(beta)))
#   plt.show()
  
#   GD_nmax = 50000
#   GD_nout = 10
#   GD_learning_rate = 1e-6
#   GD_momentum = 0.5
#   GD_diff = 0.0

#   for i in range(GD_nmax):
#       if(i%(GD_nmax//GD_nout) == 0):
#         print('loss: ' + str(loss(beta)))
#       GD_diff = GD_momentum * GD_diff - GD_learning_rate * PIML_grad(beta)
#       beta += GD_diff


#   plt.plot(np.ravel(PIML_grad(beta)))
#   plt.show()
    
#   return beta
  
  
  
# import optax as opt

# #def sgd(state, target, alpha=0.0, lambda1=0.0):
  
#   #def loss(beta):
    
#     #pred = state.T @ beta
#     #res = pred - target.T 

#     #err_lstsqs = jnp.linalg.norm(res, 'fro')**2
#     #error = err_lstsqs

#     #err_reg = alpha * jnp.linalg.norm(beta, 'fro')**2
#     #error +=  err_reg 
    
#     #ones = jnp.ones(target.T.shape[1])
#     #err_density = lambda1 * np.sum( jnp.square( (pred - target.T) @ ones ) ) 
#     #error += err_density

#     #return error
  
#   #PIML_grad = jax.jit(jax.grad(loss))
  
#   #beta = ELPH_utils.get_ridge_regression_weights(state, target, alpha=alpha)
#   ##beta += np.random.normal(loc=0.0, scale=1e-2, size=beta.shape)
  
  
#   #print('ridge regression loss: ' + str(loss(beta)))
#   #plt.plot(np.ravel(PIML_grad(beta)))
#   #plt.show()
  
#   #start_learning_rate = 1e-5
#   #optimizer = opt.adam(start_learning_rate)
  
#   #opt_state = optimizer.init(beta)
  
#   #sgd_nmax = 10000
#   #sgd_nout = 10
  
#   #for i in range(sgd_nmax):
#     #if(i%(sgd_nmax//sgd_nout) == 0):
#         #print('loss: ' + str(loss(beta)))
#     #grads = PIML_grad(beta)
#     #updates, opt_state = optimizer.update(grads, opt_state)
#     #beta = opt.apply_updates(beta, updates)

#   #plt.plot(np.ravel(PIML_grad(beta)))
#   #plt.show()
    
#   #return beta
  
  
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
  
  
  
  
  
