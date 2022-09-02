import numpy as np


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

    
from pysindy.optimizers import STLSQ
class stlsq(base_optimizer):
    def __init__(self, alpha, threshold):
        self.alpha = alpha
        self.threshold = threshold
        pass
    
    def solve(self, state, target):
        opt = STLSQ(alpha=self.alpha, threshold=self.threshold)
        opt.fit(state.T, target.T)
        return opt.coef_.T



import jax
import jax.numpy as jnp
import optax as opt
import matplotlib.pyplot as plt

        
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
  

  
  
  
