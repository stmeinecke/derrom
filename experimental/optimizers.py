import sys
sys.path.append("../derrom/")
import derrom_optimizers


import numpy as np

from pysindy.optimizers import STLSQ
class stlsq(derrom_optimizers.base_optimizer):
    def __init__(self, alpha = 1e-6, threshold = 1e-6):
        self.alpha = alpha
        self.threshold = threshold
        pass
    
    def solve(self, feature_matrix, target_matrix):
        opt = STLSQ(alpha=self.alpha, threshold=self.threshold)
        opt.fit(feature_matrix, target_matrix)
        return opt.coef_.T



import jax
import jax.numpy as jnp
import optax as opt
import matplotlib.pyplot as plt
import ELPH_dyn

        
class PIML_adam(derrom_optimizers.base_optimizer):
    def __init__(self, dim_reducer = None, alpha = 1e-6, lambda1=0.0, mini_batch_size = 50, epochs = 1):
        
        assert dim_reducer != None
        
        self.dim_reducer = dim_reducer
        
        self.alpha = alpha
        self.lambda1 = lambda1
        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        
    def solve(self, feature_matrix, target_matrix):
        
        red_dim = target_matrix.shape[1]
        full_dim = self.dim_reducer.reconstruct(target_matrix[:1]).shape[1]
        #print(full_dim)
        
        dk=4./full_dim
        n_states_vec = np.zeros(full_dim)
        for k in range(full_dim):
            n_states_vec[k] = ELPH_dyn.get_k(dk,k)*dk/2./np.pi
  
        
        def loss(beta, feature_matrix, target_matrix):
    
            #pred = beta.T @ feature_matrix
            pred = feature_matrix @ beta
            res = pred - target_matrix

            err_lstsqs = jnp.sum(jnp.square(res))
            #err_lstsqs = jnp.amax(jnp.abs(res))
            error = err_lstsqs

            err_reg = self.alpha * jnp.sum(jnp.square(beta))
            error +=  err_reg 

            ones = jnp.ones(target_matrix.shape[0])
            #err_density = self.lambda1 * np.sum( jnp.square( self.dim_reducer.reconstruct(pred - target_matrix) @ n_states_vec ) )
            err_density = self.lambda1 * np.sum( jnp.square( (pred - target_matrix) @ self.dim_reducer.U[:,:red_dim].T @ n_states_vec ) ) 
            
            error += err_density

            return error
  
        PIML_grad = jax.jit(jax.grad(loss))
        
        ridge_optimizer = ridge(alpha=self.alpha)
        beta = ridge_optimizer.solve(feature_matrix, target_matrix)
#         beta += np.random.normal(loc=0.0, scale=1e-2, size=beta.shape)


        print('ridge regression loss: ' + str(loss(beta, feature_matrix, target_matrix)))
        #plt.plot(np.ravel(PIML_grad(beta, feature_matrix, target_matrix)))
        #plt.show()


#         print(feature_matrix.shape)
#         print(target_matrix.shape)

#         n_batches = 1000
        n_batches = feature_matrix.shape[0]/self.mini_batch_size

        rng = np.random.default_rng(42)
  
  
        def fit(params, optimizer):
            opt_feature_matrix = optimizer.init(params)

            @jax.jit
            def step(params, opt_feature_matrix, batch, labels):
                loss_value, grads = jax.value_and_grad(loss)(params, batch, labels)
                updates, opt_feature_matrix = optimizer.update(grads, opt_feature_matrix, params)
                params = opt.apply_updates(params, updates)
                return params, opt_feature_matrix, loss_value

            for k in range(self.epochs):
                permuted_inds = rng.permutation(feature_matrix.shape[0])
                TRAINING_DATA = np.array_split( np.array(feature_matrix)[permuted_inds] , n_batches, axis=0)
                LABELS = np.array_split( np.array(target_matrix)[permuted_inds] , n_batches, axis=0)

                for i, (batch, labels) in enumerate(zip(TRAINING_DATA, LABELS)):
                    params, opt_feature_matrix, loss_value = step(params, opt_feature_matrix, batch, labels)
#                     if i % 100 == 0:
#                         print(f'step {i}, loss: {loss_value}')
                        
                print('epoch:', k+1, 'loss:', loss(params, feature_matrix, target_matrix) )

            return params

        
        optimizer = opt.adam(learning_rate=1e-5)
        beta = fit(beta, optimizer)
  
        print('regression loss: ' + str(loss(beta, feature_matrix, target_matrix)))

        return beta 
