import numpy as np


class base_transformer:
    def __init__(self):
        pass
    
    def transform(self):
        raise NotImplementedError
        
    def setup(self):
        raise NotImplementedError
        
        
class polynomial_features(base_transformer):
    
    def __init__(self, order=2):
        self.order = order
    
    def __build_VAR_p_Vec(self, VAR_vec, order):
        VAR_p_Vec = [VAR_vec]
        VARp = VAR_vec
        for p in range(1,order):
            VARp = np.outer(VAR_vec,VARp)
            VARp = VARp[ np.triu_indices(VARp.shape[0], m=VARp.shape[1]) ]
            VAR_p_Vec.append(VARp)
        return np.concatenate(VAR_p_Vec, axis=0)
    
    def transform(self,VAR_state_matrix):
        
        if VAR_state_matrix.shape[0] == 1:
            return self.__build_VAR_p_Vec(VAR_state_matrix[0], self.order)
          
        else:        
          nCols = self.__build_VAR_p_Vec(VAR_state_matrix[0], self.order).size
          nRows = VAR_state_matrix.shape[0]
          
          poly_features = np.zeros((nRows,nCols)) 
          
          for k in range( VAR_state_matrix.shape[0] ):
              poly_features[k] = self.__build_VAR_p_Vec(VAR_state_matrix[k], self.order)

          return poly_features
        
    def setup(self, n_VAR_features):
        pass
        
        
        
class ELM_features(base_transformer):
    
    def __init__(self, ELM_nodes = 200, ELM_weights_mean = 0.0, ELM_weights_std = 1.0, seed=817, activation_function=np.tanh):
        self.rng = np.random.default_rng(seed=seed)
        self.ELM_nodes = ELM_nodes
        self.ELM_weights_mean = ELM_weights_mean
        self.ELM_weights_std = ELM_weights_std
        self.projection_matrix = None
        self.bias_matrix = None
        self.activation_function = activation_function
        
    def setup(self, n_VAR_features):
        self.projection_matrix = self.rng.normal(self.ELM_weights_mean, self.ELM_weights_std/np.sqrt(n_VAR_features), (n_VAR_features, self.ELM_nodes))
        self.bias_matrix = self.rng.uniform(-1.0, 1.0, self.ELM_nodes)
    
    def transform(self,VAR_state_matrix):
        projected_data = self.activation_function( VAR_state_matrix @ self.projection_matrix + self.bias_matrix )
        state = np.concatenate( [VAR_state_matrix,projected_data], axis=1)

        return state
