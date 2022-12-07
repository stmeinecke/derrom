import numpy as np


class base_optimizer:
    def __init__(self):
        pass
    
    def solve(self):
        raise NotImplementedError


class lstsqrs(base_optimizer):
    def __init__(self):
        pass
    
    def solve(self, feature_matrix, target_matrix):
        return np.asarray( np.linalg.lstsq(feature_matrix, target_matrix, rcond = -1)[0] )
      
      

class ridge(base_optimizer):
    def __init__(self, alpha = 1e-6):
        self.alpha = alpha
    
    def solve(self, feature_matrix, target_matrix):
        return np.linalg.inv(feature_matrix.T @ feature_matrix + self.alpha * np.identity(feature_matrix.shape[1])) @ feature_matrix.T @ target_matrix


  

  
  
  
