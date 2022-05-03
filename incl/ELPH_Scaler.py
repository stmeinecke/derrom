import numpy as np
import sys

sys.path.append("incl/")
import ELPH_utils

class data_scaler:
    def __init__(self):
      self.is_trained = False

class standardize_scaler(data_scaler):
    def __init__(self):
        super().__init__() 
   
    def train(self, data_matrix):
        self.data_matrix = data_matrix
      
        self.mean = np.mean(self.data_matrix, axis=1)
        self.std = np.std(self.data_matrix, axis=1)
        
    def transform(self, data_matrix):
        return ((data_matrix.T - self.mean)/self.std).T
    
    def inverse_transform(self, data_matrix):
        return ( (data_matrix.T * self.std)+self.mean ).T
    
class normalize_scaler(data_scaler):
    def __init__(self, rel_scale=1.0):
        super().__init__()
        self.rel_scale = rel_scale
   
    def train(self, data_matrix):
        self.data_matrix = data_matrix
        
        self.max = np.amax(self.data_matrix, axis=1)
        self.min = np.amin(self.data_matrix, axis=1)
        self.scale = (self.max - self.min)/self.rel_scale
        
    def transform(self, data_matrix):
        return ( ( (data_matrix.T - self.min)/self.scale) - 0.5*self.rel_scale).T
    
    def inverse_transform(self, data_matrix):
        return ( ( (data_matrix.T + 0.5*self.rel_scale) * self.scale)+self.min ).T
