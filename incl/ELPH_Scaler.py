import numpy as np
import sys

sys.path.append("incl/")
import ELPH_utils

class data_scaler:
    def __init__(self, data_matrix):
        self.data_matrix = data_matrix
    
    def load_data_matrix(self,data_matrix):
        self.data_matrix = data_matrix

class standardize_scaler(data_scaler):
    def __init__(self, data_matrix):
        super().__init__(data_matrix) 
   
    def train(self):
        self.mean = np.mean(self.data_matrix, axis=1)
        self.std = np.std(self.data_matrix, axis=1)
        
    def transform(self, data_matrix):
        return ((data_matrix.T - self.mean)/self.std).T
    
    def inverse_transform(self, data_matrix):
        return ( (data_matrix.T * self.std)+self.mean ).T
    
class normalize_scaler(data_scaler):
    def __init__(self, data_matrix):
        super().__init__(data_matrix) 
   
    def train(self):
        self.max = np.amax(self.data_matrix, axis=1)
        self.min = np.amin(self.data_matrix, axis=1)
        self.scale = self.max - self.min
        
    def transform(self, data_matrix):
        return ( ( (data_matrix.T - self.min)/self.scale) - 0.5).T
    
    def inverse_transform(self, data_matrix):
        return ( ( (data_matrix.T + 0.5) * self.scale)+self.min ).T