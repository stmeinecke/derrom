import numpy as np


class data_scaler:
    def __init__(self):
        self.is_trained = False
        
    def train(self):
        raise NotImplementedError
        
    def transform(self):
        raise NotImplementedError
        
    def inverse_transform(self):
        raise NotImplementedError
        

class standardize_scaler(data_scaler):
    def __init__(self, rel_scale=1.0):
        super().__init__() 
        self.rel_scale = rel_scale
   
    def train(self, data_matrix):
        data_matrix = data_matrix
      
        self.mean = np.mean(data_matrix, axis=0)
        self.scale = np.std(data_matrix, axis=0) / self.rel_scale
        
        #do not scale constant features
        for k in range(self.scale.size):
            if np.abs(self.scale[k]) < 1e-8:
                self.scale[k] = 1.0
        
    def transform(self, data_matrix):
        return (data_matrix - self.mean)/self.scale
    
    def inverse_transform(self, data_matrix):
        return (data_matrix * self.scale) + self.mean
    
class normalize_scaler(data_scaler):
    def __init__(self, rel_scale=1.0):
        super().__init__()
        self.rel_scale = rel_scale
   
    def train(self, data_matrix):
        
        self.max = np.amax(data_matrix, axis=0)
        self.min = np.amin(data_matrix, axis=0)
        self.scale = (self.max - self.min)/self.rel_scale
        
        self.rel_scale_vec = np.full(self.scale.shape, self.rel_scale)
        #do not scale constant features
        for k in range(self.scale.size):
            if np.abs(self.scale[k]) < 1e-8:
                self.scale[k] = 1.0
                self.rel_scale_vec[k] = 0.0
        
    def transform(self, data_matrix):
        n_features = data_matrix.shape[1]
        
        return ( ( (data_matrix - self.min[:n_features])/self.scale[:n_features]) - 0.5*self.rel_scale_vec[:n_features])
    
    def inverse_transform(self, data_matrix):
        n_features = data_matrix.shape[1]
        return ( ( (data_matrix + 0.5*self.rel_scale_vec[:n_features]) * self.scale[:n_features])+self.min[:n_features] )
      
      
class tanh_scaler(data_scaler):
    def __init__(self, arg_scale=1.0, out_scale=1.0):
        super().__init__()
        self.arg_scale = arg_scale
        self.out_scale = out_scale
   
    def train(self, data_matrix):
        
        self.max = np.amax(data_matrix, axis=0)
        self.min = np.amin(data_matrix, axis=0)
        self.scale = (self.max - self.min)/self.arg_scale
        
        self.arg_scale_vec = np.full(self.scale.shape, self.arg_scale)
        #do not scale constant features
        for k in range(self.scale.size):
            if np.abs(self.scale[k]) < 1e-8:
                self.scale[k] = 1.0
                self.arg_scale_vec[k] = 0.0
        
    def transform(self, data_matrix):
        return np.tanh( (data_matrix - self.min)/self.scale - 0.5*self.arg_scale_vec ) * self.out_scale
    
    def inverse_transform(self, data_matrix):
        return ( (np.arctanh(data_matrix/self.out_scale) + 0.5*self.arg_scale_vec) * self.scale)+self.min
