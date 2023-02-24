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
    """
    Feature scaler which subtracts the mean and devices by the standard deviation for each feature.
    
    Parameters
    ----------
    rel_scale : float
        Standard deviation of the resulting scaled features. Default is 1.
    """
    
    def __init__(self, rel_scale=1.0):
        super().__init__() 
        self.rel_scale = rel_scale
   
    def train(self, data_matrix):
        """
        Computes the mean and standard deviation of each feature. 
        
        Detects quasi-constant features, i.e. features with a standard deviation < 1e-8 and does not scale them.
        
        Parameters
        ----------
        data_matrix : 2D numpy.ndarray
            Training data matrix, where the data vectors are stored in the rows (first index). Hence, the features are defined by their column index (second index).
        """      
        self.mean = np.mean(data_matrix, axis=0)
        self.scale = np.std(data_matrix, axis=0) / self.rel_scale
        
        #do not scale constant features
        for k in range(self.scale.size):
            if np.abs(self.scale[k]) < 1e-8:
                self.scale[k] = 1.0
        
    def transform(self, data_matrix):
        """
        Applies standardization to each feature.
        
        Parameters
        ----------
        data_matrix : 2D numpy.ndarray
            unscaled data matrix with the datavectors as rows.
            
            
        Returns
        -------
        scaled_data_matrix : 2D numpy.ndarray
            scaled data matrix
        """
        return (data_matrix - self.mean)/self.scale
    
    def inverse_transform(self, data_matrix):
        """
        Reverses standardization of each feature by multiplying by the standard deviation and adding the mean (of the training data)
        
        Parameters
        ----------
        data_matrix : 2D numpy.ndarray
            scaled data matrix with the datavectors as rows.
            
            
        Returns
        -------
        scaled_data_matrix : 2D numpy.ndarray
            unscaled data matrix
        """
        return (data_matrix * self.scale) + self.mean
    
class normalize_scaler(data_scaler):
    """
    Feature scaler which linearly maps each feature to a specified symmetric range around the origin.
    
    Parameters
    ----------
    rel_scale : float
        Range to which the features are mapped. The default rel_scale = 1.0 maps to :math:`[-0.5,0.5]`
    """
    def __init__(self, rel_scale=1.0):
        super().__init__()
        self.rel_scale = rel_scale
   
    def train(self, data_matrix):
        """
        Determines the range if each feature, i.e., the maximum and minimum value.
        
        Detects quasi-constant features, i.e. features with a standard deviation < 1e-8 and does not scale them.
        
        Parameters
        ----------
        data_matrix : 2D numpy.ndarray
            Training data matrix, where the data vectors are stored in the rows (first index). Hence, the features are defined by their column index (second index).
        """  
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
        """
        Applies normalization to each feature.
        
        Parameters
        ----------
        data_matrix : 2D numpy.ndarray
            unscaled data matrix with the datavectors as rows.
            
            
        Returns
        -------
        scaled_data_matrix : 2D numpy.ndarray
            scaled data matrix
        """
        return ( ( (data_matrix - self.min)/self.scale) - 0.5*self.rel_scale_vec)
    
    def inverse_transform(self, data_matrix):
        """
        Reverses normalization of each feature
        
        Parameters
        ----------
        data_matrix : 2D numpy.ndarray
            scaled data matrix with the datavectors as rows.
            
            
        Returns
        -------
        scaled_data_matrix : 2D numpy.ndarray
            unscaled data matrix
        """
        return ( ( (data_matrix + 0.5*self.rel_scale_vec) * self.scale)+self.min )
      
      
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
