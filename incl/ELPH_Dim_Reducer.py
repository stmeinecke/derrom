import numpy as np
import sys

sys.path.append("incl/")
import ELPH_utils 

class base_dim_reducer:
    def __init__(self):
        pass
    
    def train(self):
        raise NotImplementedError
        
    def reduce(self):
        raise NotImplementedError
        
    def expand(self):
        raise NotImplementedError
        
        
class SVD(base_dim_reducer):
    def __init__(self):
        pass
    
    def train(self, data_matrix):
        self.U,self.S = np.linalg.svd(data_matrix, full_matrices=False)[:2]
        
    def reduce(self, data_matrix, prdim):
        return self.U[:,:prdim].T @ data_matrix
    
    def expand(self, coef_matrix):
        dim = coef_matrix.shape[0]
        return self.U[:,:dim] @ coef_matrix 
      
      
class FFT(base_dim_reducer):
    def __init__(self):
        pass
    
    def train(self, data_matrix):
        self.full_dim = data_matrix.shape[0]
        
    def reduce(self, data_matrix, prdim):
        
        assert prdim%2==0, "prdim must be an even number for the FFT dim reducer"
        
        FT = np.fft.rfft(data_matrix, axis=0)
  
        real_matrix = np.zeros((FT.shape[0]*2,FT.shape[1]))
        
        real_matrix[::2] = np.real(FT)
        real_matrix[1::2] = np.imag(FT)

        return real_matrix[:prdim]
    
    def expand(self, coef_matrix):
        
        complex_matrix = coef_matrix[::2] + 1.j*coef_matrix[1::2]
        iFT = np.fft.irfft(complex_matrix, n=self.full_dim, axis=0)
        
        return iFT
