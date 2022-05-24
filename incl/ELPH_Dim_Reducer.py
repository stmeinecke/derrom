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
        
        FT = np.fft.rfft(data_matrix, axis=0)[:prdim//2]
        
        real_matrix = np.concatenate((np.real(FT),np.imag(FT)[1:]), axis=0)

        return real_matrix
    
    def expand(self, coef_matrix):
        
        ind_split = (coef_matrix.shape[0]+1)//2
        
        complex_matrix = coef_matrix[:ind_split] + 1.j*np.concatenate( (np.zeros((1,coef_matrix.shape[1])), coef_matrix[ind_split:]), axis=0) 
        
        iFT = np.fft.irfft(complex_matrix, n=self.full_dim, axis=0)
        
        return iFT
