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
    def __init__(self, sorted=False):
        self.sorted = sorted
        pass
    
    def train(self, data_matrix):
        self.full_dim = data_matrix.shape[0]
        
        if self.sorted:
            FT = np.fft.rfft(data_matrix, axis=0)

            real_matrix = np.zeros((FT.shape[0]*2,FT.shape[1]))

            real_matrix[::2] = np.real(FT)
            real_matrix[1::2] = np.imag(FT)

            self.mean_coefs = np.mean(real_matrix, axis=1)

            self.sort_inds = np.flip(np.argsort(np.abs(self.mean_coefs)))
            self.unsort_inds = np.argsort(self.sort_inds)
        
        
    def reduce(self, data_matrix, prdim):
        
        assert prdim%2==0, "prdim must be an even number for the FFT dim reducer"
        
        FT = np.fft.rfft(data_matrix, axis=0)
  
        real_matrix = np.zeros((FT.shape[0]*2,FT.shape[1]))
        
        real_matrix[::2] = np.real(FT)
        real_matrix[1::2] = np.imag(FT)
        
        if self.sorted:
            return real_matrix[self.sort_inds][:prdim]
        else:
            return real_matrix[:prdim]
    
    def expand(self, coef_matrix):
        
        if self.sorted:
            real_matrix = np.zeros((2 * self.full_dim, coef_matrix.shape[1]))
            real_matrix[:coef_matrix.shape[0]] = coef_matrix
            real_matrix = real_matrix[self.unsort_inds]
          
            complex_matrix = real_matrix[::2] + 1.j*real_matrix[1::2]  
        else:
            complex_matrix = coef_matrix[::2] + 1.j*coef_matrix[1::2]
            
        iFT = np.fft.irfft(complex_matrix, n=self.full_dim, axis=0)
        
        return iFT


from scipy.special import eval_hermite
from scipy.optimize import minimize_scalar
class Hermite(base_dim_reducer):
    def __init__(self, x):
        self.x = x
        pass
      
    def train(self, data_matrix):
        self.full_dim = data_matrix.shape[0]
        
        self.H_matrix = np.zeros((self.full_dim, self.full_dim))
        for k in range(self.full_dim):
            self.H_matrix[k]  = eval_hermite(k,self.x) * np.exp(-0.5*self.x**2) / np.sqrt(np.sqrt(np.pi)*(2**k)*np.math.factorial(k))
          
        #def loss(x_max):
            #x_test = np.linspace(0,x_max,self.full_dim)
            #for k in range(self.full_dim):
                #self.H_matrix[k] = eval_hermite(k,x_test) * np.exp(-0.5*x_test**2) / np.sqrt(np.sqrt(np.pi)*(2**k)*np.math.factorial(k))
                
            #apprx = self.expand(self.reduce(data_matrix,self.full_dim))
            
            ##return np.linalg.norm(data_matrix-apprx, ord='fro')
            #return np.abs(np.ravel(data_matrix-apprx)).max()
          
        #res = minimize_scalar(loss, bounds=(0,40))
        #res.x
        #print(res.x)
        
        #self.x = np.linspace(0,res.x,self.full_dim)
        ##self.H_matrix = np.zeros((self.full_dim, self.full_dim))
        #for k in range(self.full_dim):
            #self.H_matrix[k]  = eval_hermite(k,self.x) * np.exp(-0.5*self.x**2) / np.sqrt(np.sqrt(np.pi)*(2**k)*np.math.factorial(k))
        
        
    def reduce(self,data_matrix,prdim):
        
        return self.H_matrix[:prdim] @ data_matrix
        
    def expand(self, coef_matrix):
        
        dim = coef_matrix.shape[0]
      
        return np.linalg.pinv(self.H_matrix[:dim]) @ coef_matrix
        
            
        
        
        
    
  
