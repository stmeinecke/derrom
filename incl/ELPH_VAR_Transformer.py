import numpy as np
import sys

sys.path.append("incl/")
import ELPH_utils 

class base_VAR_transformer:
    def __init__(self):
        pass
    
    def transform(self):
        raise NotImplementedError
        
        
class polynomial_features(base_VAR_transformer):
    
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
    
    def transform(self,data_matrix):
        
        nRows = self.__build_VAR_p_Vec(data_matrix[:,0], self.order).size
        nCols = data_matrix.shape[1]
        
        poly_features = np.zeros((nRows,nCols)) 
        
        for k in range( data_matrix.shape[1] ):
            poly_features[:,k] = self.__build_VAR_p_Vec(data_matrix[:,k], self.order)

        return poly_features
        
        
        
