import numpy as np
import sys

sys.path.append("incl/")
import ELPH_utils
from ELPH_VAR import SVDVAR

from sklearn.linear_model import MultiTaskElasticNet
from sklearn.linear_model import MultiTaskLasso
from sklearn.linear_model import Lasso

from pysindy.optimizers import STLSQ


class SVDNVAR(SVDVAR):
  
    def __init__(self, runs, rdim = 1, n_VAR_steps = 1, NVAR_p = 1):
        super().__init__(runs, rdim, n_VAR_steps) 
        self.NVAR_p = NVAR_p
    
    
    def build_VAR_p_Vec(self, VAR_vec, order=2):
        VAR_p_Vec = [VAR_vec]
        VARp = VAR_vec
        for p in range(1,order):
            VARp = np.outer(VAR_vec,VARp)
            VARp = VARp[ np.triu_indices(VARp.shape[0], m=VARp.shape[1]) ]
            VAR_p_Vec.append(VARp)
        return np.concatenate(VAR_p_Vec, axis=0)
  
  
  #def build_VAR_p_Vec(self, VAR_vec, order=2):
    #VAR_p_Vec = [VAR_vec]
    #VARp = VAR_vec
    #for p in range(1,order):
      #VARp = np.multiply(VARp,VAR_vec)
      #VAR_p_Vec.append(VARp)
    #return np.concatenate(VAR_p_Vec, axis=0)
  
    
    def __build_NVAR_training_matrices(self):
    
        nRows = self.build_VAR_p_Vec(self.VAR_state[:,0], order=self.NVAR_p).size
        nCols = self.VAR_state.shape[1]

        NVAR_state = np.zeros((nRows,nCols)) 
        #print(NVAR_state.shape)

        for k in range( NVAR_state.shape[1] ):
            NVAR_state[:,k] = self.build_VAR_p_Vec(self.VAR_state[:,k], order=self.NVAR_p)

        return NVAR_state
  
  
    def train(self, rdim = None, n_VAR_steps = None, NVAR_p = None, intercept=None, full_hist=None, scaler = None, method='ridge', **kwargs):
        
        if rdim != None:
            self.rdim = rdim
        if n_VAR_steps != None:
            self.n_VAR_steps = n_VAR_steps
        if NVAR_p != None:
            self.NVAR_p = NVAR_p
        if intercept != None:
            self.intercept = intercept
        if full_hist != None:
            self.full_hist = full_hist
        if scaler != None:
            self.scaler = scaler
            self.standardize = True
        else:
            self.standardize = False


        self.U, self.S = ELPH_utils.get_SVD_from_runs(self.runs)
        self.Uhat = self.U[:,:self.rdim]

        self.red_coef_matrix = ELPH_utils.get_reduced_coef_matrix(self.runs, self.U, self.rdim)

        if self.standardize:
            self.scaler.train(self.red_coef_matrix)
            self.red_coef_matrix = self.scaler.transform(self.red_coef_matrix)

        self.coef_runs = ELPH_utils.get_coef_runs(self.red_coef_matrix, self.n_runs)

        self.VAR_state, self.target = self._SVDVAR__build_VAR_training_matrices()

        self.NVAR_state = self.__build_NVAR_training_matrices()


        if intercept:
            self.NVAR_state = np.concatenate( [self.NVAR_state, np.ones((1,self.NVAR_state.shape[1]))], axis=0 )

        if method == 'ridge':
            self.w = ELPH_utils.get_ridge_regression_weights(self.NVAR_state, self.target, **kwargs)
        elif method == 'lstsq':
            self.w = np.asarray( np.linalg.lstsq(self.NVAR_state.T, self.target.T, rcond = -1)[0] )
        elif method == 'mten':
            MTEN = MultiTaskElasticNet(alpha=alpha, **kwargs)
            MTEN.fit(self.NVAR_state.T,self.target.T)
            self.w = MTEN.coef_.T
        elif method == 'mtl':
            MTL = MultiTaskLasso(alpha=alpha, **kwargs)
            MTL.fit(self.NVAR_state.T,self.target.T)
            self.w = MTL.coef_.T
        elif method == 'lasso':
            L = Lasso(alpha=alpha, **kwargs)
            L.fit(self.NVAR_state.T,self.target.T)
            self.w = L.coef_.T
        elif method == 'stlsq':
            opt = STLSQ(**kwargs)
            opt.fit(self.NVAR_state.T, self.target.T)
            self.w = opt.coef_.T
        else:
            print('unknown training method') 
                          

    def predict_single_run(self, run):

        coef_run = self.Uhat.T @ run
        if self.standardize:
            coef_run = self.scaler.transform(coef_run)

        pred = np.zeros(coef_run.shape)

        if (self.full_hist == False):
            j_start = 1
            pred[:,0] = coef_run[:,0]
        else:
            j_start = self.n_VAR_steps
            for l in range(self.n_VAR_steps):
                pred[:,l] = coef_run[:,l]

        for j in range(j_start,pred.shape[1]):
        
            VAR_vec = self._SVDVAR__build_VAR_vec(pred, j-self.n_VAR_steps, self.n_VAR_steps)

            NVAR_vec = self.build_VAR_p_Vec(VAR_vec, order=self.NVAR_p)

            if self.intercept:
                NVAR_vec = np.append(NVAR_vec, 1.0)
                          
            pred[:,j] = self.w.T @ NVAR_vec

        if self.standardize:
            pred = self.scaler.inverse_transform(pred)
        pred = self.Uhat @ pred 

        return pred
                          
                          
    def print_status(self):
        print('rdim: ', self.rdim)
        print('n_VAR_steps: ', self.n_VAR_steps)
        print('NVAR_p: ', self.NVAR_p)
        print('VAR state shape: ', self.VAR_state.shape)
        print('NVAR state shape: ', self.NVAR_state.shape)
        print('target shape: ', self.target.shape)
        print('weights shape: ', self.w.shape)
