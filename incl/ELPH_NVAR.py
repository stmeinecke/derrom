import numpy as np
import sys

sys.path.append("incl/")
import ELPH_utils
from ELPH_VAR import SVDVAR


class SVDNVAR(SVDVAR):
  
  def __init__(self, runs, rdim = 1, n_VAR_steps = 1, NVAR_p = 1):
    super().__init__(runs, rdim, n_VAR_steps) 
    self.NVAR_p = NVAR_p
  
  
  def build_VAR_p_Vec(self, VAR_vec, order=2):
    VAR_p_Vec = [VAR_vec]
    VARp = VAR_vec
    for p in range(1,order):
      VARp = np.outer(VARp,VAR_vec)
#         print('p: ', p+1, " VARp shape: ", VARp.shape)
#         print(VARp)
#         print(np.tril(VARp))
      VARp = VARp[ np.tril_indices(VARp.shape[0], m=VARp.shape[1]) ]
      VAR_p_Vec.append(VARp)
    return np.concatenate(VAR_p_Vec, axis=0)
  
    
  def __build_NVAR_training_matrices(self):
    
    #print(type(self.VAR_state[:,0]))
    
    nRows = self.build_VAR_p_Vec(self.VAR_state[:,0], order=self.NVAR_p).size
    nCols = self.VAR_state.shape[1]
  
    NVAR_state = np.zeros((nRows,nCols)) 
    #print(NVAR_state.shape)
    
    for k in range( NVAR_state.shape[1] ):
      NVAR_state[:,k] = self.build_VAR_p_Vec(self.VAR_state[:,k], order=self.NVAR_p)
                          
    return NVAR_state
  
  
  def train(self, alpha=1e-6, rdim = 0, n_VAR_steps = 0, NVAR_p = 0, method='ridge'):
        
    if rdim != 0:
        self.rdim = rdim
    if n_VAR_steps != 0:
        self.n_VAR_steps = n_VAR_steps
    if NVAR_p != 0:
        self.NVAR_p = NVAR_p

    self._SVDVAR__calc_reduced_coef_runs()

    self.VAR_state, self.target = self._SVDVAR__build_VAR_training_matrices()
                          
    self.NVAR_state = self.__build_NVAR_training_matrices()

    if method == 'ridge':
        self.w = ELPH_utils.get_ridge_regression_weights(self.NVAR_state, self.target, alpha)
    elif method == 'lstsq':
        self.w = np.asarray( np.linalg.lstsq(self.NVAR_state.T, self.target.T, rcond = -1)[0] )
    else:
        print('unknown training method') 
                          
                          
  def predict_single_run(self, run):

      coef_run = self.Uhat.T @ run
      coef_run = (((coef_run.T - self.coef_mean)/self.coef_std)).T

      pred = np.zeros(coef_run.shape)

      for l in range(self.n_VAR_steps):
          pred[:,l] = coef_run[:,l]

      for j in range(self.n_VAR_steps,pred.shape[1]):
          VARpredList = []
          for l in range(self.n_VAR_steps):
              VARpredList.append( pred[:,j-self.n_VAR_steps+l] )
          
          VAR_vec = np.concatenate( VARpredList, axis=0 )
                          
          pred[:,j] = self.w.T @ self.build_VAR_p_Vec(VAR_vec, order=self.NVAR_p)

      pred = ELPH_utils.destandardize_data_matrix(pred, self.coef_mean, self.coef_std)
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
