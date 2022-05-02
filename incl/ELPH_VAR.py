import numpy as np
import sys

sys.path.append("incl/")
import ELPH_utils
import ELPH_Scaler

    
class SVDVAR:
    
    def __init__(self, runs, rdim = 1, n_VAR_steps = 1, intercept = False, standardize=True, full_hist=False):
        self.runs = runs
        self.n_runs = len(runs)
        
        self.n_VAR_steps = n_VAR_steps
        self.rdim = rdim
        
        self.intercept = intercept
        self.standardize = standardize
        
        self.full_hist = full_hist
    
    def load_runs(self,runs):
        self.runs = runs
        self.n_runs = len(runs)
    
#     def __calc_reduced_coef_runs(self): 
#         self.U, self.S = ELPH_utils.get_SVD_from_runs(self.runs)
#         self.Uhat = self.U[:,:self.rdim]

#         self.red_coef_matrix = ELPH_utils.get_reduced_coef_matrix(self.runs, self.U, self.rdim)
    
#         if self.standardize:
#             self.red_coef_matrix, self.coef_mean, self.coef_std = ELPH_utils.standardize_data_matrix(self.red_coef_matrix)

#         self.coef_runs = ELPH_utils.get_coef_runs(self.red_coef_matrix, self.n_runs)
    
    
    def __build_VAR_vec(self, matrix, col, n_VAR_steps):
        VAR_vec = []
        for k in range(n_VAR_steps):
            if col+k < 0:
                VAR_vec.append(matrix[:,0])
            else:
                VAR_vec.append(matrix[:,col+k])
        return np.concatenate(VAR_vec, axis=0)
    
        
    def __build_VAR_training_matrices(self):
    
        state = []
        target = []

        for r in range(len(self.coef_runs)):

            if(self.full_hist == False):
                nCols = self.coef_runs[r].shape[1]-1
                Delta_j = self.n_VAR_steps-1
            else:
                nCols = self.coef_runs[r].shape[1]-self.n_VAR_steps
                Delta_j = 0
                
            nRows = self.coef_runs[r].shape[0]*self.n_VAR_steps

            run_VAR_matrix = np.zeros((nRows,nCols))
            for j in range(nCols):
                run_VAR_matrix[:,j] =self. __build_VAR_vec(self.coef_runs[r], j-Delta_j, self.n_VAR_steps)

            state.append(run_VAR_matrix)
            if self.full_hist == False:
                target.append(self.coef_runs[r][:,1:])
            else:
                target.append(self.coef_runs[r][:,self.n_VAR_steps:])

        state = np.concatenate(state, axis=1)
        target = np.concatenate(target, axis=1)

        return state,target
    
    
    def train(self, alpha=1e-6, rdim = None, n_VAR_steps = None, intercept=None, standardize=None, full_hist = None, method='ridge'):
        
        if rdim != None:
            self.rdim = rdim
        if n_VAR_steps != None:
            self.n_VAR_steps = n_VAR_steps
        if intercept != None:
            self.intercept = intercept
        if standardize != None:
            self.standardize = standardize
        if full_hist != None:
            self.full_hist = full_hist


        self.U, self.S = ELPH_utils.get_SVD_from_runs(self.runs)
        self.Uhat = self.U[:,:self.rdim]

        self.red_coef_matrix = ELPH_utils.get_reduced_coef_matrix(self.runs, self.U, self.rdim)

        if self.standardize:
            self.scaler = ELPH_Scaler.standardize_scaler(self.red_coef_matrix)
            self.scaler.train()
            self.red_coef_matrix = self.scaler.transform(self.red_coef_matrix)

        self.coef_runs = ELPH_utils.get_coef_runs(self.red_coef_matrix, self.n_runs)
        
        self.state, self.target = self.__build_VAR_training_matrices()
        
        if method == 'ridge':
            self.w = ELPH_utils.get_ridge_regression_weights(self.state, self.target, alpha)
        elif method == 'lstsq':
            self.w = np.asarray( np.linalg.lstsq(self.state.T, self.target.T, rcond = -1)[0] )
        else:
            print('unknown training method') 
            
        
    def predict_single_run(self, run):
        
        coef_run = self.Uhat.T @ run
        if self.standardize:
#             coef_run = (((coef_run.T - self.coef_mean)/self.coef_std)).T
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
            pred[:,j] = self.w.T @ self.__build_VAR_vec(pred, j-self.n_VAR_steps, self.n_VAR_steps)
        
        
        if self.standardize:
#             pred = ELPH_utils.destandardize_data_matrix(pred, self.coef_mean, self.coef_std)
            pred = self.scaler.inverse_transform(pred)
        pred = self.Uhat @ pred 
        
        return pred
    
    def get_error(self, run, pred=np.zeros(1), norm='fro', errSVD = False):
        if pred.size == 1:
            pred = self.predict_single_run(run)
        
        if errSVD == True:
            run = self.Uhat @ self.Uhat.T @ run
        
        if norm == 'fro': #Frobenius norm
            err = np.linalg.norm(run-pred, ord='fro')
        elif norm == 'var2': # mean of dynamical variable wise 2-norms
            err = np.mean( np.sum( np.square(run-pred), axis = 1 ) )
        elif norm =='max':
            err = np.abs(run-pred).max()
        else:
            print('unknown norm')
        
        return err
    
    def print_status(self):
        print('rdim: ', self.rdim)
        print('n_VAR_steps: ', self.n_VAR_steps)
        print('state shape: ', self.state.shape)
        print('target shape: ', self.target.shape)
        print('weights shape: ', self.w.shape)
        
    def score_multiple_runs(self,runs, **kwargs):
        scores = []
        for k in range(len(runs)):
            scores.append(self.get_error(runs[k], **kwargs))
        
        mean = np.mean(scores)
        return mean, scores
        
  
