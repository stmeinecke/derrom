import numpy as np
import sys

sys.path.append("incl/")
import ELPH_utils


class SVDAPPRX:
    
    def __init__(self, runs, rdim = 1):
        self.runs = runs
        self.n_runs = len(runs)
        self.rdim = rdim
    
    def load_runs(self, runs):
        self.runs = runs
        self.n_runs = len(runs)
    
    def train(self, rdim=None, wcols=np.zeros(1)):
        
        if rdim != None:
            self.rdim = rdim
        if wcols.size != 1:
            self.wcols = wcols
        else:
            self.wcols = np.ones(self.runs[0].shape[1])
            
#         self.U, self.S = ELPH_utils.get_SVD_from_runs(self.runs)

        data_matrix = np.concatenate(self.runs,axis=1)
    
        n_cols = self.runs[0].shape[1]
        for r in range(len(self.runs)):
            for t in range(n_cols):
                data_matrix[:,r*n_cols+t] *= self.wcols[t]

        self.U,self.S,V = np.linalg.svd(data_matrix, full_matrices=False)

        self.Uhat = self.U[:,:self.rdim]
        
    def approx_single_run(self, run, rdim=None):
        if rdim == None:
            rdim = self.rdim
                
        return self.U[:,:rdim] @ self.U[:,:rdim].T @ run
    
    def get_error(self, run, approx=np.zeros(1), rdim=None, norm='fro'):
        
        if rdim == None:
            rdim = self.rdim
        
        if approx.size == 1:
            approx = self.approx_single_run(run, rdim=rdim)
            
        if norm=='fro':
            err = np.linalg.norm(run-approx, ord='fro')       
        elif norm == 'var2': # mean of dynamical variable wise 2-norms
            err = np.mean( np.sum( np.square(run-approx), axis = 1 ) )
        elif norm =='max':
            err = np.abs(run-approx).max()
        else:
            print('unknown norm') 

    
          
        return err
    
    def score_multiple_runs(self,runs, **kwargs):
        scores = []
        for k in range(len(runs)):
            scores.append(self.get_error(runs[k], **kwargs))
        
        mean = np.mean(scores)
        return mean, scores
