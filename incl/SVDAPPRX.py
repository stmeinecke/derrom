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
    
    def train(self, rdim=None):
        
        if rdim != None:
            self.rdim = rdim
            
        self.U, self.S = ELPH_utils.get_SVD_from_runs(self.runs)
        self.Uhat = self.U[:,:self.rdim]
        
    def approx_single_run(self, run, rdim=None):
        if rdim == None:
            rdim = self.rdim
        return self.U[:,:rdim] @ self.U[:,:rdim].T @ run
    
    def get_error(self, run, approx=np.zeros(1), rdim=None):
        
        if rdim == None:
            rdim = self.rdim
        
        if approx.size == 1:
            approx = self.approx_single_run(run, rdim=rdim)
            
        err = np.linalg.norm(run-approx, ord='fro')
        return err
    
    def score_multiple_runs(self,runs, **kwargs):
        scores = []
        for k in range(len(runs)):
            scores.append(self.get_error(runs[k], **kwargs))
        
        mean = np.mean(scores)
        return mean, scores
