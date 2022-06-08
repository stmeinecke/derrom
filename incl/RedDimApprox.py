import numpy as np
import sys

sys.path.append("incl/")
import ELPH_utils


class RedDimApprox:
    
    def __init__(self, runs = None, dim_reducer = None, rdim = 1):
        self.runs = runs
        if runs != None:
            self.n_runs = len(runs)
        self.rdim = rdim
        self.dim_reducer = dim_reducer
    
    def load_runs(self, runs):
        self.runs = runs
        self.n_runs = len(runs)
    
    def train(self, rdim=None, dim_reducer = None):
        
        if self.runs == None:
            raise ValueError('no runs loaded')
        
        if dim_reducer != None:
            self.dim_reducer = dim_reducer
            
        if self.dim_reducer == None:
            raise ValueError('no dim reducer object as been passed')

        data_matrix = np.concatenate(self.runs,axis=1)
        
        self.dim_reducer.train(data_matrix)
        
    def approx_single_run(self, run, rdim=None):
        if rdim == None:
            rdim = self.rdim
               
        return self.dim_reducer.expand( self.dim_reducer.reduce(run, rdim) )
    
    def get_error(self, run, approx=np.zeros(1), rdim=None, norm='std'):
        
        if rdim == None:
            rdim = self.rdim
        
        if approx.size == 1:
            approx = self.approx_single_run(run, rdim=rdim)
             
        if norm=='fro':
            err = np.linalg.norm(run-approx, ord='fro')       
        elif norm =='max':
            err = np.abs(run-approx).max()
        elif norm == 'std':
            err = np.std(np.ravel(run-approx))
        else:
            print('unknown norm') 

        return err
    
    def score_multiple_runs(self,runs,**kwargs):
        scores = []
        for k in range(len(runs)):
            scores.append(self.get_error(runs[k], **kwargs))
        
        mean = np.mean(scores)
        return mean, scores
