import numpy as np



#######################################
### KFold cross validation
#######################################

def get_KFold_trajectories(trajectories, folds=5):
    KFold_trajectories = np.array_split(trajectories, folds) #numpy returns list of ndarrays
    KFold_trajectories = [list(array) for array in KFold_trajectories] #covert ndarrays back to list such that KFold_trajectories is a list of lists of ndarrays (the individual trajectories)
    return KFold_trajectories
  
  
def get_KFold_CV_scores(model, trajectories, folds=5, seed=817, norms = ['std'], train_kwargs={}):
    
    #create shuffled copy of the trajectories
    rng = np.random.default_rng(seed=seed)
    strajectories = trajectories.copy()
    rng.shuffle(strajectories)
    
    #split trajectories into folds
    KFold_trajectories = get_KFold_trajectories(strajectories, folds=folds)
    
    scores = [[] for n in range(len(norms))]  #of the individual folds - for each error norm in the norms list
    
    for k in range(folds):
        train_trajectories = KFold_trajectories.copy()
        test_trajectories = train_trajectories.pop(k) #test_trajectories = the kth fold, train_trajectories to remaining folds

        train_trajectories = [item for sublist in train_trajectories for item in sublist] #unpack the training folds into one flattened list

        #train the model on the training trajectories and get scores from the testing trajectories
        model.load_trajectories(train_trajectories) 
        model.train(**train_kwargs)
        
        #score the test trajectories for each error norm in the norms list
        for l,norm in enumerate(norms): 
            fold_mean_score, fold_all_scores = model.score_multiple_trajectories(test_trajectories, norm=norm)
            scores[l].append(fold_all_scores)
            
    for n in range(len(norms)):
        scores[n] = [item for sublist in scores[n] for item in sublist] #unpack the training folds into one flattened list
    
    return scores


#######################################
### save and load trajectories
#######################################

def save_trajectories(trajectories, filename='../trajectories'):
    np.savez(filename, trajectories)
    
def load_trajectories(filename='../trajectories.npz'):
    
    from os.path import exists

    if not exists(filename):
        print('trajectories file ot found')
        return None
    else:
        npz_trajectories = np.load(filename)
        trajectories = np.split(npz_trajectories['arr_0'], npz_trajectories['arr_0'].shape[0], axis=0)

        for k in range(len(trajectories)):
            trajectories[k] = np.reshape(trajectories[k], trajectories[k].shape[1:])
        return trajectories


#######################################
### plot shortcuts
#######################################

import matplotlib.pyplot as plt
import matplotlib.colors as colors

def plot_trajectory(data,title='trajectory'):
    plt.imshow(data, aspect='auto', interpolation='none',origin='lower',cmap='Reds')
    plt.title(title)
    plt.xlabel(r'time $n_k$')
    plt.ylabel(r'electron momentum $n_t$')
    cb = plt.colorbar()
    cb.set_label('occupation')
    plt.show()

    
def plot_difference(test,truth,title='difference'):
    
    err = test-truth
    
    plt.imshow(err, aspect='auto', interpolation='none',origin='lower',cmap='bwr', norm=colors.CenteredNorm(vcenter=0.0))
    plt.title(title)
    plt.xlabel(r'time $n_k$')
    plt.ylabel(r'electron momentum $n_t$')
    cb = plt.colorbar()
    cb.set_label('error')
    plt.show()


#######################################
### class to benchmark the dimensionality reduction with the KFold function
#######################################

class reducer_helper_class:
    
    def __init__(self, trajectories = None, dim_reducer = None, rdim = 1):
        self.trajectories = trajectories
        if trajectories != None:
            self.n_trajectories = len(trajectories)
        self.rdim = rdim
        self.dim_reducer = dim_reducer
    
    def load_trajectories(self, trajectories):
        self.trajectories = trajectories
        self.n_trajectories = len(trajectories)
    
    def train(self, rdim=None, dim_reducer = None):
        
        if self.trajectories == None:
            raise ValueError('no trajectories loaded')
        
        if dim_reducer != None:
            self.dim_reducer = dim_reducer
            
        if self.dim_reducer == None:
            raise ValueError('no dim reducer object as been passed')
            
        if rdim != None:
            self.rdim = rdim

        data_matrix = np.concatenate(self.trajectories,axis=0)
        
        self.dim_reducer.train(data_matrix)
        
    def approx_single_run(self, run, rdim=None):
        if rdim == None:
            rdim = self.rdim
               
        return self.dim_reducer.reconstruct( self.dim_reducer.reduce(run, rdim) )
    
    def get_error(self, run, approx=np.zeros(1), rdim=None, norm='NF'):
        
        if rdim == None:
            rdim = self.rdim
        
        if approx.size == 1:
            approx = self.approx_single_run(run, rdim=rdim)
        
        err=-1.
        if norm=='fro':
            err = np.linalg.norm(run-approx, ord='fro')       
        elif norm =='max':
            err = np.abs(run-approx).max()
        elif norm == 'std':
            err = np.std(np.ravel(run-approx))
        elif norm == 'NF':
            err = np.sqrt( np.mean( np.square(run-approx) ) )
        else:
            print('unknown norm') 

        return err
    
    def score_multiple_trajectories(self,trajectories,**kwargs):
        scores = []
        for k in range(len(trajectories)):
            scores.append(self.get_error(trajectories[k], **kwargs))
        
        mean = np.mean(scores)
        return mean, scores
