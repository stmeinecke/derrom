import numpy as np



#######################################
### KFold cross validation
#######################################

def get_KFold_runs(runs, folds=5):
    KFold_runs = np.array_split(runs, folds) #numpy returns list of ndarrays
    KFold_runs = [list(array) for array in KFold_runs] #covert ndarrays back to list such that KFold_runs is a list of lists of ndarrays (the individual runs)
    return KFold_runs
  
  
def get_KFold_CV_scores(model, runs, folds=5, seed=817, norms = ['std'], train_kwargs={}):
    
    #create shuffled copy of the runs
    rng = np.random.default_rng(seed=seed)
    sruns = runs.copy()
    rng.shuffle(sruns)
    
    #split runs into folds
    KFold_runs = get_KFold_runs(sruns, folds=folds)
    
    scores = [[] for n in range(len(norms))]  #of the individual folds - for each error norm in the norms list
    
    for k in range(folds):
        train_runs = KFold_runs.copy()
        test_runs = train_runs.pop(k) #test_runs = the kth fold, train_runs to remaining folds

        train_runs = [item for sublist in train_runs for item in sublist] #unpack the training folds into one flattened list

        #train the model on the training runs and get scores from the testing runs
        model.load_trajectories(train_runs) 
        model.train(**train_kwargs)
        
        #score the test runs for each error norm in the norms list
        for l,norm in enumerate(norms): 
            fold_mean_score, fold_all_scores = model.score_multiple_trajectories(test_runs, norm=norm)
            scores[l].append(fold_all_scores)
            
    for n in range(len(norms)):
        scores[n] = [item for sublist in scores[n] for item in sublist] #unpack the training folds into one flattened list
    
    return scores


#######################################
### save and load runs
#######################################

def save_runs(runs, filename='../runs'):
    np.savez(filename, runs)
    
def load_runs(filename='../runs.npz'):
    
    from os.path import exists

    if not exists(filename):
        print('runs file ot found')
        return None
    else:
        npz_runs = np.load(filename)
        runs = np.split(npz_runs['arr_0'], npz_runs['arr_0'].shape[0], axis=0)

        for k in range(len(runs)):
            runs[k] = np.reshape(runs[k], runs[k].shape[1:])
        return runs


#######################################
### plot shortcuts
#######################################

import matplotlib.pyplot as plt

def plot_trajectory(data,title='trajectory'):
    plt.imshow(data, aspect='auto', interpolation='none',origin='lower',cmap='Reds')
    plt.title(title)
    plt.xlabel(r'time $n_k$')
    plt.ylabel(r'electron momentum $n_t$')
    cb = plt.colorbar()
    cb.set_label('occupation')
    plt.show()
    

import matplotlib.colors as colors
class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        self.vcenter = vcenter
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=False):
        """
        Map value to the interval [0, 1]. The clip argument is unused.
        """
        result, is_scalar = self.process_value(value)
        self.autoscale_None(result)  # sets self.vmin, self.vmax if None

        if not self.vmin <= self.vcenter <= self.vmax:
            raise ValueError("vmin, vcenter, vmax must increase monotonically")
        result = np.ma.masked_array(
            np.interp(result, [self.vmin, self.vcenter, self.vmax],
                      [0, 0.5, 1.]), mask=np.ma.getmask(result))
        if is_scalar:
            result = np.atleast_1d(result)[0]
        return result
    
def plot_difference(truth,test,title='difference'):
    
    err = test-truth

    midnorm = MidpointNormalize(vmin=err.min(), vcenter=0.0, vmax=err.max())
    
    plt.imshow(err, aspect='auto', interpolation='none',origin='lower',cmap='bwr')
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
            
        if rdim != None:
            self.rdim = rdim

        data_matrix = np.concatenate(self.runs,axis=0)
        
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
    
    def score_multiple_runs(self,runs,**kwargs):
        scores = []
        for k in range(len(runs)):
            scores.append(self.get_error(runs[k], **kwargs))
        
        mean = np.mean(scores)
        return mean, scores
