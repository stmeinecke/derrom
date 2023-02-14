import numpy as np



#######################################
### KFold cross validation
#######################################

def get_KFolds(data_list, folds=5):
    KFolds = np.array_split(data_list, folds) #numpy returns list of ndarrays
    KFolds = [list(array) for array in KFolds] #covert ndarrays back to list such that KFolds is a list of lists of ndarrays (the individual trajectories)
    return KFolds
  
  
def get_KFold_CV_scores(model, trajectories, targets='AR', folds=5, seed=817, norms = ['rms'], train_kwargs={}):
    
    if targets != 'AR':
        assert len(trajectories) == len(targets)
    
    #create shuffled copy of the trajectories
    rng = np.random.default_rng(seed=seed)
    shuffled_inds = [i for i in range(len(trajectories))]
    rng.shuffle(shuffled_inds)
    
    strajectories = [trajectories[ind] for ind in shuffled_inds]
    if targets != 'AR':
        stargets = [targets[ind] for ind in shuffled_inds]
    
    
    #split trajectories into folds
    KFold_trajectories = get_KFolds(strajectories, folds=folds)
    if targets != 'AR':
        KFold_targets = get_KFolds(stargets, folds=folds)
    
    scores = [[] for n in range(len(norms))]  #of the individual folds - for each error norm in the norms list
    
    for k in range(folds):
        train_trajectories = KFold_trajectories.copy()
        test_trajectories = train_trajectories.pop(k) #test_trajectories = the kth fold, train_trajectories to remaining folds
        train_trajectories = [item for sublist in train_trajectories for item in sublist] #unpack the training folds into one flattened list
        
        if targets != 'AR':
            train_targets = KFold_targets.copy()
            test_targets = train_targets.pop(k)
            train_targets = [item for sublist in train_targets for item in sublist]
        else:
            test_targets = None
        
        #train the model on the training trajectories and get scores from the testing trajectories
        
        if targets == 'AR':
            model.fit(trajectories=train_trajectories, targets='AR', **train_kwargs)
        else:
            model.fit(trajectories=train_trajectories, targets=train_targets, **train_kwargs)
        
        
        predictions=[]
        for trajectory in test_trajectories:
            predictions.append(model.predict(trajectory))
        
        #score the test trajectories for each error norm in the norms list
        for l,norm in enumerate(norms): 
            fold_mean_score, fold_all_scores = model.score_multiple_trajectories(test_trajectories, test_targets, predictions=predictions, norm=norm)
            scores[l].append(fold_all_scores)
            
    for n in range(len(norms)):
        scores[n] = [item for sublist in scores[n] for item in sublist]
    return scores


#######################################
### save and load models
#######################################

import pickle 

def save_model(model,filename='../model.obj'):
    file = open(filename, 'wb') 
    pickle.dump(model, file)

def load_model(filename='../model.obj'):
    file = open(filename, 'rb') 
    return pickle.load(file)

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
    plt.ylabel(r'time $t_n$')
    plt.xlabel(r'electron momentum $k_n$')
    cb = plt.colorbar()
    cb.set_label('occupation')
    plt.show()

    
def plot_difference(test,truth,title='difference'):
    
    err = test-truth
    
    plt.imshow(err, aspect='auto', interpolation='none',origin='lower',cmap='bwr', norm=colors.CenteredNorm(vcenter=0.0))
    plt.title(title)
    plt.ylabel(r'time $t_n$')
    plt.xlabel(r'electron momentum $t_n$')
    cb = plt.colorbar()
    cb.set_label('error')
    plt.show()


#######################################
### class to benchmark the dimensionality reduction with the KFold function
#######################################

class reducer_helper_class:
    
    def __init__(self, dim_reducer = None, rdim = 1):

        self.rdim = rdim
        self.dim_reducer = dim_reducer
        
        self.reg_mode = 'AR' #fake AR to make the KFold CV scores work
    
    def fit(self, trajectories, targets='AR', rdim=None, dim_reducer = None):
        
        if dim_reducer != None:
            self.dim_reducer = dim_reducer
            
        if self.dim_reducer == None:
            raise ValueError('no dim reducer object as been passed')
            
        if rdim != None:
            self.rdim = rdim

        data_matrix = np.concatenate(trajectories,axis=0)
        
        self.dim_reducer.train(data_matrix, self.rdim)
        
    def predict(self, run, rdim=None):
        if rdim == None:
            rdim = self.rdim
               
        return self.dim_reducer.reconstruct( self.dim_reducer.reduce(run, rdim) )
    
    def get_error(self, trajectory, pred=None, rdim=None, norm='rms'):
        
        if rdim == None:
            rdim = self.rdim
        
        if pred is None:
            pred = self.predict(trajectory, rdim=rdim)
        
        err=-1.
        if norm=='fro':
            err = np.linalg.norm(trajectory-pred, ord='fro')       
        elif norm =='max':
            err = np.abs(trajectory-pred).max()
        elif norm == 'rms':
            err = np.sqrt( np.mean( np.square(trajectory-pred) ) )
        else:
            print('unknown norm') 

        return err
    
    
    def score_multiple_trajectories(self,trajectories, targets=None, predictions=None, **kwargs):
        scores = []
        
        if predictions is None:
            for k in range(len(trajectories)):
                scores.append(self.get_error(trajectory=trajectories[k], **kwargs))

        else:
            for k in range(len(trajectories)):
                scores.append(self.get_error(trajectory=trajectories[k], pred=predictions[k], **kwargs))
        
        mean = np.mean(scores)
        return mean, scores
