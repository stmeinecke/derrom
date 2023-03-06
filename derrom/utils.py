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
    plt.ylabel(r'time $t$')
    plt.xlabel(r'state variable $s_n$')
    cb = plt.colorbar()
    cb.set_label('value')
    plt.show()

    
def plot_difference(test,truth,title='difference'):
    
    err = test-truth
    
    plt.imshow(err, aspect='auto', interpolation='none',origin='lower',cmap='bwr', norm=colors.CenteredNorm(vcenter=0.0))
    plt.title(title)
    plt.ylabel(r'time $t$')
    plt.xlabel(r'state variable $s_n$')
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

    
#######################################
### ode integrator classes, which allow for the integration of derrom.
#######################################

class ivp_integrator:
    
    def __init__(self, model, derivs=None, dt=1., dt_out=1., method='Heun'):
        self.dt = dt
        self.dt_out = dt_out
        self.method = method
        
        self.targets = 'AR'
        
        self.model = model
        self.model_hist_option = model.full_hist
        self.model.full_hist = True
        
        if derivs is None:
            self.derivs = self.model.predict
        else:
            self.derivs = derivs
    
    
    def fit(self, **kwargs):
        
        self.model.full_hist = self.model_hist_option
        
        self.model.fit(**kwargs)
        
        self.model.full_hist = True
    
    
    def __Euler(self, init, n_steps, dt, dt_out):
        
        sol = np.zeros((n_steps,init.shape[1]))
        
        sol[:init.shape[0]] = init
        
        state = sol[:1]
        
        j_out = int(dt_out/dt)
        j_max = sol.shape[0]*j_out
        
        
        for j in range(1,sol.shape[0]*j_out):
            state = state + dt*self.derivs(state)
            
            if j%j_out == 0:
                sol[j//j_out] = state
                
        return sol


    def __Euler_wdelay(self, init, n_steps, dt, dt_out):
        
        sol = np.zeros((n_steps,init.shape[1]))
        
        sol[0] = init[0]
        
        state = sol[:1]
        
        j_out = int(dt_out/dt)
        j_max = sol.shape[0]*j_out
        
        hist_length = (self.model.DE_l-1)*j_out + 1
        hist_ind = hist_length - 1
        hist = np.zeros((hist_length,init.shape[1]))
        for k in range(hist.shape[0]):
            hist[k] = init[0]
        
        
        for j in range(1,sol.shape[0]*j_out):
            
            vecs = np.stack( [ hist[(hist_ind - n*j_out + hist_length)%hist_length] for n in range(self.model.DE_l-1,-1,-1)] )
            
            state = state + dt*self.derivs(vecs)
            
            hist_ind = (hist_ind+1)%hist_length
            
            hist[hist_ind] = state
            
            if j%j_out == 0:
                sol[j//j_out] = state
                
        return sol
    
    
    def __Heun(self, init, n_steps, dt, dt_out):
        
        sol = np.zeros((n_steps,init.shape[1]))
        
        sol[:init.shape[0]] = init
        
        state = sol[:1]
        
        j_out = int(dt_out/dt)
        j_max = sol.shape[0]*j_out
        
        
        for j in range(1,sol.shape[0]*j_out):
            
            f1 = self.derivs(state)
            f2 = self.derivs(state + dt*f1)
            
            state = state + 0.5*dt*(f1+f2)
            
            if j%j_out == 0:
                sol[j//j_out] = state
                
        return sol

    
    def __Heun_wdelay(self, init, n_steps, dt, dt_out):
        
        sol = np.zeros((n_steps,init.shape[1]))
        
        sol[0] = init[0]
        
        state = sol[:1]
        
        j_out = int(dt_out/dt)
        j_max = sol.shape[0]*j_out
        
        hist_length = (self.model.DE_l-1)*j_out + 1
        hist_ind = hist_length - 1
        hist = np.zeros((hist_length,init.shape[1]))
        for k in range(hist.shape[0]):
            hist[k] = init[0]
        
        
        for j in range(1,sol.shape[0]*j_out):
            
            vecs = np.stack( [hist[(hist_ind - n*j_out + hist_length)%hist_length] for n in range(self.model.DE_l-1,-1,-1)] )
            
            f1 = self.derivs(vecs).flatten()
            
            hist_ind = (hist_ind+1)%hist_length
            
            vecs = np.stack( [hist[(hist_ind - n*j_out + hist_length)%hist_length] for n in range(self.model.DE_l-1,0,-1)]+[(state+dt*f1).flatten()] )
            
            f2 = self.derivs(vecs)
            
            state = state + 0.5*dt*(f1+f2)

            hist[hist_ind] = state
            
            if j%j_out == 0:
                sol[j//j_out] = state
                
        return sol
    
    
    def integrate(self, init, n_steps, dt=None, dt_out=None):
        
        if dt is None:
            dt = self.dt
        if dt_out is None:
            dt_out = self.dt_out
        
        if self.method == 'Heun':
            if self.model.DE_l == 1:
                return self.__Heun(init,n_steps,dt,dt_out)
            else:
                return self.__Heun_wdelay(init,n_steps,dt,dt_out)
        elif self.method == 'Euler':
            if self.model.DE_l == 1:
                return self.__Euler(init,n_steps,dt,dt_out)
            else:
                return self.__Euler_wdelay(init,n_steps,dt,dt_out)
        else:
            raise ValueError('integration method >> ' + self.method + ' << does not exist')
    
    
    def predict(self, trajectory):
        return self.integrate(trajectory, trajectory.shape[0])
    
    
    def get_error(self, truth, pred=None, norm='rms'):
        
        if pred is None:
            pred = self.predict(truth)
        
        assert pred.shape == truth.shape
        
        err = -1.
        if norm =='rms': #normalized Frobenius norm
            err = np.sqrt( np.mean( np.square(truth-pred) ) )
        elif norm == 'fro': #Frobenius norm
            err = np.linalg.norm(truth-pred, ord='fro')
        elif norm =='max': #absolute max norm
            err = np.abs(truth-pred).max()
        else:
            print('unknown norm')
        
        return err
    
    
    def score_multiple_trajectories(self, trajectories, targets=None, predictions=None, **kwargs):
        scores = []
        
        if predictions is None:
            for k in range(len(trajectories)):
                scores.append(self.get_error(trajectories[k],**kwargs))
        else:
            assert len(trajectories) == len(predictions)
            for k in range(len(trajectories)):
                scores.append(self.get_error(trajectories[k], pred=predictions[k], **kwargs))
        
        mean = np.mean(scores)
        return mean, scores
    


class PHELPH_ivp_integrator(ivp_integrator):
    
    def fit(self, trajectories, targets, **kwargs):
        el_trajectories = [trajectory[:,:-1] for trajectory in trajectories]
        self.model.fit(el_trajectories, targets, **kwargs)
    
    def get_error(self, truth, pred=None, norm='rms'):
        
        if pred is None:
            pred = self.predict(truth)
        
        assert pred.shape == truth.shape
        
        I_truth = truth[:,-1]
        el_truth = truth[:,:-1]
        
        I_pred = pred[:,-1]
        el_pred = pred[:,:-1]
        
        
        err = -1.
        if norm =='rms':
            err = np.sqrt( np.mean( np.square(el_truth-el_pred) ) )
        elif norm =='max': #absolute max norm
            err = np.abs(el_truth-el_pred).max()
        elif norm == 'I_max':
            err = (I_pred.max()- I_truth.max())
        elif norm == 'I_max_pos':
            err = (np.argmax(I_pred)-np.argmax(I_truth))*self.dt_out
        elif norm == 'I_area':
            err = (np.sum(I_pred) - np.sum(I_truth))
        else:
            print('unknown norm')
        
        return err

