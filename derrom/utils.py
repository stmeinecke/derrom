import numpy as np



#######################################
### KFold cross validation
#######################################

def get_KFolds(data_list, folds=5):
    """
    Split the given list of trajectories into K-folds for cross-validation.

    Parameters
    ----------
    data_list : list
        An list of trajectories, where each trajectory is a 2D numyp.ndarray.
    folds : int, optional
        The number of folds to create. Defaults to 5.

    Returns
    -------
    folds : list
        A list of K folds, where each fold is a list of trajectories.
    """
    
    KFolds = np.array_split(data_list, folds) #numpy returns list of ndarrays
    KFolds = [list(array) for array in KFolds] #covert ndarrays back to list such that KFolds is a list of lists of ndarrays (the individual trajectories)
    return KFolds
  
  
def get_KFold_CV_scores(model, trajectories, targets='AR', folds=5, seed=817, norms = ['rms'], train_kwargs={}):
    """
    Conduct K-fold cross-validation on the given model using the given trajectories and targets, and return the scores for each fold.

    Parameters
    ----------
    model : object
        An object that implements the `fit()` and `score_multiple_trajectories()` methods, which will be used to train and score the model.
    trajectories : list
        An list of trajectories, where each trajectory is a 2D numpy.ndarray.
    targets : list or str, optional
        An list of targets corresponding to the given trajectories, or 'AR' for autoregression mode. Defaults to 'AR'.
    folds : int, optional
        The number of folds to create for cross-validation. Defaults to 5.
    seed : int, optional
        The seed value to use for the random number generator when shuffling the trajectories. Defaults to 817.
    norms : list of str, optional
        A list of error norms to use for scoring the trajectories. Defaults to ['rms'].
    train_kwargs : dict, optional
        A dictionary of keyword arguments to pass to the `fit()` method when training the model. Defaults to {}.

    Returns
    -------
    scores : list
        A list of n lists, where n is the number of error norms specified in `norms`. Each list contains the scores for each each error norm.
    """
    
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
    """
    Save a model object to a file using the pickle module.

    Parameters
    ----------
    model : object
        The model to be saved to a file.
    filename : str, optional
        The name of the file to save the object to. Defaults to '../model.obj'.
    """
    file = open(filename, 'wb') 
    pickle.dump(model, file)

    
def load_model(filename='../model.obj'):
    """
    Load a Python object from a file using the pickle module.

    Parameters
    ----------
    filename : str, optional
        The name of the file to load the object from. Defaults to '../model.obj'.

    Returns
    -------
    model : object
        The loaded model object.
    """
    file = open(filename, 'rb') 
    return pickle.load(file)



#######################################
### save and load trajectories
#######################################

def save_trajectories(trajectories, filename='../trajectories'):
    """
    Save the given list of trajectories to a numpy .npz file.

    Parameters
    ----------
    trajectories : list
        list of trajectories to save.
    filename : str, optional
        Name of the file to save the trajectories to. Defaults to '../trajectories'.
    """
    np.savez(filename, trajectories)
    
def load_trajectories(filename='../trajectories.npz'):
    """
    Load a list of trajectories from a numpy .npz file.

    Parameters
    ----------
    filename : str, optional
        Name of the file to load the trajectories from. Defaults to '../trajectories.npz'.

    Returns
    -------
    trajectories : list
        List of loaded trajectories if file exists, otherwise None.
    """
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
    """
    Plot a 2D trajectory using matplotlib.

    Parameters
    ----------
    data : 2D numpy.ndarray
        A 2D array of the trajectory to be plotted. Each row represents a time step and each column represents a state variable.
    title : str, optional
        The title of the plot. Defaults to 'trajectory'.
    """
    plt.imshow(data, aspect='auto', interpolation='none',origin='lower',cmap='Reds')
    plt.title(title)
    plt.ylabel(r'time $t$')
    plt.xlabel(r'state variable $s_n$')
    cb = plt.colorbar()
    cb.set_label('value')
    plt.show()

    
def plot_difference(test,truth,title='difference'):
    """
    Plot the element-wise difference between two 2D arrays using matplotlib.

    Parameters
    ----------
    test : numpy.ndarray
        The first 2D array to be compared.
    truth : numpy.ndarray
        The second 2D array to be compared.
    title : str, optional
        The title of the plot. Defaults to 'difference'.
    """
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
    """
    A helper class for benchmarking dimensionality reducers. Implements the `fit()` and `score_multiple()` methods to work with the `get_KFold_CV_scores()` function.

    Parameters
    ----------
    dim_reducer : object or None, optional
        The dimensionality reduction object
    rdim : int, optional
        The reduced dimensionality, i.e., the number of components to retain after dimensionality reduction. Default is 1.

    Attributes
    ----------
    rdim : int
        The reduced dimensionality, i.e., the number of components to retain after dimensionality reduction.
    dim_reducer : object
        The dimensionality reduction object.
    reg_mode : str
        This string is set to 'AR' to properly work the the KFold_CV function
    """
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
    """
    Quick and dirty numerical integrator for solving initial value problems (IVPs), where either all or only parts of the right-hand-side of the equations of motion are provided by a data-driven model. Implements the `fit()` and `score_multiple()` methods to work with the `get_KFold_CV_scores()` function.

    Parameters
    ----------
    model : object
        Model object with the methods 'predict' and 'fit'. 'predict' provides either all or parts of the derivatives. The 'fit' method can be invoked by the integrator during KFold_CV scoring.
    derivs : function, optional
        Function that calculates the system derivatives with
        respect to time. Default is None, in which case the 'predict' method
        of the model object is used.
    dt : float, optional
        Time step for numerical integration. Default is 1.
    dt_out : float, optional
        Time step for saving output data. Default is 1.
    method : {'Heun', 'Euler'}, optional
        Numerical integration method to use. Default is 'Heun'.
        
        
    Attributes
    ----------
    model_hist_option : bool
        Boolean indicating whether the model's 'full_hist' attribute is True or
        False.

    """
    
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
        """
        fits the data driven model to the training data presented via the `**kwargs`. Used by the KFold_CV function
        """
        
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
        """
        Integrate the differential equations using the specified method and time step.

        Parameters
        ----------
        init : 2D numpy.ndarray
            Initial state of the system.
        n_steps : int
            Number of integration steps to perform.
        dt : float, optional
            Time step for the integration. Defaults to `self.dt` if not specified.
        dt_out : float, optional
            Time step for outputting results. Defaults to `self.dt_out` if not specified.

        Returns
        -------
        trajectory : 2D numpy.ndarray
            Array containing the state of the system at each time step.
        """
        
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
        """
        Use the first time steps of the given trajectory as initial conditions and integrate the system to obtain a solution with identical length. Intended to score the solutions against the input trajectories.
        
        Parameters
        ----------
        trajectory : 2D numpy.ndarray
            Test trajectory, i.e., ground truth

        Returns
        -------
        prediction : 2D numpy.ndarray
            An array containing the predicted trajectory of the system.
        """
        return self.integrate(trajectory, trajectory.shape[0])
    
    
    def get_error(self, truth, pred=None, norm='rms'):
        """
        Calculates the error between the ground truth values and predicted values.

        Parameters
        ----------
        truth: 2D numpy.ndarray
            The ground truth values.
        pred: 2D numpy.ndarray, optional
            The predicted values. If not provided, the function uses the predict method of the class to make predictions.
        norm: str, optional
            The type of norm to be used to calculate the error. Default is 'rms'.
            Possible values are:
            - 'rms': Normalized Frobenius norm
            - 'fro': Frobenius norm
            - 'max': Absolute maximum norm

        Returns
        -------
        error : float
            The calculated error value.
        """
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
        """
        helper function to obtain error scores for multiple trajectories. Used by the derrom.utils.get.get_KFold_CV_scores function.
        
        Parameters
        ----------
        trajectories : list
            A list of the trajectories to be scored, where each element of the list is expected to be a 2D numpy.ndarray that represents an individual trajectories. Time slices must be stored in the rows (first index) and the system state variables in the columns (second index). All trajectories must have the same number of variables (columns), the number of time slices, however, may vary.
        targets : list
            A list of the targets, where each element is an numpy.ndarray that corresponds to the trajectory with the identical list index. Each element must have the same number of rows as the corresponding trajectory. If set to 'AR', i.e., autoregression, no targets are required.
        predictions : list
            A list of the predictions, where each element is an numpy.ndarray that corresponds to the trajectory with the identical list index. Supplying predictions reduces computaion time if derrom.utils.get.get_KFold_CV_scores is to be evaluated for multiple error norms. 
            
            
        Returns
        -------
        mean : float
            mean regression error from all supplied trajectories
        scores : list
            list of all individual regression errors
            
            
        """
        
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

