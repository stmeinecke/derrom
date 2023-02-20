import numpy as np

from sklearn.base import BaseEstimator

import derrom.optimizers as optimizers


class derrom_estimator(BaseEstimator):
    """
    Delay Embedded Regressive Reduced Order Model (derrom)
    
    This is the estimator class with facilitates the training of the model, and the prediction and scoring of trajectories. It is designed to be modular, such that different approaches for the dimensionality reduction, feature scaling, nonlinear transformation and regression weight optimization can be passed via objects that implement the appropriate methods. 
    
    Parameters
    ----------
    rdim : int
        number of reduced latent space dimensions. The default rdim = 1 is most likely an inappropriate choice...
    DE_l : int
        delay embedding length. The default DE_l = 1 corresponds to no embedding, i.e., only the most recent system state is used.
    full_hist : bool
        If set to False (default), the delay embedding is padded with the least recent state if no sufficient system history is available
    intercept : bool
        If set to True, a bias/intercept term is added to the regression step.
    dim_reducer : dim_reducer object
        Object that implements the train, reduce, and reconstruct methods. If set to None, no dimensionality reduction is performed
    scaler : scaler object
        Object that implements the train, transform, and inverse transform methods. If set to None, no feature scaling is performed
    NL_transformer : transformer object
        Object that implements the setup and transform methods. If set to None, no nonlinear transform of the delay-embedded system state is performed.
    optimizer : optimizer object
        Object that implements the solve method. In set to None, a least-squares method is applied.
    
    
    Class attributes: (attributes, which are generated during the initialization are omitted.)
    
    Attributes
    ----------
    w : numpy.ndarray
        Regression weights obtained from the fit method contained in a matrix (2D numpy.ndarray)
    reg_mode : str 
        Regression mode. Can be set to 'AR' (autoregression) by the fit method. Toggles the predict method to forecast in autonomous mode, i.e., to feed its own output back as an input.
    
    """
  
    def __init__(self, rdim = 1, DE_l = 1, full_hist=False, intercept = False, dim_reducer = None, scaler = None, NL_transformer = None, optimizer = None):
        
        
        self.w = None
        self.reg_mode = None
        self.n_target_vars = None
        
        self.rdim = rdim
        
        self.DE_l = DE_l
        
        self.intercept = intercept
        self.full_hist = full_hist
        
        if dim_reducer != None:
            self.dim_reducer = dim_reducer
            self.reduce_dim = True
        else:
            self.reduce_dim = False
        
        if scaler != None:
            self.scaler = scaler
            self.standardize = True
        else:
            self.standardize = False
            
        if NL_transformer != None:
            self.NL_transformer = NL_transformer
            self.NL_transform = True
        else:
            self.NL_transform = False
            
        if optimizer != None:
            self.optimizer = optimizer
        else:
            self.optimizer = optimizers.lstsqrs()
 
    
    
    def __compare_trajectories_targets(self,trajectories,targets):
        n_trajectory_data_vectors = np.sum([trajectory.shape[0] for trajectory in trajectories])
        n_target_data_vectors = np.sum([target.shape[0] for target in targets])
        
        if (n_trajectory_data_vectors == n_target_data_vectors):
            return True
        else:
            return False
    
    
    def __build_DE_vec(self, matrix, row, DE_l):
        if DE_l == 1:
            return matrix[row]
        else:
            DE_vec = []
            for k in range(DE_l):
                if row+k < 0:
                    DE_vec.append(matrix[0])
                else:
                    DE_vec.append(matrix[row+k])
            return np.concatenate(DE_vec, axis=0)
    
    
    def __build_DE_matrix(self, reduced_trajectories):
        
        assert self.DE_l > 0
        
        if self.DE_l == 1:
            return np.concatenate(reduced_trajectories,axis=0)
        
        else:
            DE_matrix = []

            for r in range(len(reduced_trajectories)):

                if(self.full_hist == False):
                    nRows = reduced_trajectories[r].shape[0]
                    Delta_j = self.DE_l-1
                else:
                    nRows = reduced_trajectories[r].shape[0]-(self.DE_l-1)
                    Delta_j = 0
                    
                nCols = self.rdim*self.DE_l

                run_DE_matrix = np.zeros((nRows,nCols))
                for j in range(nRows):
                    run_DE_matrix[j] = self. __build_DE_vec(reduced_trajectories[r][:,:self.rdim], j-Delta_j, self.DE_l)

                DE_matrix.append(run_DE_matrix)

            DE_matrix = np.concatenate(DE_matrix, axis=0)

            return DE_matrix
    
    
    def __build_target_matrix(self, targets):
        
        if self.full_hist == False:
            target_matrix = np.concatenate(targets, axis=0)
        else:
            assert self.DE_l > 0
            
            target_matrix = []
            
            for r in range(len(targets)):
                target_matrix.append(targets[r][self.DE_l-1:])
                
            target_matrix = np.concatenate(target_matrix, axis=0)
        
        return target_matrix

  
    def fit(self, trajectories, targets='AR', rdim = None, DE_l = None, intercept=None, full_hist=None, dim_reducer = None, scaler = None, NL_transformer = None, optimizer = None):
        """
        fit a derrom model
        
        Parameters
        ----------
        trajectories : list
            A list of the training trajectories, where each element of the list is expected to be a 2D numpy.ndarray that represents an individual trajectories. Time slices must be stored in the rows (first index) and the system state variables in the columns (second index). All trajectories must have the same number of variables (columns), the number of time slices, however, may vary.
        targets : 'AR', list
            A list of the training targets, where each element is an numpy.ndarray that corresponds to training trajectory with the identical list index. Each element must have the same number of rows as the corresponding trajectory. If set to 'AR', i.e., autoregression, the targets are automatically generated from the time-shifted trajectories and the last and first time slices are dropped from the training and target data, respectively.


        The remaining parameters are identical to the init method
        
        """

        n_trajectories = len(trajectories)
        
        if targets == 'AR':
            self.reg_mode = 'AR'
        
        else:
            self.reg_mode = 'reg'
            n_targets = len(targets)
            self.n_target_vars = targets[0][0].size
        
            ### check data consistency
            assert self.__compare_trajectories_targets(trajectories,targets)
        
        
        if rdim != None:
            self.rdim = rdim
        
        if DE_l != None:
            self.DE_l = DE_l
        if intercept != None:
            self.intercept = intercept
        if full_hist != None:
            self.full_hist = full_hist
        
        if dim_reducer != None:
            self.dim_reducer = dim_reducer
            self.reduce_dim = True
        
        if scaler != None:
            self.scaler = scaler
            self.standardize = True
            
        if NL_transformer != None:
            self.NL_transformer = NL_transformer
            self.NL_transform = True
            
        if optimizer != None:
            self.optimizer = optimizer
        
        #apply the dimensionality reduction to get the reduced coefficient matrix with rdim features via the dim_reducer object
        if self.reduce_dim == True:
            self.dim_reducer.train(np.concatenate(trajectories,axis=0),self.rdim)
            reduced_trajectories = [self.dim_reducer.reduce(trajectory,self.rdim) for trajectory in trajectories]
        else:
            reduced_trajectories = trajectories
            self.rdim = trajectories[0].shape[1]

        #apply data/feature scaling via scaler object
        if self.standardize:
            self.scaler.train(np.concatenate(reduced_trajectories,axis=0))
            reduced_trajectories = [self.scaler.transform(reduced_trajectory) for reduced_trajectory in reduced_trajectories]
           
        #create training data matrices
        if self.reg_mode == 'reg':
            training_matrix = self.__build_DE_matrix(reduced_trajectories)    
            target_matrix = self.__build_target_matrix(targets)
        else:
            training_matrix = self.__build_DE_matrix( [reduced_trajectory[:-1] for reduced_trajectory in reduced_trajectories] ) 
            target_matrix = self.__build_target_matrix( [reduced_trajectory[1:] for reduced_trajectory in reduced_trajectories] )
        
        
        #apply transformation to the DE state
        if self.NL_transform == True:
            self.NL_transformer.setup(training_matrix.shape[1])
            training_matrix = self.NL_transformer.transform(training_matrix)

        #add bias/intercept
        if self.intercept:
            training_matrix = np.concatenate( [ training_matrix, np.ones( (training_matrix.shape[0],1) ) ] , axis=1 )

        #calculate weight matrix via optimizer object
        self.w = self.optimizer.solve(training_matrix, target_matrix)
    
    
    
    def predict(self, trajectory):
        """
        Predicts targets for each snapshot of the input trajectory. In autoregression mode, the first time slice is used (the first DE_l in the case of full_hist = True) as initial conditions to generate a prediction with the same length (rows) as the input trajectory.
        
        
        Parameters
        ----------
        trajectory : 2D numpy.ndarray
            Trajectory where the time slices are stored in the rows (first index) and the state variables in the columns (second indx)
            
            
        Returns
        -------
            matrix : (2D numpy.ndarray)
                Calculated prediction
        """
        
        
        if self.reg_mode == 'AR':
            return self.forecast(trajectory,trajectory.shape[0])
        else:
            if trajectory.ndim == 1:
                trajectory = trajectory.reshape(1,-1)
            
            #apply the dimensionality reduction to the trajectory
            if self.reduce_dim == True:
                reduced_trajectory = self.dim_reducer.reduce(trajectory,self.rdim)
            else:
                reduced_trajectory = trajectory
            
            #apply data/feature scaling
            if self.standardize:
                reduced_trajectory = self.scaler.transform(reduced_trajectory)

            #setup numpy array for the prediction
            if self.full_hist == True:
                pred_length = trajectory.shape[0]-(self.DE_l-1)
            if self.full_hist == False:
                pred_length = trajectory.shape[0]
            
            pred = np.zeros((pred_length,self.n_target_vars))
            
            if self.NL_transform == True:
                feature_matrix = self.NL_transformer.transform(self.__build_DE_matrix([reduced_trajectory]))
            else:
                feature_matrix = self.__build_DE_matrix([reduced_trajectory])
            
            #add bias/intercept 
            if self.intercept:
                feature_matrix = np.concatenate( [ feature_matrix, np.ones( (feature_matrix.shape[0],1) ) ] , axis=1 )
      
            #let the machine predict the dynamics
            for j in range(0, pred.shape[0]):
                #predict the next step
                pred[j] = feature_matrix[j] @ self.w

            return pred
    
    
    def forecast(self,init,n_steps):
        """
        Forecast n steps into the future in autoregression mode
        
        Parameters
        ----------
        
        init : 2D numpy.ndarray
            Initial condition. For full_hist = False, only the first snapshot (first row) of is used. Otherwise the first DE_l snapshots are used. 
        n_steps : int
            Length of the forecasted trajectory. The initial condition is thus included in n_steps
            
            
        Returns
        -------
            matrix : (2D numpy.ndarray)
                Forecasted trajectory
        """
        
        
        assert self.reg_mode == 'AR'
        
        #apply the dimensionality reduction to the initital conditions
        if self.reduce_dim == True:
            reduced_init = self.dim_reducer.reduce(init,self.rdim)
        else:
            reduced_init = init
        
        #apply data/feature scaling
        if self.standardize:
            reduced_init = self.scaler.transform(reduced_init)

        #setup numpy array for the auto prediction
        pred = np.zeros((n_steps,reduced_init.shape[1]))

        #build initial condition for the auto predictions
        if (self.full_hist == False):
            j_start = 1
            pred[0] = reduced_init[0]
        else:
            j_start = self.DE_l
            pred[:self.DE_l] = reduced_init[:self.DE_l]
        
        #let the machine predict the dynamics
        for j in range(j_start,pred.shape[0]):
        
            #build the DE vector from the past steps
            DE_vec = self.__build_DE_vec(pred[:,:self.rdim], j-self.DE_l, self.DE_l)
            DE_vec = DE_vec.reshape((1,DE_vec.size))
            
            #apply transformation to the DE state
            if self.NL_transform == True:
                transform = self.NL_transformer.transform(DE_vec)

            #add intercept/bias
            if self.intercept:
                transform = np.append(transform, 1.0)
                          
            #predict the next step
            pred[j] = transform @ self.w

        #undo the data/feature scaling
        if self.standardize:
            pred = self.scaler.inverse_transform(pred)
        
        #reconstruct the full from the reduced representation
        if self.reduce_dim == True:
            pred = self.dim_reducer.reconstruct(pred)

        return pred
        
        
    
    def get_error(self, trajectory=None, truth=None, pred=None, norm='rms'):
        """
        Computes the regression error
        
        Parameters
        ----------
        trajectory : 2D numpy.ndarray
            If no prediction is supplied, it can be computed from the trajectory. Interchangeable with truth in autoregression (AR) mode
        truth : 2D numpy.ndarray
            Ground truth, against which the prediction is compared. Interchangeable with trajectory in autoregression (AR) mode
        pred : 2D numpy.ndarray
            Prediction corresponding to the truth. 
        
        """
        
        if self.reg_mode == 'AR':
            if truth is None and trajectory is None:
                raise ValueError('no trajectory supplied')
            elif truth is None:
                truth = trajectory
            elif trajectory is None:
                trajectory = truth
                
        else:
            if truth is None:
                raise ValueError('no truth supplied')
            
        if pred is None:
            if trajectory is not None:
                pred = self.predict(trajectory)
            else:
                raise ValueError('no trajectory supplied to compute prediction')
        
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
                
                
    def score_multiple_trajectories(self,trajectories, targets=None, predictions=None, **kwargs):
        scores = []
        
        if predictions is None:
            for k in range(len(trajectories)):
                if targets is None or self.reg_mode=='AR':
                    scores.append(self.get_error(trajectory=trajectories[k], **kwargs))
                else:
                    scores.append(self.get_error(trajectory=trajectories[k], truth=targets[k], **kwargs))
        else:
            for k in range(len(trajectories)):
                if targets is None or self.reg_mode=='AR':
                    scores.append(self.get_error(trajectory=trajectories[k], pred=predictions[k], **kwargs))
                else:
                    scores.append(self.get_error(trajectory=trajectories[k], truth=targets[k], pred=predictions[k], **kwargs))
        
        mean = np.mean(scores)
        return mean, scores
    
    
                
    def print_status(self):
        """
        print values of the relevant class attributes
        """
        
        assert self.w is not None
        
        print('full_hist: ', self.full_hist)
        print('intercept: ', self.intercept)
        print('standardize: ', self.standardize)
        print('rdim: ', self.rdim)
        print('DE_l: ', self.DE_l)
        print('weights shape: ', self.w.shape)
        
        
