import numpy as np

import derrom_dim_reducers as dim_reducers
import derrom_optimizers as optimizers
import derrom_scalers as scalers
import derrom_transformers as transformers
import derrom_utils as utils


class derrom:
  
    def __init__(self, trajectories = None, targets = None, rdim = 1, VAR_l = 1, full_hist=False, intercept = False, dim_reducer = None, scaler = None, VAR_transformer = None, optimizer = None):
        
        if trajectories != None:
            self.trajectories = trajectories
            self.n_trajectories = len(trajectories)
            
        if targets != None:
            self.targets = targets
            if self.targets != 'AR':
                self.n_targets = len(targets)
                self.n_target_vars = targets[0][0].size
    
                ### check data consistency
                assert self.__compare_trajectories_targets()
        
        self.rdim = rdim
        
        self.VAR_l = VAR_l
        
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
            
        if VAR_transformer != None:
            self.VAR_transformer = VAR_transformer
            self.transform_VAR = True
        else:
            self.transform_VAR = False
            
        if optimizer != None:
            self.optimizer = optimizer
        else:
            self.optimizer = optimizers.lstsqrs()
        
        
    def load_data(self,trajectories,targets='AR'):
        self.trajectories = trajectories
        self.n_trajectories = len(trajectories)
        
        self.targets = targets
        if targets != 'AR':
            self.n_targets = len(targets)
            self.n_target_vars = targets[0][0].size
        
            ### check data consistency
            assert self.__compare_trajectories_targets()
        
    
    def load_trajectories(self,trajectories):
        self.trajectories = trajectories
        self.n_trajectories = len(trajectories)
    
    
    def __compare_trajectories_targets(self):
        n_trajectory_data_vectors = np.sum([trajectory.shape[0] for trajectory in self.trajectories])
        n_target_data_vectors = np.sum([target.shape[0] for target in self.targets])
        
        if ( (n_trajectory_data_vectors == n_target_data_vectors) and (self.n_trajectories == self.n_targets) ):
            return True
        else:
            return False
    
    
    def __build_VAR_vec(self, matrix, row, VAR_l):
        if VAR_l == 1:
            return matrix[row]
        else:
            VAR_vec = []
            for k in range(VAR_l):
                if row+k < 0:
                    VAR_vec.append(matrix[0])
                else:
                    VAR_vec.append(matrix[row+k])
            return np.concatenate(VAR_vec, axis=0)
    
    
    def __build_VAR_matrix(self, reduced_trajectories):
        
        assert self.VAR_l > 0
        
        if self.VAR_l == 1:
            return np.concatenate(reduced_trajectories,axis=0)
        
        else:
            VAR_matrix = []

            for r in range(len(reduced_trajectories)):

                if(self.full_hist == False):
                    nRows = reduced_trajectories[r].shape[0]
                    Delta_j = self.VAR_l-1
                else:
                    nRows = reduced_trajectories[r].shape[0]-(self.VAR_l-1)
                    Delta_j = 0
                    
                nCols = self.rdim*self.VAR_l

                run_VAR_matrix = np.zeros((nRows,nCols))
                for j in range(nRows):
                    run_VAR_matrix[j] = self. __build_VAR_vec(reduced_trajectories[r][:,:self.rdim], j-Delta_j, self.VAR_l)

                VAR_matrix.append(run_VAR_matrix)

            VAR_matrix = np.concatenate(VAR_matrix, axis=0)

            return VAR_matrix
    
    
    def __build_target_matrix(self, targets):
        
        if self.full_hist == False:
            target_matrix = np.concatenate(targets, axis=0)
        else:
            assert self.VAR_l > 0
            
            target_matrix = []
            
            for r in range(len(targets)):
                target_matrix.append(targets[r][self.VAR_l-1:])
                
            target_matrix = np.concatenate(target_matrix, axis=0)
        
        return target_matrix

  
    def train(self, rdim = None, VAR_l = None, intercept=None, full_hist=None, dim_reducer = None, scaler = None, VAR_transformer = None, optimizer = None):
        
        assert (self.trajectories != None and self.targets != None)
        
        if self.targets != 'AR':
            ### check data consistency
            assert self.__compare_trajectories_targets()
        
        if rdim != None:
            self.rdim = rdim
        
        if VAR_l != None:
            self.VAR_l = VAR_l
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
            
        if VAR_transformer != None:
            self.VAR_transformer = VAR_transformer
            self.transform_VAR = True
            
        if optimizer != None:
            self.optimizer = optimizer
        
        #apply the dimensionality reduction to get the reduced coefficient matrix with rdim features via the dim_reducer object
        if self.reduce_dim == True:
            self.dim_reducer.train(np.concatenate(self.trajectories,axis=0),self.rdim)
            self.reduced_trajectories = [self.dim_reducer.reduce(trajectory,self.rdim) for trajectory in self.trajectories]
        else:
            self.reduced_trajectories = self.trajectories
            self.rdim = self.trajectories[0].shape[1]

        #apply data/feature scaling via scaler object
        if self.standardize:
            self.scaler.train(np.concatenate(self.reduced_trajectories,axis=0))
            self.reduced_trajectories = [self.scaler.transform(reduced_trajectory) for reduced_trajectory in self.reduced_trajectories]
           
        #create training data matrices
        if self.targets != 'AR':
            self.training_matrix = self.__build_VAR_matrix(self.reduced_trajectories)    
            self.target_matrix = self.__build_target_matrix(self.targets)
        else:
            self.training_matrix = self.__build_VAR_matrix( [reduced_trajectory[:-1] for reduced_trajectory in self.reduced_trajectories] ) 
            self.target_matrix = self.__build_target_matrix( [reduced_trajectory[1:] for reduced_trajectory in self.reduced_trajectories] )
        
        
        #apply transformation to the VAR state
        if self.transform_VAR == True:
            self.VAR_transformer.setup(self.training_matrix.shape[1])
            self.training_matrix = self.VAR_transformer.transform(self.training_matrix)

        #add bias/intercept
        if self.intercept:
            self.training_matrix = np.concatenate( [ self.training_matrix, np.ones( (self.training_matrix.shape[0],1) ) ] , axis=1 )

        #calculate weight matrix via optimizer object
        self.w = self.optimizer.solve(self.training_matrix, self.target_matrix)
    
    
    
    def predict(self, trajectory):
        
        if self.targets == 'AR':
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
                pred_length = trajectory.shape[0]-(self.VAR_l-1)
            if self.full_hist == False:
                pred_length = trajectory.shape[0]
            
            pred = np.zeros((pred_length,self.n_target_vars))
            
            if self.transform_VAR == True:
                feature_matrix = self.VAR_transformer.transform(self.__build_VAR_matrix([reduced_trajectory]))
            else:
                feature_matrix = self.__build_VAR_matrix([reduced_trajectory])
            
            #add bias/intercept 
            if self.intercept:
                feature_matrix = np.concatenate( [ feature_matrix, np.ones( (feature_matrix.shape[0],1) ) ] , axis=1 )
      
            #let the machine predict the dynamics
            for j in range(0, pred.shape[0]):
                #predict the next step
                pred[j] = feature_matrix[j] @ self.w

            return pred
    
    
    def forecast(self,init,n_steps):
        
        assert self.targets == 'AR'
        
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
            j_start = self.VAR_l
            pred[:self.VAR_l] = reduced_init[:self.VAR_l]
        
        #let the machine predict the dynamics
        for j in range(j_start,pred.shape[0]):
        
            #build the VAR vector from the past steps
            VAR_vec = self.__build_VAR_vec(pred[:,:self.rdim], j-self.VAR_l, self.VAR_l)
            VAR_vec = VAR_vec.reshape((1,VAR_vec.size))
            
            #apply transformation to the VAR state
            if self.transform_VAR == True:
                transform = self.VAR_transformer.transform(VAR_vec)

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
        
        
    
    def get_error(self, trajectory=None, truth=None, pred=None, norm='NF'):
        
        if self.targets == 'AR':
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
        if norm =='NF': #normalized Frobenius norm
            err = np.sqrt( np.mean( np.square(truth-pred) ) )
        elif norm == 'fro': #Frobenius norm
            err = np.linalg.norm(truth-pred, ord='fro')
        elif norm =='max': #absolute max norm
            err = np.abs(truth-pred).max()
        else:
            print('unknown norm')
        
        return err
                
                
    def score_multiple_trajectories(self,trajectories, targets=None, **kwargs):
        scores = []
        for k in range(len(trajectories)):
            if targets is None or self.targets=='AR':
                scores.append(self.get_error(trajectory=trajectories[k], **kwargs))
            else:
                scores.append(self.get_error(trajectory=trajectories[k], truth=targets[k], **kwargs))
        
        mean = np.mean(scores)
        return mean, scores
    
                
    def print_status(self):
        print('full_hist: ', self.full_hist)
        print('intercept: ', self.intercept)
        print('standardize: ', self.standardize)
        print('rdim: ', self.rdim)
        print('VAR_l: ', self.VAR_l)
        print('train shape: ', self.training_matrix.shape)
        print('target shape: ', self.target_matrix.shape)
        print('weights shape: ', self.w.shape)
        
        
