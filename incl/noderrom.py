import numpy as np

import narrom_dim_reducer as dim_reducer
import narrom_optimizer as optimizer
import narrom_scaler as scaler
import narrom_transformer as transformer
import narrom_utils as utils


class noderrom:
  
    def __init__(self, trajectories = None, targets = None, rdim = 1, prdim=None, VAR_l = 1, full_hist=False, intercept = False, dim_reducer = None, scaler = None, VAR_transformer = None, optimizer = None):
        
        if trajectories != None:
            self.trajectories = trajectories
            self.n_trajectories = len(trajectories)
            
        if targets != None:
            self.targets = targets
            self.n_targets = len(targets)
            self.n_target_vars = targets[0][0].size
    
        ### check data consistency
        assert self.__compare_trajectories_targets()
        
        self.rdim = rdim
        if prdim == None:
            self.prdim = self.rdim
        else:
            self.prdim = prdim
        
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
            self.optimizer = optimizer.lstsqrs()
        
        
    def load_data(self,trajectories,targets):
        self.trajectories = trajectories
        self.n_trajectories = len(trajectories)
        
        self.targets = targets
        self.n_targets = len(targets)
        self.n_target_vars = targets[0][0].size
        
        ### check data consistency
        assert self.__compare_trajectories_targets()
    
    
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
    
    
    def __build_VAR_matrix(self, data_matrix):
        
        assert self.VAR_l > 0
        
        if self.VAR_l == 1:
            return data_matrix
        
        else:
            VAR_matrix = []
            
            #transform reduced data matrix back to an ndarray of the individual reduced trajectories
            reduced_trajectories = np.asarray(np.split(data_matrix, self.n_trajectories, axis=0))

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
    
    
    def __build_target_matrix(self):
        
        if self.full_hist == False:
            target_matrix = np.concatenate(self.targets, axis=0)
        else:
            assert self.VAR_l > 0
            
            target_matrix = []
            
            for r in range(len(self.targets)):
                target_matrix.append(self.targets[r][self.VAR_l-1:])
                
            target_matrix = np.concatenate(target_matrix, axis=0)
        
        return target_matrix

  
    def train(self, rdim = None, prdim = None, VAR_l = None, intercept=None, full_hist=None, dim_reducer = None, scaler = None, VAR_transformer = None, optimizer = None):
        
        assert (self.trajectories != None and self.targets != None)
        
        ### check data consistency
        assert self.__compare_trajectories_targets()
        
        if rdim != None:
            self.rdim = rdim
        if prdim != None:
            self.prdim = prdim
        else:
            if rdim != None:
                self.prdim = rdim
        
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
        
        #apply the dimensionality reduction to get the reduced coefficient matrix with prdim features via the dim_reducer object
        data_matrix = np.concatenate(self.trajectories,axis=0)

        if self.reduce_dim == True:
            self.dim_reducer.train(data_matrix,self.prdim)
            self.reduced_data_matrix = self.dim_reducer.reduce(data_matrix,self.prdim)
        else:
            self.reduced_data_matrix = data_matrix
            self.rdim = data_matrix.shape[1]
            self.prdim = data_matrix.shape[1]

        #apply data/feature scaling via scaler object
        if self.standardize:
            self.scaler.train(self.reduced_data_matrix)
            self.reduced_data_matrix = self.scaler.transform(self.reduced_data_matrix)
           
        #create training data matrices
        self.training_matrix = self.__build_VAR_matrix(self.reduced_data_matrix)    
        self.target_matrix = self.__build_target_matrix()
        
        
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

        #setup numpy array for the predession
        pred = np.zeros((trajectory.shape[0],self.n_target_vars))
        
        
        feature_matrix = self.VAR_transformer.transform(self.__build_VAR_matrix(reduced_trajectory))
        
        #add bias/intercept 
        if self.intercept:
            feature_matrix = np.concatenate( [ feature_matrix, np.ones( (feature_matrix.shape[0],1) ) ] , axis=1 )
        
        #let the machine predict the dynamics
        for j in range(0, pred.shape[0]):
            #predict the next step
            pred[j] = feature_matrix[j] @ self.w

        return pred
    
    
                
    def print_status(self):
        print('full_hist: ', self.full_hist)
        print('intercept: ', self.intercept)
        print('standardize: ', self.standardize)
        print('rdim: ', self.rdim)
        print('prdim: ', self.prdim)
        print('VAR_l: ', self.VAR_l)
        print('train shape: ', self.training_matrix.shape)
        print('target shape: ', self.target_matrix.shape)
        print('weights shape: ', self.w.shape)
