import numpy as np

import narrom_dim_reducer as dim_reducer
import narrom_optimizer as optimizer
import narrom_scaler as scaler
import narrom_transformer as transformer


class narrom:
  
    def __init__(self, trajectories = None, rdim = 1, prdim=None, VAR_k = 1, full_hist=False, intercept = False, dim_reducer = None, scaler = None, VAR_transformer = None, optimizer = None):
        
        if trajectories != None:
            self.trajectories = trajectories
            self.n_trajectories = len(trajectories)
        
        self.rdim = rdim
        if prdim == None:
            self.prdim = self.rdim
        else:
            self.prdim = prdim
        
        self.VAR_k = VAR_k
        
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
        
        
    def load_trajectories(self,trajectories):
        self.trajectories = trajectories
        self.n_trajectories = len(trajectories)
    
    
    def __build_VAR_vec(self, matrix, row, VAR_k):
        VAR_vec = []
        for k in range(VAR_k):
            if row+k < 0:
                VAR_vec.append(matrix[0])
            else:
                VAR_vec.append(matrix[row+k])
        return np.concatenate(VAR_vec, axis=0)
    
    
    def __build_VAR_training_matrices(self):
    
        training_matrix = []
        target_matrix = []
        
        #transform reduced data matrix back to an ndarray of the individual reduced trajectories
        reduced_trajectories = np.asarray(np.split(self.reduced_data_matrix, self.n_trajectories, axis=0))

        for r in range(len(reduced_trajectories)):

            if(self.full_hist == False):
                nRows = reduced_trajectories[r].shape[0]-1
                Delta_j = self.VAR_k-1
            else:
                nRows = reduced_trajectories[r].shape[0]-self.VAR_k
                Delta_j = 0
                
            nCols = self.rdim*self.VAR_k

            run_VAR_matrix = np.zeros((nRows,nCols))
            for j in range(nRows):
                run_VAR_matrix[j] = self. __build_VAR_vec(reduced_trajectories[r][:,:self.rdim], j-Delta_j, self.VAR_k)

            training_matrix.append(run_VAR_matrix)
            if self.full_hist == False:
                target_matrix.append(reduced_trajectories[r][1:])
            else:
                target_matrix.append(reduced_trajectories[r][self.VAR_k:])

        training_matrix = np.concatenate(training_matrix, axis=0)
        target_matrix = np.concatenate(target_matrix, axis=0)

        return training_matrix, target_matrix
      

  
    def train(self, rdim = None, prdim = None, VAR_k = None, intercept=None, full_hist=None, dim_reducer = None, scaler = None, VAR_transformer = None, optimizer = None):
        
        assert self.trajectories != None
        
        if rdim != None:
            self.rdim = rdim
        if prdim != None:
            self.prdim = prdim
        else:
            if rdim != None:
                self.prdim = rdim
        
        if VAR_k != None:
            self.VAR_k = VAR_k
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
            self.dim_reducer.train(data_matrix)
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
        self.training_matrix, self.target_matrix = self.__build_VAR_training_matrices()    
        
        #apply transformation to the VAR state
        if self.transform_VAR == True:
            self.VAR_transformer.setup(self.training_matrix.shape[0])
            self.training_matrix = self.VAR_transformer.transform(self.training_matrix)

        #add bias/intercept
        if self.intercept:
            self.training_matrix = np.concatenate( [ self.training_matrix, np.ones( (self.training_matrix.shape[0],1) ) ] , axis=1 )

        #calculate weight matrix via optimizer object
        self.w = self.optimizer.solve(self.training_matrix, self.target_matrix)
                          

    def predict_test_trajectory(self, test_trajectory):
        
        #apply the dimensionality reduction to the test_trajectory
        if self.reduce_dim == True:
            reduced_test_trajectory = self.dim_reducer.reduce(test_trajectory,self.prdim)
        else:
            reduced_test_trajectory = test_trajectory

        #apply data/feature scaling
        if self.standardize:
            reduced_test_trajectory = self.scaler.transform(reduced_test_trajectory)

        #setup numpy array for the auto prediction
        pred = np.zeros(reduced_test_trajectory.shape)

        #build initial condition for the auto predictions
        if (self.full_hist == False):
            j_start = 1
            pred[0] = reduced_test_trajectory[0]
        else:
            j_start = self.VAR_k
            for l in range(self.VAR_k):
                pred[l] = reduced_test_trajectory[l]

        #let the machine predict the dynamics
        for j in range(j_start,pred.shape[0]):
        
            #build the VAR vector from the past steps
            VAR_vec = self.__build_VAR_vec(pred[:,:self.rdim], j-self.VAR_k, self.VAR_k)
            VAR_vec = VAR_vec.reshape((self.rdim*self.VAR_k,1))
            
            #apply transformation to the VAR state
            if self.transform_VAR == True:
                transform = self.VAR_transformer.transform(VAR_vec.T)

            #add intercept/bias
            if self.intercept:
                transform = np.append(transform, 1.0)
                          
            #predict the next step
            pred[j] = self.w.T @ transform

        #undo the data/feature scaling
        if self.standardize:
            pred = self.scaler.inverse_transform(pred)
        
        #reconstruct the full from the reduced representation
        if self.reduce_dim == True:
            pred = self.dim_reducer.reconstruct(pred)

        return pred
    
    def predict(self,init,n_steps):
        #apply the dimensionality reduction to the initital conditions
        if self.reduce_dim == True:
            reduced_init = self.dim_reducer.reduce(init,self.prdim)
        else:
            reduced_init = init
        
        #apply data/feature scaling
        if self.standardize:
            reduced_init = self.scaler.transform(reduced_init)

        #setup numpy array for the auto prediction
        pred = np.zeros((reduced_init.shape[0],n_steps))

        #build initial condition for the auto predictions
        if (self.full_hist == False):
            j_start = 1
            pred[:,0] = coef_init[:,0]
        else:
            j_start = self.VAR_k
            for l in range(self.VAR_k):
                pred[:,l] = reduced_init[:,l]

        #let the machine predict the dynamics
        for j in range(j_start,pred.shape[1]):
        
            #build the VAR vector from the past steps
            VAR_vec = self.__build_VAR_vec(pred[:self.rdim], j-self.VAR_k, self.VAR_k)
            VAR_vec = VAR_vec.reshape((self.rdim*self.VAR_k,1))

            #apply transformation to the VAR state
            if self.transform_VAR == True:
                transform = self.VAR_transformer.transform(VAR_vec)

            #add intercept/bias
            if self.intercept:
                transform = np.append(transform, 1.0)
                          
            #predict the next step
            pred[:,j] = self.w.T @ transform

        #undo the data/feature scaling
        if self.standardize:
            pred = self.scaler.inverse_transform(pred)
        
        #reconstruct the full from the reduced representation
        if self.reduce_dim == True:
            pred = self.dim_reducer.reconstruct(pred)

        return pred
        
        
        
          
    def get_error(self, test_trajectory, pred=np.zeros(1), norm='NF'):
        if pred.size == 1:
            pred = self.predict_test_trajectory(test_trajectory)
        
        err = -1.
        if norm == 'fro': #Frobenius norm
            err = np.linalg.norm(test_trajectory-pred, ord='fro')
        elif norm =='max': #absolute max norm
            err = np.abs(test_trajectory-pred).max()
        elif norm =='NF': #normalized Frobenius norm
            err = np.sqrt( np.mean( np.square(test_trajectory-pred) ) )
        else:
            print('unknown norm')
        
        return err
                
                
    def score_multiple_trajectories(self,trajectories, **kwargs):
        scores = []
        for k in range(len(trajectories)):
            scores.append(self.get_error(trajectories[k], **kwargs))
        
        mean = np.mean(scores)
        return mean, scores
                
    def print_status(self):
        print('full_hist: ', self.full_hist)
        print('intercept: ', self.intercept)
        print('standardize: ', self.standardize)
        print('rdim: ', self.rdim)
        print('prdim: ', self.prdim)
        print('VAR_k: ', self.VAR_k)
        print('train shape: ', self.training_matrix.shape)
        print('target shape: ', self.target_matrix.shape)
        print('weights shape: ', self.w.shape)
