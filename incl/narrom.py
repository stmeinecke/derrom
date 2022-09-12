import numpy as np

import narrom_dim_reducer as dim_reducer
import narrom_optimizer as optimizer
import narrom_scaler as scaler
import narrom_transformer as transformer


class narrom:
  
    def __init__(self, runs = None, rdim = 1, prdim=None, n_VAR_steps = 1, full_hist=False, intercept = False, dim_reducer = None, scaler = None, VAR_transformer = None, optimizer = None):
        
        if runs != None:
            self.runs = runs
            self.n_runs = len(runs)
        
        self.rdim = rdim
        if prdim == None:
            self.prdim = self.rdim
        else:
            self.prdim = prdim
        
        self.n_VAR_steps = n_VAR_steps
        
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
        
        
    def load_runs(self,runs):
        self.runs = runs
        self.n_runs = len(runs)
    
    
    def __build_VAR_vec(self, matrix, row, n_VAR_steps):
        VAR_vec = []
        for k in range(n_VAR_steps):
            if row+k < 0:
                VAR_vec.append(matrix[0])
            else:
                VAR_vec.append(matrix[row+k])
        return np.concatenate(VAR_vec, axis=0)
    
    
    def __build_VAR_training_matrices(self):
    
        training_matrix = []
        target_matrix = []
        
        #transform coeffiecient matrix back to an ndarray of the individual coefficient runs
        coef_runs = np.asarray(np.split(self.coef_matrix, self.n_runs, axis=0))

        for r in range(len(coef_runs)):

            if(self.full_hist == False):
                nRows = coef_runs[r].shape[0]-1
                Delta_j = self.n_VAR_steps-1
            else:
                nRows = coef_runs[r].shape[0]-self.n_VAR_steps
                Delta_j = 0
                
            nCols = self.rdim*self.n_VAR_steps

            run_VAR_matrix = np.zeros((nRows,nCols))
            for j in range(nRows):
                run_VAR_matrix[j] = self. __build_VAR_vec(coef_runs[r][:,:self.rdim], j-Delta_j, self.n_VAR_steps)

            training_matrix.append(run_VAR_matrix)
            if self.full_hist == False:
                target_matrix.append(coef_runs[r][1:])
            else:
                target_matrix.append(coef_runs[r][self.n_VAR_steps:])

        training_matrix = np.concatenate(training_matrix, axis=0)
        target_matrix = np.concatenate(target_matrix, axis=0)

        return training_matrix, target_matrix
      

  
    def train(self, rdim = None, prdim = None, n_VAR_steps = None, intercept=None, full_hist=None, dim_reducer = None, scaler = None, VAR_transformer = None, optimizer = None):
        
        assert self.runs != None
        
        if rdim != None:
            self.rdim = rdim
        if prdim != None:
            self.prdim = prdim
        else:
            if rdim != None:
                self.prdim = rdim
        
        if n_VAR_steps != None:
            self.n_VAR_steps = n_VAR_steps
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
        data_matrix = np.concatenate(self.runs,axis=0)
        
        print('data matrix shape: ', data_matrix.shape)
        
        if self.reduce_dim == True:
            self.dim_reducer.train(data_matrix)
            self.coef_matrix = self.dim_reducer.reduce(data_matrix,self.prdim)
        else:
            self.coef_matrix = data_matrix
            self.rdim = data_matrix.shape[1]
            self.prdim = data_matrix.shape[1]

        print('reduced data matrix shape: ', self.coef_matrix.shape)
        
        #apply data/feature scaling via scaler object
        if self.standardize:
            self.scaler.train(self.coef_matrix)
            self.coef_matrix = self.scaler.transform(self.coef_matrix)
            
        print('reduced data matrix shape: ', self.coef_matrix.shape)

        #create training data matrices
        self.training_matrix, self.target_matrix = self.__build_VAR_training_matrices()    
        
        
        print('training matrix shape: ', self.training_matrix.shape)
        print('target matrix shape: ', self.target_matrix.shape)
        
        #apply transformation to the VAR state
        if self.transform_VAR == True:
            self.VAR_transformer.setup(self.training_matrix.shape[0])
            self.training_matrix = self.VAR_transformer.transform(self.training_matrix)

        #add bias/intercept
        if self.intercept:
            self.training_matrix = np.concatenate( [ self.training_matrix, np.ones( (self.training_matrix.shape[0],1) ) ] , axis=1 )

        #calculate weight matrix via optimizer object
        self.w = self.optimizer.solve(self.training_matrix, self.target_matrix)
                          

    def predict_single_run(self, run):
        
        #apply the dimensionality reduction to the run
        if self.reduce_dim == True:
            coef_run = self.dim_reducer.reduce(run,self.prdim)
        else:
            coef_run = run

        #apply data/feature scaling
        if self.standardize:
            coef_run = self.scaler.transform(coef_run)

        #setup numpy array for the auto prediction
        pred = np.zeros(coef_run.shape)

        #build initial condition for the auto predictions
        if (self.full_hist == False):
            j_start = 1
            pred[0] = coef_run[0]
        else:
            j_start = self.n_VAR_steps
            for l in range(self.n_VAR_steps):
                pred[l] = coef_run[l]

        #let the machine predict the dynamics
        for j in range(j_start,pred.shape[0]):
        
            #build the VAR vector from the past steps
            VAR_vec = self.__build_VAR_vec(pred[:,:self.rdim], j-self.n_VAR_steps, self.n_VAR_steps)
            VAR_vec = VAR_vec.reshape((self.rdim*self.n_VAR_steps,1))
            
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
            coef_init = self.dim_reducer.reduce(init,self.prdim)
        else:
            coef_init = init
        
        #apply data/feature scaling
        if self.standardize:
            coef_init = self.scaler.transform(coef_init)

        #setup numpy array for the auto prediction
        pred = np.zeros((coef_init.shape[0],n_steps))

        #build initial condition for the auto predictions
        if (self.full_hist == False):
            j_start = 1
            pred[:,0] = coef_init[:,0]
        else:
            j_start = self.n_VAR_steps
            for l in range(self.n_VAR_steps):
                pred[:,l] = coef_init[:,l]

        #let the machine predict the dynamics
        for j in range(j_start,pred.shape[1]):
        
            #build the VAR vector from the past steps
            VAR_vec = self.__build_VAR_vec(pred[:self.rdim], j-self.n_VAR_steps, self.n_VAR_steps)
            VAR_vec = VAR_vec.reshape((self.rdim*self.n_VAR_steps,1))

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
        
        
        
          
    def get_error(self, run, pred=np.zeros(1), norm='fro'):
        if pred.size == 1:
            pred = self.predict_single_run(run)
        
        if norm == 'fro': #Frobenius norm
            err = np.linalg.norm(run-pred, ord='fro')
        elif norm =='max':
            err = np.abs(run-pred).max()
        elif norm =='std':
            err = np.sqrt( np.mean( np.square(run-pred) ) )
        else:
            print('unknown norm')
        
        return err
                
                
    def score_multiple_runs(self,runs, **kwargs):
        scores = []
        for k in range(len(runs)):
            scores.append(self.get_error(runs[k], **kwargs))
        
        mean = np.mean(scores)
        return mean, scores
                
    def print_status(self):
        print('full_hist: ', self.full_hist)
        print('intercept: ', self.intercept)
        print('standardize: ', self.standardize)
        print('rdim: ', self.rdim)
        print('prdim: ', self.prdim)
        print('n_VAR_steps: ', self.n_VAR_steps)
        print('train shape: ', self.training_matrix.shape)
        print('target shape: ', self.target_matrix.shape)
        print('weights shape: ', self.w.shape)
