import numpy as np
import sys

sys.path.append("incl/")


class RDNLregr:
  
    def __init__(self, runs, rdim = 1, prdim=None, n_VAR_steps = 1, intercept = False, scaler = None, full_hist=False, NVAR_p = 1):
        
        self.runs = runs
        self.n_runs = len(runs)
        
        self.rdim = rdim
        if prdim == None:
            self.prdim = self.rdim
        else:
            self.prdim = prdim
        self.n_VAR_steps = n_VAR_steps
        self.NVAR_p = NVAR_p
        
        self.intercept = intercept
        self.full_hist = full_hist
        
        if scaler != None:
            self.scaler = scaler
            self.standardize = True
        else:
            self.standardize = False

        
    
    
    def load_runs(self,runs):
        self.runs = runs
        self.n_runs = len(runs)
    
    
    def __build_VAR_vec(self, matrix, col, n_VAR_steps):
        VAR_vec = []
        for k in range(n_VAR_steps):
            if col+k < 0:
                VAR_vec.append(matrix[:,0])
            else:
                VAR_vec.append(matrix[:,col+k])
        return np.concatenate(VAR_vec, axis=0)
    
    
    def __build_VAR_training_matrices(self):
    
        state = []
        target = []
        
        #transform coeffiecient matrix back to an ndarray of the individual coefficient runs
        coef_runs = np.asarray(np.split(self.coef_matrix, self.n_runs, axis=1))

        for r in range(len(coef_runs)):

            if(self.full_hist == False):
                nCols = coef_runs[r].shape[1]-1
                Delta_j = self.n_VAR_steps-1
            else:
                nCols = coef_runs[r].shape[1]-self.n_VAR_steps
                Delta_j = 0
                
            nRows = self.rdim*self.n_VAR_steps

            run_VAR_matrix = np.zeros((nRows,nCols))
            for j in range(nCols):
                run_VAR_matrix[:,j] =self. __build_VAR_vec(coef_runs[r][:self.rdim,:], j-Delta_j, self.n_VAR_steps)

            state.append(run_VAR_matrix)
            if self.full_hist == False:
                target.append(coef_runs[r][:,1:])
            else:
                target.append(coef_runs[r][:,self.n_VAR_steps:])

        state = np.concatenate(state, axis=1)
        target = np.concatenate(target, axis=1)

        return state,target
  
    def __build_VAR_p_Vec(self, VAR_vec, order=2):
        VAR_p_Vec = [VAR_vec]
        VARp = VAR_vec
        for p in range(1,order):
            VARp = np.outer(VAR_vec,VARp)
            VARp = VARp[ np.triu_indices(VARp.shape[0], m=VARp.shape[1]) ]
            VAR_p_Vec.append(VARp)
        return np.concatenate(VAR_p_Vec, axis=0)
  
  
  #def build_VAR_p_Vec(self, VAR_vec, order=2):
    #VAR_p_Vec = [VAR_vec]
    #VARp = VAR_vec
    #for p in range(1,order):
      #VARp = np.multiply(VARp,VAR_vec)
      #VAR_p_Vec.append(VARp)
    #return np.concatenate(VAR_p_Vec, axis=0)
  
    
    def __build_NVAR_training_matrices(self):
    
        nRows = self.__build_VAR_p_Vec(self.VAR_state[:,0], order=self.NVAR_p).size
        nCols = self.VAR_state.shape[1]

        NVAR_state = np.zeros((nRows,nCols)) 
        #print(NVAR_state.shape)

        for k in range( NVAR_state.shape[1] ):
            NVAR_state[:,k] = self.__build_VAR_p_Vec(self.VAR_state[:,k], order=self.NVAR_p)

        return NVAR_state

  
    def train(self, rdim = None, prdim = None, n_VAR_steps = None, NVAR_p = None, intercept=None, full_hist=None, scaler = None, optimizer = None, dim_reducer = None, VAR_transformer = None,  **kwargs):
        
        if rdim != None:
            self.rdim = rdim
        if prdim != None:
            self.prdim = prdim
        else:
            if rdim != None:
                self.prdim = rdim
        if n_VAR_steps != None:
            self.n_VAR_steps = n_VAR_steps
        if NVAR_p != None:
            self.NVAR_p = NVAR_p
        if intercept != None:
            self.intercept = intercept
        if full_hist != None:
            self.full_hist = full_hist
        
        if scaler != None:
            self.scaler = scaler
            self.standardize = True
        else:
            self.standardize = False
        
        if optimizer != None:
            self.optimizer = optimizer
        else:
            self.optimizer = ELPH_Optimizer.lstsqrs()
            
        if dim_reducer != None:
            self.dim_reducer = dim_reducer
            
        if VAR_transformer != None:
            self.VAR_transformer = VAR_transformer


        #apply the dimensionality reduction to get the reduced coefficient matrix with prdim features via the dim_reducer object
        data_matrix = np.concatenate(self.runs,axis=1)
        self.dim_reducer.train(data_matrix)
        self.coef_matrix = self.dim_reducer.reduce(data_matrix,self.prdim)

        
        #apply data/feature scaling via scaler object
        if self.standardize:
            self.scaler.train(self.coef_matrix)
            self.coef_matrix = self.scaler.transform(self.coef_matrix)

        #create training data matrices
        self.training_matrix, self.target = self.__build_VAR_training_matrices()
        
        self.training_matrix = self.VAR_transformer.transform(self.training_matrix)

#         self.NVAR_state = self.__build_NVAR_training_matrices()

        #add bias/intercept
        if intercept:
            self.training_matrix = np.concatenate( [self.training_matrix, np.ones((1,self.training_matrix.shape[1]))], axis=0 )

        #calculate weight matrix via optimizer object
        self.w = self.optimizer.solve(self.training_matrix, self.target)
                          

    def predict_single_run(self, run):
        
        #apply the dimensionality reduction to the run
        coef_run = self.dim_reducer.reduce(run,self.prdim)

        #apply data/feature scaling
        if self.standardize:
            coef_run = self.scaler.transform(coef_run)

        #setup numpy array for the auto prediction
        pred = np.zeros(coef_run.shape)

        #build initial condition for the auto predictions
        if (self.full_hist == False):
            j_start = 1
            pred[:,0] = coef_run[:,0]
        else:
            j_start = self.n_VAR_steps
            for l in range(self.n_VAR_steps):
                pred[:,l] = coef_run[:,l]

        #let the machine predict the dynamics
        for j in range(j_start,pred.shape[1]):
        
            #build the VAR vector from the past steps
            VAR_vec = self.__build_VAR_vec(pred[:self.rdim], j-self.n_VAR_steps, self.n_VAR_steps)
            VAR_vec = VAR_vec.reshape((self.rdim*self.n_VAR_steps,1))

            #build the NVAR vector to the specified order from the VAR vector
            transform = self.VAR_transformer.transform(VAR_vec)

            #add intercept/bias
            if self.intercept:
                transform = np.append(transform, 1.0)
                          
            #predict the next step
            pred[:,j] = self.w.T @ transform
            #pred[:,j] = pred[:,j-1] + self.w.T @ NVAR_vec

        #undo the data/feature scaling
        if self.standardize:
            pred = self.scaler.inverse_transform(pred)
        
        #expand the reduced representation to full dimension
        pred = self.dim_reducer.expand(pred)

        return pred
          
          
    def get_error(self, run, pred=np.zeros(1), norm='fro', errSVD = False):
        if pred.size == 1:
            pred = self.predict_single_run(run)
        
        if errSVD == True:
            run = self.U_rdim @ self.U_rdim.T @ run
        
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
        print('rdim: ', self.rdim)
        print('n_VAR_steps: ', self.n_VAR_steps)
#         print('NVAR_p: ', self.NVAR_p)
#         print('VAR state shape: ', self.VAR_state.shape)
        print('train shape: ', self.training_matrix.shape)
        print('target shape: ', self.target.shape)
        print('weights shape: ', self.w.shape)
