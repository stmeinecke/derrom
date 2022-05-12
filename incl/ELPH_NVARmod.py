import numpy as np
import sys
import copy


sys.path.append("incl/")
from ELPH_VAR import SVDVAR

class SVDNVAR(SVDVAR):
  
    def __init__(self, runs, rdim = 1, n_VAR_steps = 1, NVAR_p = 1):
        super().__init__(runs, rdim, n_VAR_steps) 
        self.NVAR_p = NVAR_p
    
    
    def build_VAR_p_Vec(self, VAR_vec, order=2):
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
    
        nRows = self.build_VAR_p_Vec(self.VAR_state[:,0], order=self.NVAR_p).size
        nCols = self.VAR_state.shape[1]

        NVAR_state = np.zeros((nRows,nCols)) 
        #print(NVAR_state.shape)

        for k in range( NVAR_state.shape[1] ):
            NVAR_state[:,k] = self.build_VAR_p_Vec(self.VAR_state[:,k], order=self.NVAR_p)

        return NVAR_state
    
    
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

        for r in range(len(self.red_coef_runs)):

            if(self.full_hist == False):
                nCols = self.red_coef_runs[r].shape[1]-1
                Delta_j = self.n_VAR_steps-1
            else:
                nCols = self.red_coef_runs[r].shape[1]-self.n_VAR_steps
                Delta_j = 0
                
            nRows = self.red_coef_runs[r].shape[0]*self.n_VAR_steps

            run_VAR_matrix = np.zeros((nRows,nCols))
            for j in range(nCols):
                run_VAR_matrix[:,j] =self. __build_VAR_vec(self.red_coef_runs[r], j-Delta_j, self.n_VAR_steps)

            state.append(run_VAR_matrix)
            if self.full_hist == False:
                target.append(self.coef_runs[r][:,1:])
                #target.append(self.red_coef_runs[r][:,1:])
                #target.append(self.red_coef_runs[r][:,1:]-self.red_coef_runs[r][:,:-1])
            else:
                target.append(self.coef_runs[r][:,self.n_VAR_steps:])
                #target.append(self.red_coef_runs[r][:,self.n_VAR_steps:])
                #target.append(self.red_coef_runs[r][:,self.n_VAR_steps:]-self.red_coef_runs[r][:,self.n_VAR_steps-1:-1])

        state = np.concatenate(state, axis=1)
        target = np.concatenate(target, axis=1)

        return state,target
  
  
    def train(self, rdim = None, n_VAR_steps = None, NVAR_p = None, intercept=None, full_hist=None, scaler = None, optimizer = None, column_weights = np.zeros(1),  **kwargs):
        
        if rdim != None:
            self.rdim = rdim
        if n_VAR_steps != None:
            self.n_VAR_steps = n_VAR_steps
        if NVAR_p != None:
            self.NVAR_p = NVAR_p
        if intercept != None:
            self.intercept = intercept
        if full_hist != None:
            self.full_hist = full_hist
        
        if scaler != None:
            self.scaler = copy.deepcopy(scaler)
            self.scaler_red = copy.deepcopy(scaler)
            self.standardize = True
        else:
            self.standardize = False
        
        if optimizer != None:
            self.optimizer = optimizer
        else:
            self.optimizer = ELPH_Optimizer.lstsqrs()
            
        if column_weights.size != 1:
            self.column_weights = column_weights
        else:
            self.column_weights = np.ones(self.runs[0].shape[1])


        #calculate SVD decomposition of the training runs
        data_matrix = np.concatenate(self.runs,axis=1)
        n_cols = self.runs[0].shape[1]
        #apply column_weights
        for r in range(len(self.runs)):
            data_matrix[:,r*n_cols:(r+1)*n_cols] *= self.column_weights
        self.U,self.S = np.linalg.svd(data_matrix, full_matrices=False)[:2]
        self.Uhat = self.U[:,:self.rdim]

        #project training data onto the first rdim columns of the SVD U-Matrix
        self.coef_matrix = self.U.T @ data_matrix
        self.red_coef_matrix = self.Uhat.T @ data_matrix
        
        #apply data/feature scaling via scaler object
        if self.standardize:
            self.scaler.train(self.coef_matrix)
            self.coef_matrix = self.scaler.transform(self.coef_matrix)
            self.scaler_red.train(self.red_coef_matrix)
            self.red_coef_matrix = self.scaler_red.transform(self.red_coef_matrix)

        #transform coeffiecient matrix back to an ndarray of the individual coefficient runs
        self.coef_runs = np.asarray(np.split(self.coef_matrix, self.n_runs, axis=1))
        self.red_coef_runs = np.asarray(np.split(self.red_coef_matrix, self.n_runs, axis=1))

        #create training data matrices
        self.VAR_state, self.target = self.__build_VAR_training_matrices()

        self.NVAR_state = self.__build_NVAR_training_matrices()

        #add bias/intercept
        if intercept:
            self.NVAR_state = np.concatenate( [self.NVAR_state, np.ones((1,self.NVAR_state.shape[1]))], axis=0 )

        #calculate weight matrix via optimizer object
        self.w = self.optimizer.solve(self.NVAR_state, self.target)
                          

    def predict_single_run(self, run):

        #project run onto the first rdim SVD components, i.e. the first rdim columns of the SVD U-matrix
        
        #weighted cols
        wrun = run * self.column_weights
        
        coef_run = self.U.T @ wrun
        red_coef_run = self.Uhat.T @ wrun
        
        red_size = red_coef_run.shape[0]
        #apply data/feature scaling
        if self.standardize:
            coef_run = self.scaler.transform(coef_run)
            red_coef_run = self.scaler_red.transform(red_coef_run)

        #setup numpy array for the auto prediction
        pred = np.zeros(coef_run.shape)
        red_pred = np.zeros(red_coef_run.shape)

        #build initial condition for the auto predictions
        if (self.full_hist == False):
            j_start = 1
            pred[:,0] = coef_run[:,0]
            red_pred[:,0] = red_coef_run[:,0]
        else:
            j_start = self.n_VAR_steps
            for l in range(self.n_VAR_steps):
                pred[:,l] = coef_run[:,l]
                red_pred[:,l] = red_coef_run[:,l]

        #let the machine predict the electron dynamics
        for j in range(j_start,pred.shape[1]):
        
            #build the VAR vector from the past steps
            VAR_vec = self._SVDVAR__build_VAR_vec(red_pred, j-self.n_VAR_steps, self.n_VAR_steps)

            #build the NVAR vector to the specified order from the VAR vector
            NVAR_vec = self.build_VAR_p_Vec(VAR_vec, order=self.NVAR_p)

            #add intercept/bias
            if self.intercept:
                NVAR_vec = np.append(NVAR_vec, 1.0)
                          
            #predict the next step
            pred[:,j] = self.w.T @ NVAR_vec
            red_pred[:,j] = pred[:red_size,j]
            #pred[:,j] = pred[:,j-1] + self.w.T @ NVAR_vec

        #undo the data/feature scaling
        if self.standardize:
            pred = self.scaler.inverse_transform(pred)
            red_pred = self.scaler_red.inverse_transform(red_pred)
        #project back onto the k space of the electron distribution
        pred = self.U @ pred 
        red_pred = self.Uhat @ red_pred 
        
        #weighted cols
        pred /= self.column_weights
        red_pred /= self.column_weights

        return pred
                          
                          
    def print_status(self):
        print('rdim: ', self.rdim)
        print('n_VAR_steps: ', self.n_VAR_steps)
        print('NVAR_p: ', self.NVAR_p)
        print('VAR state shape: ', self.VAR_state.shape)
        print('NVAR state shape: ', self.NVAR_state.shape)
        print('target shape: ', self.target.shape)
        print('weights shape: ', self.w.shape)
