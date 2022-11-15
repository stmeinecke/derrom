import numpy as np

class ivp_integrator:
    
    def __init__(self, model, dt=1., dt_out=1., method='Heun'):
        self.model = model
        self.dt = dt
        self.dt_out = dt_out
        self.method = method
        
        self.targets = 'AR'
    
    def load_data(self, **kwargs):
        self.model.load_data(**kwargs)
    
    def train(self, **kwargs):
        self.model.train(**kwargs)
    
    
    #def __Euler(self, init, n_steps, dt, dt_out):
        
        #sol = np.zeros((n_steps,init.shape[1]))
        
        #sol[:init.shape[0]] = init
        
        #state = sol[0]
        
        #j_out = int(dt_out/dt)
        #j_max = sol.shape[0]*j_out
        
        
        #for j in range(1,sol.shape[0]*j_out):
            #state = state + dt*self.model.predict(state)
            
            #if j%j_out == 0:
                #sol[j//j_out] = state
                
        #return sol


    def __Euler(self, init, n_steps, dt, dt_out):
        
        sol = np.zeros((n_steps,init.shape[1]))
        
        sol[0] = init
        
        state = sol[0]
        
        j_out = int(dt_out/dt)
        j_max = sol.shape[0]*j_out
        
        hist_length = (self.model.VAR_l-1)*j_out + 1
        hist_ind = hist_length - 1
        hist = np.zeros((hist_length,init.shape[1]))
        for k in range(hist.shape[0]):
            hist[k] = init
        
        
        for j in range(1,sol.shape[0]*j_out):
            
            vecs = [hist[(hist_ind - n*j_out + self.model.VAR_l*j_out)%hist_length] for n in range(self.model.VAR_l)]
            
            vecs = np.stack(vecs)
            
            state = state + dt*self.model.predict(vecs)
            
            hist_ind = (hist_ind+1)%hist_length
            
            hist[hist_ind] = state
            
            if j%j_out == 0:
                sol[j//j_out] = state
                
        return sol
    
    
    def __Heun(self, init, n_steps, dt, dt_out):
        
        sol = np.zeros((n_steps,init.shape[1]))
        
        sol[:init.shape[0]] = init
        
        state = sol[0]
        
        j_out = int(dt_out/dt)
        j_max = sol.shape[0]*j_out
        
        
        for j in range(1,sol.shape[0]*j_out):
            
            f1 = self.model.predict(state)
            f2 = self.model.predict(state + dt*self.model.predict(state))
            
            state = state + 0.5*dt*(f1+f2)
            
            if j%j_out == 0:
                sol[j//j_out] = state
                
        return sol

    
    
    def integrate(self, init, n_steps, dt=None, dt_out=None):
        
        if dt is None:
            dt = self.dt
        if dt_out is None:
            dt_out = self.dt_out
        
        if self.method == 'Heun':
            return self.__Heun(init,n_steps,dt,dt_out)
        elif self.method == 'Euler':
            return self.__Heun(init,n_steps,dt,dt_out)
        else:
            raise ValueError('integration method >> ' + self.method + ' << does not exist')

    
    def get_error(self, truth, pred=None, norm='NF'):
        
        if pred is None:
            pred = self.integrate(truth, truth.shape[0])
        
        assert pred.shape == truth.shape
        
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
    
    def score_multiple_trajectories(self,trajectories,targets=None,**kwargs):
        scores = []
        for k in range(len(trajectories)):
            scores.append(self.get_error(trajectories[k],**kwargs))
        
        mean = np.mean(scores)
        return mean, scores
