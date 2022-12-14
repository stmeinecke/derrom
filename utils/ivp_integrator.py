import numpy as np

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
    
    def load_data(self, **kwargs):
        self.model.load_data(**kwargs)
    
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
        
    def load_data(self, trajectories, targets):
        el_trajectories = [trajectory[:,:-1] for trajectory in trajectories]
        self.model.load_data(el_trajectories,targets)
    
    
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
            err = (np.argmax(I_pred)-np.argmax(I_truth))
        elif norm == 'I_area':
            err = (np.sum(I_pred) - np.sum(I_truth))
        else:
            print('unknown norm')
        
        return err
