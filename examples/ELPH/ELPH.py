import numpy as np

from scipy.optimize import root as scpy_root
from scipy.integrate import solve_ivp as scpy_solve_ivp

import math


class ELPH(object):
    
    def __init__(self, kmax=2.0, n_kmax=80):
        self.kmax = kmax
        self.n_kmax = n_kmax
                
        #constants
        self.kB = 8.61745*0.00001
        self.m0 = 5.6856800
        self.mP = 10439.60413
        self.hbar = 0.658212196

        self.m = 0.5*self.m0
        self.a0 = 0.314
        self.ca = 4.1/1000.
        self.EAprime = 0.031
        self.ETO = 0.036
        self.Mr = 253*self.mP
        self.Vol = np.sqrt(3.0)/2.0*self.a0*self.a0
        self.Da = 3.4
        self.Do = 5.2*10
        self.T_cryo = 300

        #angle
        self.phimax = 2.*np.pi
        self.n_phimax = 20
        self.dphi = self.phimax/self.n_phimax
        
        self.dk = kmax/n_kmax
        self.k_vec = self.get_k(self.dk,np.arange(0,self.n_kmax-0.5,1))
        self.E_el_vec = self.electron_dispersion(self.k_vec)
        self.DOS_vec = self.electron_DOS(self.k_vec)
        
        self.in_scattering_matrix_em, self.out_scattering_matrix_em, self.in_scattering_matrix_abs, self.out_scattering_matrix_abs, self.absorption_matrix, self.emission_matrix, self.einer = self.build_boltzmann_mats(self.kmax, self.n_kmax)
    
    def get_k(self,dk,n_k):
        return (n_k+1.)*dk
  
    
    def get_phi(self,dphi,n_phi):
        return n_phi*dphi


    def electron_dispersion(self,k,phi=0):
        help = self.hbar*self.hbar*k*k/2./self.m
        return help
    
    def electron_DOS(self,k):
        return k*self.dk/2./np.pi
    
    def get_electron_density(self,state):
        dens = 0
        for j in range(state.size):
            dens += state[j] * self.electron_DOS(self.get_k(self.dk,j))
        return dens

    
    def phonon_dispersion(self,k,phi,alpha):
        if alpha==0:
            result = self.EAprime
        if alpha==1:
            result = self.ETO
        if alpha==2:
            result=self.ca*k
        if alpha==3:
            result=self.ca*k
        return result

    
    def phonon_coupling(self,k,phi,alpha):
        if alpha==0:
            result = np.sqrt(self.Vol*self.hbar*self.hbar/2.0/self.Mr/self.phonon_dispersion(k,phi,0))*self.Do
        if alpha==1:
            result = np.sqrt(self.Vol*self.hbar*self.hbar/2.0/self.Mr/self.phonon_dispersion(k,phi,1))*self.Do
        if alpha==2:
            result = np.sqrt(self.Vol*self.hbar*self.hbar/2.0/self.Mr/self.phonon_dispersion(k,phi,2))*self.Da*k
        if alpha==3:
            result = np.sqrt(self.Vol*self.hbar*self.hbar/2.0/self.Mr/self.phonon_dispersion(k,phi,3))*self.Da*k
        return result
    
    
    def phonon_occupation(self,k,phi,alpha,T):
        result = 1/(np.exp(self.phonon_dispersion(k,phi,alpha)/self.kB/T)-1.)
        return result


    def build_boltzmann_mats(self,kmax, n_kmax):
        dk = self.dk

        in_scattering_matrix_em = np.full((n_kmax,n_kmax,4,n_kmax), 0., dtype=np.float64)
        out_scattering_matrix_em = np.full((n_kmax,n_kmax,4,n_kmax), 0., dtype=np.float64)
        in_scattering_matrix_abs = np.full((n_kmax,n_kmax,4,n_kmax), 0., dtype=np.float64)
        out_scattering_matrix_abs = np.full((n_kmax,n_kmax,4,n_kmax), 0., dtype=np.float64)

        absorption_matrix = np.full((n_kmax,n_kmax,4,n_kmax), 0., dtype=np.float64)
        emission_matrix = np.full((n_kmax,n_kmax,4,n_kmax), 0., dtype=np.float64)


        for n_k in range(n_kmax):
            for n_phi in range(self.n_phimax):
                for beta in range(4):
                    kk = self.get_k(dk,n_k)
                    phi_diff = self.get_phi(self.dphi,n_phi)

                    if True:
                        def delta_fun_phon(x):
                            helper1 = np.sqrt(x*x+self.get_k(dk,n_k)*self.get_k(dk,n_k) - 2.*x*self.get_k(dk,n_k)*np.cos(phi_diff) + 5e-15)
                            result = self.electron_dispersion(self.get_k(dk,n_k),0) - self.electron_dispersion(x,0)+self.phonon_dispersion(helper1,0,beta)
                            return result
                        sol = scpy_root(delta_fun_phon, self.get_k(dk,n_k+1), args=(), method='hybr', jac=None, tol=None, callback=None, options=None)
                        if sol.x>kk:
                            n_null_elec = math.floor(sol.x/dk) - 1
                            interpolator_electrons = np.full(2,0.)
                            interpolator_electrons[0] =  np.absolute(delta_fun_phon(self.get_k(dk,n_null_elec+1)))/(np.absolute(delta_fun_phon(self.get_k(dk,n_null_elec)))+ np.absolute(delta_fun_phon(self.get_k(dk,n_null_elec+1))))
                            interpolator_electrons[1] =  np.absolute(delta_fun_phon(self.get_k(dk,n_null_elec)))/(np.absolute(delta_fun_phon(self.get_k(dk,n_null_elec)))+ np.absolute(delta_fun_phon(self.get_k(dk,n_null_elec+1))))
                            for n_kt in range (n_null_elec,n_null_elec+2):
                                kkt = self.get_k(dk,n_kt)


                                if (kkt*kkt+kk*kk - 2.*kkt*kk*np.cos(phi_diff) > 0):
                                    helper = np.sqrt(kkt*kkt+kk*kk - 2.*kkt*kk*np.cos(phi_diff))
                                    if helper>0 and n_kt<n_kmax:
                                        interpolator_phonons = np.full(2,0.)
                                        n_null_phon = int(helper/dk)-1
                                        help_diff_lower = np.absolute(self.get_k(dk,n_null_phon) - helper)
                                        help_diff_upper = np.absolute(self.get_k(dk,n_null_phon+1) - helper)
                                        interpolator_phonons[0] = help_diff_upper/(help_diff_lower + help_diff_upper)
                                        interpolator_phonons[1] = help_diff_lower/(help_diff_lower + help_diff_upper)
                                        for n_q in range(n_null_phon,n_null_phon+2):
                                            qq = self.get_k(dk,n_q)
                                            if n_q > -1 and n_q < n_kmax:
                                                part1 = self.phonon_coupling(qq,0,beta)*self.phonon_coupling(qq,0,beta)
                                                if beta<2:
                                                    part2 = 1./np.absolute(self.hbar*self.hbar*self.get_k(dk,n_kt)/self.m)/dk
                                                if beta>1:
                                                    part2 =  1./np.absolute(self.hbar*self.hbar*self.get_k(dk,n_kt)/self.m-self.hbar*self.ca*(kkt-kk*np.cos(phi_diff))/qq)/dk
                                                #absorption
                                                out_scattering_matrix_abs[n_k][n_kt][beta][n_q] +=- 2.*np.pi/self.hbar*dk*kkt*self.dphi/4./np.pi/np.pi*part1*part2*interpolator_electrons[n_kt-n_null_elec]*interpolator_phonons[n_q-n_null_phon]#(1-fkt) fk n
                                                in_scattering_matrix_abs[n_kt][n_k][beta][n_q] += 2.*np.pi/self.hbar*dk*kk*self.dphi/4./np.pi/np.pi*part1*part2*interpolator_electrons[n_kt-n_null_elec]*interpolator_phonons[n_q-n_null_phon]#(1-fk) fkt n
                                                absorption_matrix[n_k][n_kt][beta][n_q] +=  - 2.*np.pi/self.hbar*dk*kk*self.dphi/4./np.pi/np.pi*part1*part2*interpolator_electrons[n_kt-n_null_elec]*interpolator_phonons[n_q-n_null_phon]
                                                #emission
                                                in_scattering_matrix_em[n_k][n_kt][beta][n_q] +=  2.*np.pi/self.hbar*dk*kkt*self.dphi/4./np.pi/np.pi*part1*part2*interpolator_electrons[n_kt-n_null_elec]*interpolator_phonons[n_q-n_null_phon]#(1-fkt) fk (1+n)
                                                out_scattering_matrix_em[n_kt][n_k][beta][n_q] += - 2.*np.pi/self.hbar*dk*kk*self.dphi/4./np.pi/np.pi*part1*part2*interpolator_electrons[n_kt-n_null_elec]*interpolator_phonons[n_q-n_null_phon]#(1-fk) fkt (1+n)
                                                emission_matrix[n_k][n_kt][beta][n_q] += 2.*np.pi/self.hbar*dk*kk*self.dphi/4./np.pi/np.pi*part1*part2*interpolator_electrons[n_kt-n_null_elec]*interpolator_phonons[n_q-n_null_phon]

        einer = np.full(n_kmax,1.)

        return [in_scattering_matrix_em, out_scattering_matrix_em, in_scattering_matrix_abs, out_scattering_matrix_abs, absorption_matrix, emission_matrix, einer]

    
    def derivs(self,t,y):
    
        y = np.reshape(y,(5,self.n_kmax))
        right_ele = y[0]
        right_phon = y[1:]

        block_ele = self.einer - right_ele
        stim_phon = self.einer + right_phon

        help_phonon_weg = np.tensordot(self.in_scattering_matrix_em,stim_phon,axes = ([2,3],[0,1])) + np.tensordot(self.in_scattering_matrix_abs,right_phon,axes = ([2,3],[0,1]))
        help_source_weg = np.tensordot(help_phonon_weg,right_ele,axes = (1,0))
        temp_ele_in = np.multiply(help_source_weg,block_ele)

        help_phonon_weg = np.tensordot(self.out_scattering_matrix_em,stim_phon,axes = ([2,3],[0,1])) + np.tensordot(self.out_scattering_matrix_abs,right_phon,axes = ([2,3],[0,1]))
        help_source_weg = np.tensordot(help_phonon_weg,block_ele,axes = (1,0))
        temp_ele_out = np.multiply(help_source_weg,right_ele)


        block_weg = np.tensordot(self.absorption_matrix,block_ele,axes = (1,0))
        temp_phon_abs = np.tensordot(block_weg,right_ele,axes = (0,0))
        temp_phon_abs = np.multiply(temp_phon_abs,right_phon)

        block_weg = np.tensordot(self.emission_matrix,right_ele,axes = (1,0))
        block_weg_2 = np.tensordot(block_weg,block_ele,axes = (0,0))
        temp_phon_em = np.multiply(block_weg_2,stim_phon)

        left_phon = temp_phon_abs + temp_phon_em

        left_ele = temp_ele_in + temp_ele_out
        
        left_phon = left_phon.flatten()
        result = np.concatenate((left_ele,left_phon))

        return result
    
    
    def get_full_trajectory(self, init_cond, tmax = 2000.0, n_tmax = 400, **kwargs):

        t_values = np.linspace(0.0, tmax, n_tmax)
        sol = scpy_solve_ivp(self.derivs, [t_values[0],t_values[-1]], init_cond, t_eval=t_values, **kwargs)
        
        #y_values = np.reshape(np.asarray(sol.y).T,(n_tmax,5,self.n_kmax))
        #return y_values[:,:,:]
    
        return sol.y.T
    
    def get_electron_trajectory(self, init_cond, tmax = 2000.0, n_tmax = 400, **kwargs):
        return self.get_full_trajectory(init_cond, tmax = tmax, n_tmax = n_tmax, **kwargs)[:,:self.n_kmax]
    
    def get_electron_derivs(self,state):
        return self.derivs(0,state)[:self.n_kmax]
    
    
    def get_init_cond_gauss(self,max_pos=0.2, width=0.05, density = 0.1):
        dk = self.dk
        n_kmax = self.n_kmax

        initial_ele = np.full(n_kmax,0.)
        initial_phon = np.full((4,n_kmax),0.)
        for n_k in range(n_kmax):
            helper_expo = self.electron_dispersion(self.get_k(dk,n_k),0) - max_pos
            helper = np.exp(-helper_expo*helper_expo/width/width/2.)
            initial_ele[n_k] = helper
        helper = 0.
        for n_k in range(n_kmax):
            helper += dk*self.get_k(dk,n_k)/2./np.pi*initial_ele[n_k]
        initial_ele = initial_ele/helper*density
        for n_k in range(n_kmax):
            for a in range(4):
                initial_phon[a][n_k] = self.phonon_occupation(self.get_k(dk,n_k),0.,a,self.T_cryo)

        initial_ele = np.reshape(initial_ele,(1,n_kmax))

        initial_state = np.concatenate((initial_ele,initial_phon),axis = 0 )
        initial_state = np.reshape(initial_state,5*n_kmax)

        return initial_state
    
    
    ### quick and dirty Runge-Kutta solver
    def integrate(self, init, n_steps=501, dt=1.0, dt_out=5.0):
        
        sol = np.zeros((n_steps,init.size))
        sol[0] = init
        
        state = sol[0]
        
        j_out = int(dt_out/dt)
        j_max = sol.shape[0]*j_out

        
#         for j in range(1,sol.shape[0]*j_out):
            
#             f1 = self.derivs(0,state)
#             f2 = self.derivs(0,state + dt*f1)
            
#             state = state + 0.5*dt*(f1+f2)
            
#             if j%j_out == 0:
#                 sol[j//j_out] = state

        for j in range(1,sol.shape[0]*j_out):
            
            f1 = self.derivs(0,state)
            f2 = self.derivs(0,state + 0.5*dt*f1)
            f3 = self.derivs(0,state + 0.5*dt*f2)
            f4 = self.derivs(0,state + dt*f3)
            
            state = state + dt*(f1 + 2.*f2 + 2.*f3 + f4)/6.
            
            if j%j_out == 0:
                sol[j//j_out] = state
                
        return sol
