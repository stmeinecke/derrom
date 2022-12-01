import numpy as np

from scipy.optimize import root as scpy_root
from scipy.integrate import solve_ivp as scpy_solve_ivp

import math
import ELPH


class PHELPH(ELPH.ELPH):
    
    def __init__(self, kmax=2.0, n_kmax=80):
        super().__init__(kmax,n_kmax)
        
        self.tau_photon = 1000
        self.g_photon = 0.01
        self.E_photon = 0.02
        self.linewidth = 0.005
        
        self.setup_laser()
  
    
    def derivs(self,t,y):
        
        right_ele = y[:self.n_kmax]
        right_phon = y[self.n_kmax:5*self.n_kmax].reshape(4,self.n_kmax)
        I = y[5*self.n_kmax]
        

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


        left_ele = temp_ele_in + temp_ele_out

        left_phon = temp_phon_abs + temp_phon_em
        left_phon = left_phon.flatten()
       
        
        ### laser stuff ###
        dI = -I/self.tau_photon + I * np.sum(self.I_gain_helper_vec * (2.*right_ele - 1.0))
        dI += 1e-9
        
        left_ele += -self.g_photon * I * self.lineshape_vec * (2.*right_ele - 1.0)
        ### laser stuff ###
        
        result = np.concatenate((left_ele,left_phon,[dI]))

        return result
    
    
    def setup_laser(self):        
        
        def lineshape(delta_E, width):
            return (1/(np.pi*width))/(np.cosh(delta_E/width))
        
        self.lineshape_vec = lineshape(self.E_el_vec - self.E_photon, self.linewidth)
        
        self.I_gain_helper_vec = self.g_photon * self.DOS_vec * self.lineshape_vec
        
    def get_net_photon_gain(self,el_state):
        return -1.0/self.tau_photon + np.sum(self.I_gain_helper_vec * (2.*el_state - 1.0))
    
    
    def get_electron_scattering_terms(self,state):
        tmp_g = self.g_photon
        self.g_photon = 0.0
        
        scat_terms =  self.derivs(0,state)[:self.n_kmax]
        
        self.g_photon = tmp_g
        
        return scat_terms
    
    
    def get_init_cond_gauss(self,max_pos=0.1, width=0.02, density = 0.05, I_0 = 1e-6):
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
        
        initial_state = np.concatenate((initial_state,[I_0]))
        
        
        return initial_state
