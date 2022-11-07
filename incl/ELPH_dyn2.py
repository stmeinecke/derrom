import numpy as np

from scipy.optimize import root as scpy_root
from scipy.integrate import solve_ivp as scpy_solve_ivp

import math

pi = np.pi
kB = 8.61745*0.00001
m0 = 5.6856800
mP = 10439.60413
hbar = 0.658212196

m = 0.5*m0
a0 = 0.314
ca = 4.1/1000.
EAprime = 0.031
ETO = 0.036
Mr = 253*mP
Vol = np.sqrt(3.0)/2.0*a0*a0
Da = 3.4
Do = 5.2*10
T_cryo = 300
# density = 0.1

#Winkel
phimax = 2.*pi
n_phimax = 20
dphi = phimax/n_phimax


class ELPH:
    
    def __init__(self, kmax=4.0, n_kmax=80):
        self.kmax = kmax
        self.n_kmax = n_kmax
        
        self.dk = kmax/n_kmax
        
        self.in_scattering_matrix_em, self.out_scattering_matrix_em, self.in_scattering_matrix_abs, self.out_scattering_matrix_abs, self.absorption_matrix, self.emission_matrix, self.einer = self.build_boltzmann_mats(self.kmax, self.n_kmax)
    
    def get_k(self,dk,n_k):
        return (n_k+1.)*dk
  
    
    def get_phi(self,dphi,n_phi):
        return n_phi*dphi


    def electron_dispersion(self,k,phi=0):
        help = hbar*hbar*k*k/2./m
        return help
    
    def electron_DOS(self,k):
        return k*self.dk/2./np.pi
    
    def get_electron_density(self,state):
        dens = 0
        for j in range(state.size):
            dens += state[j] * self.electron_DOS(self.get_k(dk,j))
        return dens

    
    def phonon_dispersion(self,k,phi,alpha):
        if alpha==0:
            sau = EAprime
        if alpha==1:
            sau = ETO
        if alpha==2:
            sau=ca*k
        if alpha==3:
            sau=ca*k
        return sau

    
    def phonon_coupling(self,k,phi,alpha):
        if alpha==0:
            sau = np.sqrt(Vol*hbar*hbar/2.0/Mr/self.phonon_dispersion(k,phi,0))*Do
        if alpha==1:
            sau = np.sqrt(Vol*hbar*hbar/2.0/Mr/self.phonon_dispersion(k,phi,1))*Do
        if alpha==2:
            sau = np.sqrt(Vol*hbar*hbar/2.0/Mr/self.phonon_dispersion(k,phi,2))*Da*k
        if alpha==3:
            sau = np.sqrt(Vol*hbar*hbar/2.0/Mr/self.phonon_dispersion(k,phi,3))*Da*k
        return sau
    
    
    def phonon_occupation(self,k,phi,alpha,T):
        help = 1/(np.exp(self.phonon_dispersion(k,phi,alpha)/kB/T)-1.)
        return help


    def build_boltzmann_mats(self,kmax, n_kmax):
        dk = self.dk

        in_scattering_matrix_em = np.full((n_kmax,n_kmax,4,n_kmax), 0., dtype=np.float64)
        out_scattering_matrix_em = np.full((n_kmax,n_kmax,4,n_kmax), 0., dtype=np.float64)
        in_scattering_matrix_abs = np.full((n_kmax,n_kmax,4,n_kmax), 0., dtype=np.float64)
        out_scattering_matrix_abs = np.full((n_kmax,n_kmax,4,n_kmax), 0., dtype=np.float64)

        absorption_matrix = np.full((n_kmax,n_kmax,4,n_kmax), 0., dtype=np.float64)
        emission_matrix = np.full((n_kmax,n_kmax,4,n_kmax), 0., dtype=np.float64)


        for n_k in range(n_kmax):
            for n_phi in range(n_phimax):
                for beta in range(4):
                    kk = self.get_k(dk,n_k)
                    phi_diff = self.get_phi(dphi,n_phi)

                    if True:
                        def delta_fun_phon(x):
                            helper1 = np.sqrt(x*x+self.get_k(dk,n_k)*self.get_k(dk,n_k) - 2.*x*self.get_k(dk,n_k)*np.cos(phi_diff) + 1e-15)
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
                                                    part2 = 1./np.absolute(hbar*hbar*self.get_k(dk,n_kt)/m)/dk
                                                if beta>1:
                                                    part2 =  1./np.absolute(hbar*hbar*self.get_k(dk,n_kt)/m-hbar*ca*(kkt-kk*np.cos(phi_diff))/qq)/dk
                                                #absorption
                                                out_scattering_matrix_abs[n_k][n_kt][beta][n_q] +=- 2.*pi/hbar*dk*kkt*dphi/4./pi/pi*part1*part2*interpolator_electrons[n_kt-n_null_elec]*interpolator_phonons[n_q-n_null_phon]#(1-fkt) fk n
                                                in_scattering_matrix_abs[n_kt][n_k][beta][n_q] += 2.*pi/hbar*dk*kk*dphi/4./pi/pi*part1*part2*interpolator_electrons[n_kt-n_null_elec]*interpolator_phonons[n_q-n_null_phon]#(1-fk) fkt n
                                                absorption_matrix[n_k][n_kt][beta][n_q] +=  - 2.*pi/hbar*dk*kk*dphi/4./pi/pi*part1*part2*interpolator_electrons[n_kt-n_null_elec]*interpolator_phonons[n_q-n_null_phon]
                                                #emission
                                                in_scattering_matrix_em[n_k][n_kt][beta][n_q] +=  2.*pi/hbar*dk*kkt*dphi/4./pi/pi*part1*part2*interpolator_electrons[n_kt-n_null_elec]*interpolator_phonons[n_q-n_null_phon]#(1-fkt) fk (1+n)
                                                out_scattering_matrix_em[n_kt][n_k][beta][n_q] += - 2.*pi/hbar*dk*kk*dphi/4./pi/pi*part1*part2*interpolator_electrons[n_kt-n_null_elec]*interpolator_phonons[n_q-n_null_phon]#(1-fk) fkt (1+n)
                                                emission_matrix[n_k][n_kt][beta][n_q] += 2.*pi/hbar*dk*kk*dphi/4./pi/pi*part1*part2*interpolator_electrons[n_kt-n_null_elec]*interpolator_phonons[n_q-n_null_phon]

        einer = np.full(n_kmax,1.)

        return [in_scattering_matrix_em, out_scattering_matrix_em, in_scattering_matrix_abs, out_scattering_matrix_abs, absorption_matrix, emission_matrix, einer]

    
    def boltzmann_equation(self,t,y):
    
        y = np.reshape(y,(5,self.n_kmax))
        right_ele = y[0]
        right_phon = y[1:]

        temp_phon_abs = np.full((4,self.n_kmax),0.)
        temp_phon_em = np.full((4,self.n_kmax),0.)
        temp_ele_in = np.full(self.n_kmax,0.)
        temp_ele_out = np.full(self.n_kmax,0.)

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

        left_phon = np.full((4,self.n_kmax),0.)
        left_phon = temp_phon_abs + temp_phon_em

        left_ele = np.full(self.n_kmax,0.)
        left_ele = temp_ele_in + temp_ele_out

        left_ele = np.reshape(left_ele,(1,self.n_kmax))

        result = np.concatenate((left_ele,left_phon))
        result = np.reshape(result,5*self.n_kmax)
        return result
    
    
    def get_full_trajectory(self, init_cond, tmax = 2000.0, n_tmax = 400):

        t_values = np.linspace(0.0, tmax, n_tmax)
        sol = scpy_solve_ivp(self.boltzmann_equation, [t_values[0],t_values[-1]], init_cond, t_eval=t_values)
        y_values = np.reshape(np.asarray(sol.y).T,(n_tmax,5,self.n_kmax))

        return y_values[:,:,:]
    
    def get_electron_trajectory(self, init_cond, tmax = 2000.0, n_tmax = 400):
        return self.get_full_trajectory(init_cond, tmax = tmax, n_tmax = n_tmax)[:,0,:]
    
    def get_electron_derivs(self,state):
        return self.boltzmann_equation(0,state)[:self.n_kmax]
    
    
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
            helper += dk*self.get_k(dk,n_k)/2./pi*initial_ele[n_k]
        initial_ele = initial_ele/helper*density
        for n_k in range(n_kmax):
            for a in range(4):
                initial_phon[a][n_k] = self.phonon_occupation(self.get_k(dk,n_k),0.,a,T_cryo)

        #print(np.shape(initial_phon))
        initial_ele = np.reshape(initial_ele,(1,n_kmax))
        #print(np.shape(initial_ele))

        initial_state = np.concatenate((initial_ele,initial_phon),axis = 0 )
        initial_state = np.reshape(initial_state,5*n_kmax)

        return initial_state
