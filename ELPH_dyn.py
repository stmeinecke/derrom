import numpy as np

from scipy.optimize import root as scpy_root
from scipy.integrate import solve_ivp as scpy_solve_ivp

import math
import time


quiet = True

#time_start = time.time()

pi = np.pi
kB = 8.61745*0.00001
m0 = 5.6856800
mP = 10439.60413
hbar = 0.658212196
if not quiet:
  print("Naturkonstanten erstellt.")
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
if not quiet:
  print("Materialparameter erstellt.")


#Winkel
phimax = 2.*pi
n_phimax = 20
dphi = phimax/n_phimax
#print("Hyperparameter erstellt.")




def get_k(dk,n_k):
  return (n_k+1)*dk
  
def get_phi(dphi,n_phi):
  return n_phi*dphi


def electron_dispersion(k,phi):
  help = hbar*hbar*k*k/2./m
  return help

def phonon_dispersion(k,phi,alpha):
  if alpha==0:
    sau = EAprime
  if alpha==1:
    sau = ETO
  if alpha==2:
    sau=ca*k
  if alpha==3:
    sau=ca*k
  return sau

def phonon_coupling(k,phi,alpha):
  if alpha==0:
    sau = np.sqrt(Vol*hbar*hbar/2.0/Mr/phonon_dispersion(k,phi,0))*Do
  if alpha==1:
    sau = np.sqrt(Vol*hbar*hbar/2.0/Mr/phonon_dispersion(k,phi,1))*Do
  if alpha==2:
    sau = np.sqrt(Vol*hbar*hbar/2.0/Mr/phonon_dispersion(k,phi,2))*Da*k
  if alpha==3:
    sau = np.sqrt(Vol*hbar*hbar/2.0/Mr/phonon_dispersion(k,phi,3))*Da*k
  return sau



def phonon_occupation(k,phi,alpha,T):
  help = 1/(np.exp(phonon_dispersion(k,phi,alpha)/kB/T)-1.)
  return help


def build_boltzmann_mats(kmax, n_kmax):
  dk = kmax/n_kmax
  
  in_scattering_matrix_em = np.full((n_kmax,n_kmax,4,n_kmax), 0., dtype=np.float64)
  out_scattering_matrix_em = np.full((n_kmax,n_kmax,4,n_kmax), 0., dtype=np.float64)
  in_scattering_matrix_abs = np.full((n_kmax,n_kmax,4,n_kmax), 0., dtype=np.float64)
  out_scattering_matrix_abs = np.full((n_kmax,n_kmax,4,n_kmax), 0., dtype=np.float64)

  absorption_matrix = np.full((n_kmax,n_kmax,4,n_kmax), 0., dtype=np.float64)
  emission_matrix = np.full((n_kmax,n_kmax,4,n_kmax), 0., dtype=np.float64)


  for n_k in range(n_kmax):
    for n_phi in range(n_phimax):
      for beta in range(4):
        kk = get_k(dk,n_k)
        phi_diff = get_phi(dphi,n_phi)

        if True:
          def delta_fun_phon(x):
            helper1 = np.sqrt(x*x+get_k(dk,n_k)*get_k(dk,n_k) - 2.*x*get_k(dk,n_k)*np.cos(phi_diff))
            result = electron_dispersion(get_k(dk,n_k),0) - electron_dispersion(x,0)+phonon_dispersion(helper1,0,beta)
            return result
          sol = scpy_root(delta_fun_phon, get_k(dk,n_k+1), args=(), method='hybr', jac=None, tol=None, callback=None, options=None)
          if sol.x>kk:
            n_null_elec = math.floor(sol.x/dk) - 1
            interpolator_electrons = np.full(2,0.)
            interpolator_electrons[0] =  np.absolute(delta_fun_phon(get_k(dk,n_null_elec+1)))/(np.absolute(delta_fun_phon(get_k(dk,n_null_elec)))+ np.absolute(delta_fun_phon(get_k(dk,n_null_elec+1))))
            interpolator_electrons[1] =  np.absolute(delta_fun_phon(get_k(dk,n_null_elec)))/(np.absolute(delta_fun_phon(get_k(dk,n_null_elec)))+ np.absolute(delta_fun_phon(get_k(dk,n_null_elec+1))))
            for n_kt in range (n_null_elec,n_null_elec+2):
              kkt = get_k(dk,n_kt)
              

              if (kkt*kkt+kk*kk - 2.*kkt*kk*np.cos(phi_diff) > 0):
                helper = np.sqrt(kkt*kkt+kk*kk - 2.*kkt*kk*np.cos(phi_diff))
                if helper>0 and n_kt<n_kmax:
                  interpolator_phonons = np.full(2,0.)
                  n_null_phon = int(helper/dk)-1
                  help_diff_lower = np.absolute(get_k(dk,n_null_phon) - helper)
                  help_diff_upper = np.absolute(get_k(dk,n_null_phon+1) - helper)
                  interpolator_phonons[0] = help_diff_upper/(help_diff_lower + help_diff_upper)
                  interpolator_phonons[1] = help_diff_lower/(help_diff_lower + help_diff_upper)
                  for n_q in range(n_null_phon,n_null_phon+2):
                    qq = get_k(dk,n_q)
                    if n_q > -1 and n_q < n_kmax:
                      part1 = phonon_coupling(qq,0,beta)*phonon_coupling(qq,0,beta)
                      if beta<2:
                        part2 = 1./np.absolute(hbar*hbar*get_k(dk,n_kt)/m)/dk
                      if beta>1:
                        part2 =  1./np.absolute(hbar*hbar*get_k(dk,n_kt)/m-hbar*ca*(kkt-kk*np.cos(phi_diff))/qq)/dk
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
  



def get_init_cond_gauss(kmax, n_kmax, max_pos=0.2, width=0.05, density = 0.1):
  dk = kmax/n_kmax
  
  initial_ele = np.full(n_kmax,0.)
  initial_phon = np.full((4,n_kmax),0.)
#   helper_max = max_pos
#   helper_width = width
  for n_k in range(n_kmax):
    helper_expo = electron_dispersion(get_k(dk,n_k),0) - max_pos
    helper = np.exp(-helper_expo*helper_expo/width/width/2.)
    initial_ele[n_k] = helper
  helper = 0.
  for n_k in range(n_kmax):
    helper += dk*get_k(dk,n_k)/2./pi*initial_ele[n_k]
  initial_ele = initial_ele/helper*density
  for n_k in range(n_kmax):
    for a in range(4):
      initial_phon[a][n_k] = phonon_occupation(get_k(dk,n_k),0.,a,T_cryo)

  #print(np.shape(initial_phon))
  initial_ele = np.reshape(initial_ele,(1,n_kmax))
  #print(np.shape(initial_ele))

  initial_state = np.concatenate((initial_ele,initial_phon),axis = 0 )
  initial_state = np.reshape(initial_state,5*n_kmax)

  return initial_state


# def get_init_cond_gauss(kmax, n_kmax, max_pos=0.2, width=0.05):
#   dk = kmax/n_kmax
  
#   print(max_pos)
#   print(width)
  
#   initial_ele = np.full(n_kmax,0.)
#   initial_phon = np.full((4,n_kmax),0.)
#   for n_k in range(n_kmax):
#     helper_expo = electron_dispersion(get_k(dk,n_k),0) - max_pos
#     helper = np.exp(-helper_expo*helper_expo/width/width/2.)
#     initial_ele[n_k] = helper
#   helper = 0.
#   for n_k in range(n_kmax):
#     helper += dk*get_k(dk,n_k)/2./pi*initial_ele[n_k]
#   initial_ele = initial_ele/helper*density

#   for n_k in range(n_kmax):
#     for a in range(4):
#       initial_phon[a][n_k] = phonon_occupation(get_k(dk,n_k),0.,a,T_cryo)

#   # print(np.shape(initial_phon))
#   initial_ele = np.reshape(initial_ele,(1,n_kmax))
#   # print(np.shape(initial_ele))

#   initial_state = np.concatenate((initial_ele,initial_phon),axis = 0 )
#   initial_state = np.reshape(initial_state,5*n_kmax)

#   return initial_state



def get_el_dynamics(init_cond, kmax= 2.0, n_kmax = 20, tmax = 10000.0, n_tmax = 2000):
    
  in_scattering_matrix_em, out_scattering_matrix_em, in_scattering_matrix_abs, out_scattering_matrix_abs, absorption_matrix, emission_matrix, einer = build_boltzmann_mats(kmax,n_kmax)
    
    
  def boltzmann_equation(t,y):
    y = np.reshape(y,(5,n_kmax))
    right_ele = y[0]
    right_phon = y[1:]

    temp_phon_abs = np.full((4,n_kmax),0.)
    temp_phon_em = np.full((4,n_kmax),0.)
    temp_ele_in = np.full(n_kmax,0.)
    temp_ele_out = np.full(n_kmax,0.)

    block_ele = einer - right_ele
    stim_phon = einer + right_phon

    help_phonon_weg = np.tensordot(in_scattering_matrix_em,stim_phon,axes = ([2,3],[0,1])) + np.tensordot(in_scattering_matrix_abs,right_phon,axes = ([2,3],[0,1]))
    help_source_weg = np.tensordot(help_phonon_weg,right_ele,axes = (1,0))
    temp_ele_in = np.multiply(help_source_weg,block_ele)

    help_phonon_weg = np.tensordot(out_scattering_matrix_em,stim_phon,axes = ([2,3],[0,1])) + np.tensordot(out_scattering_matrix_abs,right_phon,axes = ([2,3],[0,1]))
    help_source_weg = np.tensordot(help_phonon_weg,block_ele,axes = (1,0))
    temp_ele_out = np.multiply(help_source_weg,right_ele)



    block_weg = np.tensordot(absorption_matrix,block_ele,axes = (1,0))
    temp_phon_abs = np.tensordot(block_weg,right_ele,axes = (0,0))
    temp_phon_abs = np.multiply(temp_phon_abs,right_phon)

    block_weg = np.tensordot(emission_matrix,right_ele,axes = (1,0))
    block_weg_2 = np.tensordot(block_weg,block_ele,axes = (0,0))
    temp_phon_em = np.multiply(block_weg_2,stim_phon)

    left_phon = np.full((4,n_kmax),0.)
    left_phon = temp_phon_abs + temp_phon_em

    left_ele = np.full(n_kmax,0.)
    left_ele = temp_ele_in + temp_ele_out

    left_ele = np.reshape(left_ele,(1,n_kmax))

    result = np.concatenate((left_ele,left_phon))
    result = np.reshape(result,5*n_kmax)
    return result


  t_values = np.linspace(0.0, tmax, n_tmax)
  sol = scpy_solve_ivp(boltzmann_equation, [t_values[0],t_values[-1]], init_cond, t_eval=t_values)
  y_values = np.reshape(np.asarray(sol.y).T,(n_tmax,5,n_kmax))

  return y_values[:,0,:].T
  


#kmax = 2.
#n_kmax = 30
#dk = kmax/n_kmax


#init = get_init_cond_gauss(kmax = kmax, n_kmax = n_kmax)
#nkdyn = get_el_dynamics(init, n_kmax = n_kmax)

#time_ende = time.time()
#print("Laufzeit des Codes:"+str(time_ende-time_start))


#import matplotlib.pyplot as plt
#plt.imshow(nkdyn, aspect='auto')
#plt.colorbar()
#plt.show()
  


#energy = np.full((n_tmax,5),0.)
#occupation = np.full((n_tmax,5),0.)
#for n_t in range(n_tmax):
  #for a in range(4):
    #for n_k in range(n_kmax):
      #energy[n_t][1+a] += y_values[n_t][1+a][n_k]*phonon_dispersion(get_k(dk,n_k),0.,a)*get_k(dk,n_k)*dk/2./pi
      #occupation[n_t][1+a] += y_values[n_t][1+a][n_k]*get_k(dk,n_k)*dk/2./pi
  #for n_k in range(n_kmax):
    #energy[n_t][0] +=get_k(dk,n_k)*dk/2./pi* y_values[n_t][0][n_k]*electron_dispersion(get_k(dk,n_k),0.)
    #occupation[n_t][0] +=get_k(dk,n_k)*dk/2./pi* y_values[n_t][0][n_k]
#print(np.shape(energy))

#path = "data"+str(n_kmax)
#if not os.path.exists(path):
  #os.makedirs(path)


#np.save(path+"/time_traces",np.array(y_values))
#np.save(path+"/time_grid",np.array(t_values))
#np.save(path+"/energy",np.array(energy))
#np.save(path+"/occupation",np.array(occupation))


