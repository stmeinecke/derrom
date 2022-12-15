import numpy as np
import matplotlib.pyplot as plt

class fermi_fit:
    def __init__(self, DOS, Energies, kB):
        self.DOS = DOS
        self.Energies = Energies
        self.kB = kB
        
        self.MAXERROR = 1e-10 #5E-12
        self.STEP_LIMIT = 100  #// maximale Anzahl an Schritten für eine Berechnung
        self.dmu = 0.5E-3        #// differentieller Schritt zur Berechnung der Ableitungen d/dmu,d/dT
        self.dT = 1.0
        
        self.mstep_max_init = 0.1  #// maximale Schrittweite der Iterationen
        self.Tstep_max_init = 100.
        
        self.rel_mstep_max = 0.5  #// maximale Schrittweite der Iterationen
        self.rel_Tstep_max = 0.5
        
        self.mstep_max = 0.01  #// maximale Schrittweite der Iterationen
        self.Tstep_max = 100.
        
    def fermi(self,E,mu=1,T=1):
        return 1./(np.exp((E-mu)/(self.kB*T)) + 1)
    
    #berechnet chem Pot. (eV) und Temperatur der Fermiverteilung
    #aus vorgegebener Ladungsträgerdichte + Gesamtenergie, Ausgabe in *mu, *temp
    #en_Bulk/QW/QD sind entweder Elektonen- oder Loch-Energien (werden durch Aufruf der jeweiligen calc_fermi_CP_e/h-Funktion gesetzt)
    def fit_mu_T(self, state, mu_start, T_start, FERMI_DEBUG = False, WARNINGS = False):
        
        self.mstep_max = self.mstep_max_init  #// maximale Schrittweite der Iterationen
        self.Tstep_max = self.Tstep_max_init
        
        dens = 0.
        en = 0.

        for i in range(state.size):
            #Gesamtdichte und -energie aufsummieren
            f = self.DOS[i] * state[i]
            dens += f   
            en += f * self.Energies[i]

    #     print(dens,en)


        m=mu_start
        T=T_start;

        m2 = m;
        T2 = T;

        d = 0
        d_dmu = 0
        d_dT = 0
        dDdmu = 0
        dDdT = 0;
        E = 0
        E_dmu = 0
        E_dT = 0
        dEdmu = 0
        dEdT = 0;
        f=0
        mstep=0
        Tstep=0

        Jacobi_Det = 0;
        deltaD = 0
        deltaE = 0
        for steps in range(self.STEP_LIMIT):
            d = d_dmu = d_dT = 0
            E = E_dmu = E_dT = 0
            #// resultierende Dichte + Energie aus Fermifunktion aufsummieren
            #// m2 = m+delta_m,
            #// Berechnung der Jacobi-Matrix von (Dichte, Energie) nach (mu, T)
            m2 = m+self.dmu
            T2 = T+self.dT

            for i in range(state.size):

                #Gesamtdichte und -energie aufsummieren
                f = self.DOS[i] * self.fermi(self.Energies[i], m, T)
                d += f   
                E += f * self.Energies[i]

                #Werte bei mu = mu+dmu
                f = self.DOS[i] * self.fermi(self.Energies[i], m2, T)
                d_dmu += f        
                E_dmu += f * self.Energies[i]

                #Werte bei T = T+dT
                f = self.DOS[i] * self.fermi(self.Energies[i], m, T2)
                d_dT += f        
                E_dT += f * self.Energies[i]

            # Genau genug -> fertig
            error = np.maximum( np.abs(d/dens-1.), np.abs(E/en-1.));
            if ( error <= self.MAXERROR ): 
    #             print('error: ', error)
                break

            #//Jacobi-Matrix-Elemente berechnen
            dDdmu = (d_dmu-d)/self.dmu
            dEdmu = (E_dmu-E)/self.dmu
            dDdT = (d_dT-d)/self.dT
            dEdT = (E_dT-E)/self.dT
            Jacobi_Det = dDdT*dEdmu-dDdmu*dEdT
            if (Jacobi_Det == 0 and WARNINGS == True):
                #plt.plot(state)
                #plt.show()
                print("calc_fermi_CC_eh: Jacobi_Det wird null!!!")
                print("     d=",d,"  mu=",m,"  E=",E,"  T=",T)
                break

            #// Berechne neues mu,T durch Lösung von A.(deltaMu, deltaT) = (deltaD, deltaE). => Invertierung der Jacobimatrix gibt dmu,dT
            deltaD = dens-d
            deltaE = en-E
            mstep = (deltaE*dDdT-deltaD*dEdT)/Jacobi_Det
            Tstep = (deltaD*dEdmu-deltaE*dDdmu)/Jacobi_Det
            
            #self.mstep_max = max(np.abs(m * self.rel_mstep_max),self.mstep_max_init)  #// maximale Schrittweite der Iterationen
            #self.Tstep_max = max(T * self.rel_Tstep_max, self.Tstep_max_init)
            
            if(mstep < -self.mstep_max):
                mstep = -self.mstep_max
            elif (mstep > self.mstep_max):
                mstep = self.mstep_max;
            if(Tstep < -self.Tstep_max):
                Tstep = -self.Tstep_max
            elif (Tstep > self.Tstep_max):
                Tstep = self.Tstep_max

            m += mstep;
            T += Tstep;

            if FERMI_DEBUG:
                print("calc_fermi_CC_eh: dens=",d,", en=",en)
                print("calc_fermi_CC_eh: mu=",m,", T=",T," nach ",steps," Schritten")

        if (steps == self.STEP_LIMIT-1 and WARNINGS == True):
            plt.plot(state)
            plt.show()
            print("calc_fermi_CC_eh braucht mehr als STEP_LIMIT Schritte!")
            print("calc_fermi_CC_eh: mu_start=",mu_start,",T_start=",T_start)
            print("calc_fermi_CC_eh: dens=",dens,", en=",en)
            print("calc_fermi_CC_eh: d=",d,", E=",E)
            print("calc_fermi_CC_eh: mu=",m,", T=",T)
            print("calc_fermi_CC_eh: error=",error," nach ",steps," Schritten")


        return m,T,steps


    #berechnet chem Pot. (eV) aus vorgegebener Ladungsträgerdichte + Temperatur, Ausgabe in *mu
    #en_Bulk/QW/QD sind entweder Elektonen- oder Loch-Energien (werden durch Aufruf der jeweiligen calc_fermi_CP_e/h-Funktion gesetzt)
    def fit_mu(self, state, mu_start, T, FERMI_DEBUG = False, WARNINGS = False):
        
        self.mstep_max = self.mstep_max_init  #// maximale Schrittweite der Iterationen
        
        dens = 0.
        en = 0.
        for i in range(state.size):
            #Gesamtdichte und -energie aufsummieren
            f = self.DOS[i] * state[i]
            dens += f   
            en += f * self.Energies[i]
        
        steps = 0;

        m=mu_start
        m2 = m+self.dmu

        #resultierende Dichte aus Fermifunktion aufsummieren
        # m2 = m+delta_m,
        # dDdmu = (Dens(m2)-Dens(m))/delta_m    ist Ableitung d(Dens)/d(mu) an Stelle m
        d = 0
        d2 = 0
        for i in range(state.size):
            d += self.DOS[i] * self.fermi(self.Energies[i], m, T)
            d2 += self.DOS[i] * self.fermi(self.Energies[i], m2, T)
        

        #// Ableitung dDichte / dMu
        dDdmu = (d2-d)/self.dmu;
        if (dDdmu == 0 and WARNINGS == True):
            print("calc_fermi_CP_eh: dDdmu wird null!!!")
            print("     Ziel: ", dens)
            print("     d=",d,"  mu=",m)

        # Berechne neues m über Ableitung.
        mstep = (dens-d)/dDdmu;
        
        self.mstep_max = max(np.abs(m * self.rel_mstep_max),self.mstep_max_init)  #// maximale Schrittweite der Iterationen
        
        if(mstep < -self.mstep_max):
            mstep = -self.mstep_max
        elif (mstep > self.mstep_max):
            mstep = self.mstep_max;

        # Berechnung bis gewünschte Genauigkeit erreicht ist
        while( np.abs(d/dens-1) > self.MAXERROR ):
            
            #print("target dens: " , dens , "  current dens: " , d , "  current mu: " , m)

            m += mstep;

            d = 0;
            for i in range(state.size):
                d += self.DOS[i] * self.fermi(self.Energies[i], m, T)
            
            
            if (d > dens and mstep > 0):
                mstep *= -0.5;
            elif (d < dens and mstep < 0):
                mstep *= -0.5;

            steps += 1
            
            if(steps>self.STEP_LIMIT):
                if WARNINGS == True:
                    print("fit_mu braucht mehr als STEP_LIMIT Schritte!")
                break


        if FERMI_DEBUG:
            print("calc_fermi_CP_eh: mu=" , m , " nach " , steps , " Schritten")
        
        return m,steps

