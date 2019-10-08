from __future__ import division, absolute_import, print_function
from CoolProp.CoolProp import PropsSI
import CoolProp as CP
import numpy as np
from scipy.optimize import brentq

from math import sqrt, pi, pow, atan, log
from convert_units import cms2gpm, psi2kPa, C2K, in2m

class ViperClass():

    """
    Viper model converted from EES
    """

    def __init__(self, **kwargs):
        # Load the parameters passed in
        # using the dictionary
        self.__dict__.update(kwargs)

    def Update(self, **kwargs):
        # Update the parameters passed in
        # using the dictionary
        self.__dict__.update(kwargs)
        
    def natlconv(self, g,beta,nu,Pr,T_s,T_amb,L,g_fac):
        # Natural convection function
        
        Gr_L = (g*beta*(abs(T_s - T_amb))*L**3)/nu**2 #Grashof number
        Ra_L = Gr_L*Pr
         
        if (g_fac == 1): #vertical surface
            Nu_L = (0.825 + (0.387*(Ra_L**(1/6))/(1 + (0.492/Pr)**(9/16))**(8/27)))**2         
        elif (g_fac == 2): #horizontal surface, top
            Nu_L = 0.25*(Ra_L**0.25)
        elif (g_fac == 3 and Ra_L>10^4): #horizontal surface, bottom
            Nu_L = 0.54*(Ra_L**0.25)
        else:
            Nu_L = 0.15*Ra_L**(1/3)
        return Nu_L

    def phasesep(self, x_th,del_x_sep,m_dot):
        # Phase separation
        x_vap = 1 - del_x_sep
        x_liq = del_x_sep
        m_dot_vap = (x_th - del_x_sep)*m_dot
        m_dot_liq = (1 - x_th + del_x_sep)*m_dot
        return x_vap, m_dot_vap, x_liq, m_dot_liq
    
    def mv(self, n,c_v,m_dot,p,x):
        # Metering valve pressure drop calculation
        n_max = 9.5 # max # of metering valve turns  [-]
        n_frac = n/n_max
        self.AS.update(CP.PQ_INPUTS, p, x)
        v = 1/self.AS.rhomass() # [m^3/kg]
        V_dot = m_dot*v
        V_dot_IP = cms2gpm(V_dot) #convert 'm^3/sec' to 'gal/min' 
        cap = (87.17*(n_frac**2) + 14.821*n_frac - 0.6503)/100
        c_v_eff = c_v*cap
        delta_p_IP = c_v_eff*V_dot_IP
        delta_p = psi2kPa(delta_p_IP)*1000 #convert 'psi' to 'Pa'
        return delta_p
    
    def bv(self, c_v,m_dot,p,x):
        # Ball valve pressure drop calculation
        self.AS.update(CP.PQ_INPUTS, p, x)
        v = 1/self.AS.rhomass() # [m^3/kg]
        V_dot = m_dot*v
        V_dot_IP = cms2gpm(V_dot) #convert 'm^3/sec' to 'gal/min'
        delta_p_IP = c_v*V_dot_IP
        delta_p = psi2kPa(delta_p_IP)*1000 #convert 'psi' to 'Pa'
        return delta_p
    
     
    def Calculate(self): 
        
        # AbstractState
        AS = self.AS
         
        # inlet state
        if self.pin_r > AS.p_critical(): #Supercritical
            AS.update(CP.HmassP_INPUTS, self.hin_r, self.pin_r)
            self.sin_r = AS.smass() #[J/kg-K]
            self.Tin_r = AS.T() #[K]
        else: #other refrigerants  
            AS.update(CP.PQ_INPUTS, self.pin_r, 0.0)
            Tbubble_in = AS.T() #[K]
            h_l_in = AS.hmass() #[J/kg]
            s_l_in = AS.smass() #[J/kg-K]
            AS.update(CP.PQ_INPUTS, self.pin_r, 1.0)
            Tdew_in = AS.T() #[K]
            h_v_in = AS.hmass() #[J/kg]
            s_v_in = AS.smass() #[J/kg-K]
            
            self.xin_r = (self.hin_r-h_l_in)/(h_v_in-h_l_in)
            if (self.xin_r>0.999):
                print ("ExpDev :: Upstream state in the expansion device is superheated")
                raise
            if (self.xin_r>0.0 and self.xin_r<1.0): #two-phase state at the inlet
                self.sin_r = self.xin_r*s_v_in+(1-self.xin_r)*s_l_in #[J/kg-K]
                self.Tin_r = self.xin_r*Tdew_in+(1-self.xin_r)*Tbubble_in #[K]
            else: #liquid state at the inlet
                AS.update(CP.HmassP_INPUTS, self.hin_r, self.pin_r)
                self.sin_r = AS.smass() #[J/kg-K]
                self.Tin_r = AS.T() #[K]
        
        # Refrigerant conditions"
        h_in = self.hin_r
        s_in = self.sin_r
        p_in = self.pin_r
        T_in = self.Tin_r
        p_out = self.pout_r
        AS.update(CP.HmassP_INPUTS, h_in, self.pin_r)
        rho_in = AS.rhomass() # [kg/m^3]
        
        AS.update(CP.PSmass_INPUTS, p_out, s_in)
        h_out_s = AS.hmass() #[J/kg]
        
        
        A_tube = pi*self.D_tube**2/4
        vel_in = self.m_dot/(rho_in*A_tube)
        T_cond = Tbubble_in
        T_sc = T_cond - T_in
         
        W_dot_is = self.m_dot*(h_in - h_out_s)
         
        #Ambient conditions - air properties"
        AS_air = CP.AbstractState(self.Backend, 'air')
        T_amb = C2K(35)   # [C]
        p_atm = 101325 # [Pa]
        AS_air.update(CP.PT_INPUTS, p_atm, T_amb)
        k_air = AS_air.conductivity() # [W/m-K]
        mu_air = AS_air.viscosity() # [Pa-s]
        rho_air = AS_air.rhomass() #[kg/m^3]
        nu = mu_air/rho_air # Kinematic viscosity
        beta = 1/T_amb
        Pr = AS_air.Prandtl() # Prandtl number
        
        AS.update(CP.PQ_INPUTS, p_out, 0.0)
        T_s = AS.T() #[K]
        g = 9.81     # [m/s^2]
        delta_p_total = p_in - p_out 
         
        #===================== Natural convection =====================
        #Sides, vertical, cold
        A_side = self.height*self.diameter
        P_side = 2*self.height + 2*self.diameter
        L_side = A_side/P_side
        g_fac_side = 1
        Nusselt_L_side = self.natlconv(g,beta,nu,Pr,T_s,T_amb,L_side,g_fac_side)
        h_side = Nusselt_L_side*k_air/L_side
        Q_dot_side = h_side*A_side*(T_s - T_amb)
         
        #Top, horizontal, cold
        A_top = (pi*(self.diameter**2))/4
        P_top = pi*self.diameter
        L_top = A_top/P_top
        g_fac_top = 2 
        Nusselt_L_top = self.natlconv(g,beta,nu,Pr,T_s,T_amb,L_top,g_fac_top)
        h_top = Nusselt_L_top*k_air/L_top
        Q_dot_top = h_top*A_top*(T_s - T_amb)
         
        #Bottom, horizontal, cold
        A_bottom = A_top
        L_bottom = L_top
        g_fac_bottom = 3
        Nusselt_L_bottom = self.natlconv(g,beta,nu,Pr,T_s,T_amb,L_bottom,g_fac_bottom)
        h_bottom = Nusselt_L_bottom*k_air/L_bottom
        Q_dot_bottom = h_bottom*A_bottom*(T_s - T_amb)
         
        Q_dot_loss_tot = Q_dot_bottom + Q_dot_top + Q_dot_side
        self.Q_amb = Q_dot_loss_tot
        
        def Objective(x):

            #Phase separation losses
            x_vap,m_dot_vap,x_liq,m_dot_liq = self.phasesep(x,self.del_x_sep,self.m_dot)
             
            #===================== Valve losses =====================
            #metering valve
            c_v_mv = 0.73    #[psi/(gal/min)]
            n_MV = 2    # [-]
            #ball valve"
            c_v_bv = 4.4    # [psi/(gal/min)]
             
            #Vapor line valve losses
            delta_p_mv_vap = self.mv(n_MV,c_v_mv,m_dot_vap,p_out,x_vap)
            delta_p_bv_vap = self.bv(c_v_bv,m_dot_vap,p_out,x_vap)
             
            #Liquid line valve losses
            delta_p_bv_liq = self.bv(c_v_bv,m_dot_liq,p_out,x_liq)
    
            #Power losses calculation
            self.W_dot_mech = self.W_dot_elec/self.eta_gen 
            W_dot_fluid_tot = self.W_dot_mech/self.eta_mech
             
            #===================== Nozzle calculation =====================
            #Test nozzle conditions to achieve saturated liquid conditions
            p_t_1 = p_in
            T_t_1 = T_in
            p_t_2 = p_out + self.delta_p_Viper
            
            AS.update(CP.PQ_INPUTS, p_t_1, 0.0) 
            T_sat_test = AS.T() #[K]
            AS.update(CP.PT_INPUTS, p_t_1, T_t_1)
            h_t_1 = AS.hmass() #[J/kg]
            s_t_1 = AS.smass() #[J/kg-K]
            
            AS.update(CP.PSmass_INPUTS, p_t_2, s_t_1)
            h_t2s = AS.hmass() #[J/kg]
            h_t_2 = h_t_1 + self.eta_nozzle*(h_t2s - h_t_1)
            AS.update(CP.HmassP_INPUTS, h_t_2, p_t_2)
            T_t_2 = AS.T() #[K]
            s_t_2 = AS.smass() #[J/kg-K]
            self.x_target = AS.Q() #quality [-]
            
            self.delta_p_nozzle = p_t_1 - p_t_2
            self.E_dot_nozzle = self.m_dot*(h_t_1 - h_t_2)  #Power recovered by nozzle [W]
             
            #===================== Turbine losses =====================
            self.W_dot_fluid = W_dot_fluid_tot + Q_dot_loss_tot
            self.eta_flow = self.W_dot_fluid/self.E_dot_nozzle
            
            return self.x_target - x
        
        brentq(Objective,0.00000000001,0.9999999999)
        
        para_calc = {
            'Quality': self.x_target,
            'T_sc': T_sc,
            'eta_flow': self.eta_flow,
            'delta_p_nozzle': self.delta_p_nozzle,
            'W_dot_is': W_dot_is,
            'E_dot_nozzle': self.E_dot_nozzle,
            'W_dot_fluid': self.W_dot_fluid,
            'W_dot_mech': self.W_dot_mech,
            'Q_dot_loss_tot': Q_dot_loss_tot,
            'delta_p_total' : delta_p_total
            }
        
        return para_calc

if __name__=='__main__':    
    Ref = 'R410A'
    Backend = 'HEOS'  # choose between: 'HEOS','TTSE&HEOS','BICUBIC&HEOS','REFPROP','SRK','PR'
    AS = CP.AbstractState(Backend, Ref)
    params = {
        'AS': AS,
        'eta_gen' : 0.88,    # Generator efficiency [-]
        'eta_mech' : 0.9,    # Shaft mechanical efficiency [-]
        'del_x_sep' : 0.025,  # Percentage vapor/liquid separation [-]
        'W_dot_elec': 59,      # Generator power output   [W]
        'eta_nozzle' : 0.504,
        'delta_p_Viper': 176000,
        'D_tube': in2m(0.3),    # Diameter of nozzle tube [m]
        'diameter': in2m(2.75), # Diameter of viper [m]
        'height' : in2m(9.75),  # Height of viper [m]
        'pin_r': 2709000,
        'hin_r': PropsSI('H','P',2709000,'T',42.64+ 273.15,Ref),
        'pout_r': 1297000, 
        'm_dot': 0.1025,  # Massflow [kg/s]
        'fluid': Ref,
        'Backend': Backend
    }
    
    viper = ViperClass(**params)
    Values = viper.Calculate()
    Values = ViperClass(**Values)
    print(Values.Quality,Values.eta_flow,Values.delta_p_nozzle)
    
