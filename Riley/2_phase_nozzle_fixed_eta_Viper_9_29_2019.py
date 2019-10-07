from __future__ import division, absolute_import, print_function
from CoolProp.CoolProp import PropsSI
import CoolProp as CP
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi, pow, atan, log


class NozzleFlow():

    """
    Nozzle Flow analytics of two phase flow at the inlet of the viper
    """

    def __init__(self, **kwargs):
        # Load the parameters passed in
        # using the dictionary
        self.__dict__.update(kwargs)

    def Update(self, **kwargs):
        # Update the parameters passed in
        # using the dictionary
        self.__dict__.update(kwargs)

    def Calculate_Convergent(self):
        D_m = self.D_m  # inside diameter [m]
        D_t = self.D_t  # throat diameter [m]
        P_inlet = self.pin_r
        P_outlet = self.pout_r
        m_dot = self.m_dot

        # guess for P_t and nozzle efficency
        # P_t_init = P_inlet - 5000
        P_t_init = P_inlet - 500
        # eta_n_init = 0.65
        eta_n_init = 0.472


        # thermodynamic properties
        h_inlet = PropsSI('H', 'P', P_inlet, 'T', self.T_in, self.fluid)  # [J/kg]
        print(h_inlet),
        rho_inlet = PropsSI('D', 'P', P_inlet, 'H', h_inlet, self.fluid)  # [kg/m^3]
        s_inlet = PropsSI('S', 'P', P_inlet, 'T', self.T_in, self.fluid)

        # basic calculation to start with
        A_inlet = pi * pow(D_m, 2) / 4
        u_inlet = m_dot / (A_inlet * rho_inlet)
        print(u_inlet)
        A_t = pi * pow(D_t, 2) / 4

        # initalization
        m_dot_check = 0
        # tol_massflow = 0.00001
        tol_massflow = 0.0001


        while abs(m_dot - m_dot_check) > tol_massflow:
            # calculation of enthalpy
            h_t_is = PropsSI('H', 'P', P_t_init, 'S', s_inlet, self.fluid)
            h_t = h_inlet - eta_n_init * (h_inlet - h_t_is)
            rho_g = PropsSI('D', 'P', P_t_init, 'Q', 1, self.fluid)
            rho_l = PropsSI('D', 'P', P_t_init, 'Q', 0, self.fluid)
            s_t = PropsSI('S', 'P', P_t_init, 'H', h_t, self.fluid)

            # calculation of mass fraction vapor
            x_t = PropsSI('Q', 'P', P_t_init, 'H', h_t, self.fluid)

            s = 2

            if s == 1:
                slip = pow(rho_l / rho_g, 1. / 3)
            elif s == 2:
                e = 0.12  # entrainment needs to be calculated or kept as a variable
                slip = e + (1 - e) * pow(((rho_l / rho_g) + e * ((1 - x_t) / x_t)) / (1 + e * ((1 - x_t) / x_t)),1. / 2)
            elif s == 3 and x_t > 0:
                slip = sqrt((1 + x_t * (rho_l / rho_g - 1)))
            else:
                slip = 1

            # calculation of void fraction
            alpha_t = 1 / (1 + ((1 - x_t) / x_t) * rho_g / rho_l * slip)

            if x_t <= 0:
                x_t = 0

            # calculate of mixing density
            if x_t > 0 and x_t < 1:
                rho_t = alpha_t * rho_g + (1 - alpha_t) * rho_l
            else:
                rho_t = PropsSI('D', 'P', P_t_init, 'H', h_t, self.fluid)

            u_t = sqrt(2 * (h_inlet - h_t) + pow(u_inlet, 2))

            if alpha_t > 1:
                alpha_t = 0

            # now check the caculation with the massflow
            m_dot_check = A_t * u_t * rho_t

            if abs(m_dot - m_dot_check) > tol_massflow:
                P_t_init = P_t_init - 1000
            else:
                break

        para_speed_calc = {
            'P_t': P_t_init,
            'Q': x_t,
            'alpha': alpha_t,
            'u_t': u_t,
            'm_dot': self.m_dot,
            'D_t': self.D_t,
            'pout_r': self.pout_r,
            'fluid': self.fluid,
            'S': s_t,
            'H': h_t

        }

        print(x_t,h_t,D_t,m_dot),
        return para_speed_calc

# Try if the program isn't used from outside
Ref = 'R410A'
Backend = 'HEOS'  # choose between: 'HEOS','TTSE&HEOS','BICUBIC&HEOS','REFPROP','SRK','PR'
AS = CP.AbstractState(Backend, Ref)
params = {
    'AS': AS,
    'D_m': 0.3 * 0.0254,  # inside diameter pipe [m]
    'D_t': 0.09 * 0.0254,  # inside diameter throat [m]
    
   # # NOZZLE VALIDATION CASE
   #  'pin_r': 2572000,
   #  'T_in': 39.75 + 273.15,
   # # 'pout_r': PropsSI('P', 'T', 13 + 273.15, 'Q', 0, Ref),  # downstream pressure
   #  'pout_r': 1408000,
   #  'm_dot': 0.1055,  # Massflow [kg/s]
   
    # # # Hybrid_Control_12_11_2018_Viper_I752_O95
    # 'pin_r': 2690000,
    # 'T_in': 42.5 + 273.15,
    # 'pout_r': 1279000 + 176000, # 17600 to make up for delta_p_Viper
    # 'm_dot': 0.1015,  # Massflow [kg/s]
    
    # # Hybrid_Control_12_11_2018_Viper_I743_O95
    # 'pin_r': 2680000,
    # 'T_in': 42.54 + 273.15,
    # 'pout_r': 1270000 + 176000,
    # 'm_dot': 0.1009,  # Massflow [kg/s]
    
    # # Hybrid_Control_12_11_2018_Viper_I80_O95
    # 'pin_r': 2723000,
    # 'T_in': 42.32 + 273.15,
    # 'pout_r': 1325000 + 176000,
    # 'm_dot': 0.1041,  # Massflow [kg/s]
    
    # # Hybrid_Control_12_11_2018_Viper_I77_O1076
    # 'pin_r': 3110000,
    # 'T_in': 49.94 + 273.15,
    # 'pout_r': 1405000 + 176000,
    # 'm_dot': 0.1049,  # Massflow [kg/s]
    
    # # Hybrid_Control_12_11_2018_Viper_I77_O104
    # 'pin_r': 2991000,
    # 'T_in': 47.68 + 273.15,
    # 'pout_r': 1377000 + 176000,
    # 'm_dot': 0.1043,  # Massflow [kg/s]
    
    # # Hybrid_Control_12_11_2018_Viper_I77_O95
    # 'pin_r': 2709000,
    # 'T_in': 42.64 + 273.15,
    # 'pout_r': 1297000 + 176000,
    # 'm_dot': 0.1025,  # Massflow [kg/s]
    
    # Oct. 12th 2017 0.90" straight nozzle data, no separation
    'pin_r': 2629230,
    'T_in': 41.38 + 273.15,
    'pout_r': 1154620 + 176000, 
    'm_dot': 0.09937,  # Massflow [kg/s]
    
    
    'fluid': Ref,
}

Throat = NozzleFlow(**params)
Throat_Values = Throat.Calculate_Convergent()

Throat_Values = NozzleFlow(**Throat_Values)
print(Throat_Values.P_t,Throat_Values.u_t)


