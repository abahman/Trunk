from __future__ import division, print_function, absolute_import

import numpy as np
from CoolProp.CoolProp import PropsSI
import CoolProp as CP


class VICompressorLumpkinBPIClass():

    """
    Lumpkin at al. (2018) IJR 88, 449-462
    Correlations based on Buckingham PI theorem 

    Required Parameters:

    ===========   ==========  ========================================================================
    Variable      Units       Description
    ===========   ==========  ========================================================================
    M             Ibm/hr      A numpy-like list of compressor map coefficients for mass flow
    P             Watts       A numpy-like list of compressor map coefficients for electrical power
    Ref           N/A         A string representing the refrigerant
    Tin_r         K           Refrigerant inlet temperature
    pin_r         Pa          Refrigerant suction pressure (absolute)
    pout_r        Pa          Refrigerant discharge pressure (absolute)
    fp            --          Fraction of electrical power lost as heat to ambient
    Vdot_ratio    --          Displacement Scale factor
    ===========   ==========  ========================================================================

    All variables are of double-type unless otherwise specified

    """

    def __init__(self,**kwargs):
        #Load up the parameters passed in
        # using the dictionary
        self.__dict__.update(kwargs)

    def Update(self,**kwargs):
        #Update the parameters passed in
        # using the dictionary
        self.__dict__.update(kwargs)

    def OutputList(self):

        """
            Return a list of parameters for this component for further output
            It is a list of tuples, and each tuple is formed of items with indices:
                [0] Description of value
                [1] Units of value
                [2] The value itself
        """
        return [ 
            ('Heat Loss Fraction','-',self.fq),
            ('Displacement scale factor','-',self.Vdot_ratio),
            ('Power','W',self.Wdot),
            ('Suction mass flow rate','kg/s',self.mdot_r),
            ('Injection mass flow rate','kg/s',self.mdot_inj),
            ('Total mass flow rate','kg/s',self.mdot_tot),
            ('Inlet Temperature','K',self.Tin_r),
            ('Injection Temperature','K',self.Tinj_r),
            ('Outlet Temperature','K',self.Tout_r),
            ('Inlet Enthalpy','J/kg',self.hin_r),
            ('Injection Enthalpy','J/kg',self.hinj_r),
            ('Outlet Enthalpy','J/kg',self.hout_r),
            ('Inlet Pressure','Pa',self.pin_r),
            ('Injection Pressure','Pa',self.pinj_r),
            ('Outlet Pressure','Pa',self.pout_r),
            ('Inlet Entropy','J/kg-K',self.sin_r),
            ('Injection Entropy','J/kg-K',self.sinj_r),
            ('Outlet Entropy','J/kg-K',self.sout_r),
            ('Overall isentropic efficiency','-',self.eta_oi),
            ('Volumetric efficiency','-',self.eta_oi),
            ('Pumped flow rate','m**3/s',self.Vdot_pumped),
            ('Ambient heat loss','W',self.Q_amb)
         ]

        

    def Calculate(self):

        #AbstractState
        AS = self.AS

        #Local copies of coefficients
        W=self.W
        M=self.M
        T=self.T
        Eis=self.Eis
        Ev=self.Ev
        Fq=self.Fq
        
        #Critical temperature and pressure
        Tcrit = AS.T_critical() #[K]
        pcrit = AS.p_critical() #[Pa]

        #Calculate suction and injection dew temperatures
        AS.update(CP.PQ_INPUTS, self.pin_r, 0.0)
        h_bubble=AS.hmass() #[J/kg]

        AS.update(CP.PQ_INPUTS, self.pinj_r, 0.0)
        h_bubble_inj=AS.hmass() #[J/kg]

        AS.update(CP.PT_INPUTS, self.pinj_r, self.Tinj_r)
        self.hinj_r=AS.hmass() #[J/kg]
        self.sinj_r=AS.smass() #[J/kg-K]

        AS.update(CP.PT_INPUTS, self.pin_r, self.Tin_r)
        self.hin_r=AS.hmass() #[J/kg]
        self.sin_r=AS.smass() #[J/kg-K]
        self.vin_r = 1 / AS.rhomass() #[m**3/kg]

        #AS.update(CP.QT_INPUTS, 0.0, self.Tinj_r)
        AS.update(CP.PQ_INPUTS, self.pinj_r, 0.0)
        h_f_inj=AS.hmass()#[J/kg]

        #AS.update(CP.QT_INPUTS, 1.0, self.Tinj_r)
        AS.update(CP.PQ_INPUTS, self.pinj_r, 1.0)
        h_g_inj=AS.hmass()#[J/kg]

        #AS.update(CP.QT_INPUTS, 0.0, self.Tin_r)
        AS.update(CP.PQ_INPUTS, self.pin_r, 0.0)
        h_f=AS.hmass()#[J/kg]

        #AS.update(CP.QT_INPUTS, 1.0, self.Tin_r)
        AS.update(CP.PQ_INPUTS, self.pin_r, 1.0)
        h_g=AS.hmass()#[J/kg]

        #Injection:
        Dh_shinj_r=self.hinj_r-h_bubble_inj
        Dh_fg_inj=h_g_inj-h_f_inj

        #Suction
        Dh_sh=self.hin_r-h_bubble
        Dh_fg=h_g-h_f
        
        # Maximum compressor power
        Wdot_max = self.costheta*np.sqrt(3)*self.Iph_max*self.DV
        
        #Normalized quanities
        deltah_inj_norm = Dh_shinj_r/Dh_fg_inj
        deltah_suc_norm = Dh_sh/Dh_fg
        Tsuc_amb_norm = self.Tin_r/self.Tamb
        Tsuc_crit_norm = self.Tin_r/Tcrit
        psuc_crit_norm = self.pin_r/pcrit
        pdis_crit_norm = self.pout_r/pcrit
        pinj_norm = self.pinj_r/self.pin_r
        pdis_suc_norm = self.pout_r/self.pin_r
        f_norm = self.f_comp/self.f_nom
        

        #Mass flow rate ratio [kg/s]
        m_inj_suc_norm =  M[0]*(f_norm**M[1])*(deltah_inj_norm**M[2])*(pinj_norm**M[3])*(Tsuc_amb_norm**M[4])*(pdis_suc_norm**M[5])*(deltah_suc_norm**M[6])*(pdis_crit_norm**M[7])*(psuc_crit_norm**M[8])*(Tsuc_crit_norm**M[9])
        
        #Discharge temperature [K]
        T_dis = Tcrit*(T[0]*(f_norm**T[1])*(m_inj_suc_norm**T[2])*(pinj_norm**T[3])*(Tsuc_amb_norm**T[4])*(pdis_suc_norm**T[5])*(deltah_suc_norm**T[6])*(pdis_crit_norm**T[7])*(psuc_crit_norm**T[8])*(Tsuc_crit_norm**T[9]))

        #Compressor power [W]
        Wdot_norm = W[0]*f_norm**W[1]*m_inj_suc_norm**W[2]*pinj_norm**W[3]*Tsuc_amb_norm**W[4]*pdis_suc_norm**W[5]*deltah_suc_norm**W[6]*pdis_crit_norm**W[7]*psuc_crit_norm**W[8]*Tsuc_crit_norm**W[9]

        #Compressor overall isentropic efficiency [-]
        eta_is = Eis[0]*f_norm**Eis[1]*m_inj_suc_norm**Eis[2]*pinj_norm**Eis[3]*Tsuc_amb_norm**Eis[4]*pdis_suc_norm**Eis[5]*deltah_suc_norm**Eis[6]*pdis_crit_norm**Eis[7]*psuc_crit_norm**Eis[8]*Tsuc_crit_norm**Eis[9]

        #Compressor volumetric efficiency [-]
        eta_v = Ev[0]*f_norm**Ev[1]*m_inj_suc_norm**Ev[2]*pinj_norm**Ev[3]*Tsuc_amb_norm**Ev[4]*pdis_suc_norm**Ev[5]*deltah_suc_norm**Ev[6]*pdis_crit_norm**Ev[7]*psuc_crit_norm**Ev[8]*Tsuc_crit_norm**Ev[9]

        #Compressor vheat loss factor [-]
        f_q = Fq[0]*f_norm**Fq[1]*m_inj_suc_norm**Fq[2]*pinj_norm**Fq[3]*Tsuc_amb_norm**Fq[4]*pdis_suc_norm**Fq[5]*deltah_suc_norm**Fq[6]*pdis_crit_norm**Fq[7]*psuc_crit_norm**Fq[8]*Tsuc_crit_norm**Fq[9]        
        
        
        #Suction mass flow rate [kg/s]
        mdot = eta_v*self.Vdisp*(self.f_comp*60/60)/self.vin_r
        
        #Injection mass flow rate [kg/s]
        mdot_inj = mdot*m_inj_suc_norm

        #Compressor discharge state
        self.Tout_r = T_dis #[K]
        AS.update(CP.PT_INPUTS, self.pout_r, self.Tout_r)
        self.hout_r = AS.hmass() #[J/kg]
        self.sout_r = AS.smass() #[J/kg-K]        

        #define properites for isentropic efficency (hdis_isen)
        AS.update(CP.PSmass_INPUTS, self.pout_r, self.sin_r)
        h_2s=AS.hmass() #[J/kg]
        #(hinj,isen)
        AS.update(CP.PSmass_INPUTS, self.pinj_r, self.sin_r)
        h_4s=AS.hmass() #[J/kg]


        self.eta_oi = eta_is
        self.eta_v = eta_v
        self.mdot_r = mdot
        self.mdot_inj = mdot_inj
        self.mdot_tot = mdot + mdot_inj
        self.Wdot = Wdot_norm*Wdot_max
        self.fq = f_q
        self.CycleEnergyIn = self.Wdot*(1-self.fq)
        self.Vdot_pumped = mdot*self.vin_r
        self.Q_amb = -self.fq*self.Wdot

        #isentropic efficiency defined by Groll
        h_4=(self.mdot_r*h_2s + self.mdot_inj*self.hinj_r)/self.mdot_tot

if __name__=='__main__':        

    import numpy as np
    
    #Abstract State        
    Ref = 'R407C'
    Backend = 'REFPROP' #choose between: 'HEOS','TTSE&HEOS','BICUBIC&HEOS','REFPROP','SRK','PR'
    AS = CP.AbstractState(Backend, Ref)
    
    Tsuc = 288.9 #K
    psuc = 537.5 #kPa
    pdis = 1473 #kPa
    pinj = 837.3 #kPa
    Tinj = 291.2 #K
    Tamb = 308.2 #K

    kwds={
          'M':[0.08758,1,-1.97,0.5076,-2.745,1.334,-1.47,0.6262,0.3004,-0.2136],
          'W':[2.671,1,0.06768,0.08695,0.4397,0.4899,-0.6598,0.1848,0.699,0.3221],
          'T':[1.206,1,-0.01493,-0.005106,-0.5655,0.4426,-0.7388,-0.2694,0.2932,1.474],
          'Eis':[0.4,1,0.02304,-0.3125,3.792,0.5774,0.3489,-0.4794,-0.07093,1.974],
          'Ev':[1.169,1,0.00497,-0.02236,-0.6345,0.3229,-0.1655,-0.3987,0.2806,1.856],
          'Fq':[10,1,0.3288,-1.248,6.196,0.9401,2.269,0.3922,0.4534,-2.865],
          'AS':AS,
          'Tin_r':Tsuc,
          'pin_r':psuc*1000, #Pa
          'pout_r':pdis*1000, #Pa
          'pinj_r':pinj*1000, #Pa
          'Tinj_r':Tinj,
          'Tamb':Tamb,
          'fp':0.15, #Fraction of electrical power lost as heat to ambient
          'Vdisp':67.186962e-6, #[m3/rev] compressor displacement
          'Vdot_ratio': 1.0, #Displacement Scale factor
          'f_comp': 60, #compressor frequency
          'f_nom': 60, #nominal frequency
          'DV': 230, # Voltage
          'costheta': 0.85,  # cos(theta) factor
          'Iph_max': 18 #[A]
          }

    Comp=VICompressorLumpkinBPIClass(**kwds)
    Comp.Calculate()

    print (Comp.Wdot,'W')
    print (Comp.Tout_r,'K')
    print (Comp.mdot_r,'kg/s')
    print (Comp.mdot_inj,'kg/s')
    print (Comp.Q_amb, 'W')
    print (Comp.eta_oi*100, '%')
    print (Comp.eta_v*100, '%')
