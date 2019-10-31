from __future__ import division, absolute_import, print_function
from CoolProp.CoolProp import PropsSI
import CoolProp as CP

from scipy.optimize import brentq
from math import pi,exp,log,sqrt,tan,cos,sin,pow,atan
# from  ACHP.convert_units import cms2gpm, psi2kPa, C2K, in2m
from  convert_units import cms2gpm, psi2kPa, C2K, in2m

class ExpDevClass():
    """
    Expansion devices models
    """
    def __init__(self,**kwargs):
        #Load the parameters passed in
        # using the dictionary
        self.__dict__.update(kwargs)
        
    def Update(self,**kwargs):
        #Update the parameters passed in
        # using the dictionary
        self.__dict__.update(kwargs)
    
    def OutputList(self): #TODO: fix this list of outputs
        """
            Return a list of parameters for this component for further output
             
            It is a list of tuples, and each tuple is formed of items:
                [0] Description of value
                [1] Units of value
                [2] The value itself
        """
         
        return [
            ('Expansion Device Type','-',self.ExpType),
            ('Upstream Pressure','Pa',self.pin_r),
            ('Upstream Enthalpy','j/kg',self.hin_r),
            ('Downstream Pressure','Pa',self.pout_r),
            ('Downstream Quality','-',self.xout_r),
            ('Mass flow rate','kg/s',self.mdot_r),

         ]
        
    def Initialize(self):
        
        # AbstractState
        assert hasattr(self,'AS'), 'Please specify the Abstract State'
        
        # If the user doesn't include the ExpType, fail
        assert hasattr(self,'ExpType'), 'Please specify the type of the expansion device'
    
    def natlconv(self, g,beta,nu,Pr,T_s,T_amb,L,g_fac):
        " Used in Viper expander "
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
        " Used in Viper expander "
        # Phase separation
        x_vap = 1 - del_x_sep
        x_liq = del_x_sep
        m_dot_vap = x_th*m_dot
        m_dot_liq = (1 - x_th)*m_dot
        return x_vap, m_dot_vap, x_liq, m_dot_liq
    
    def mv(self, n,c_v,m_dot,p,x):
        " Used in Viper expander "
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
        " Used in Viper expander "
        # Ball valve pressure drop calculation
        self.AS.update(CP.PQ_INPUTS, p, x)
        v = 1/self.AS.rhomass() # [m^3/kg]
        V_dot = m_dot*v
        V_dot_IP = cms2gpm(V_dot) #convert 'm^3/sec' to 'gal/min'
        delta_p_IP = c_v*V_dot_IP
        delta_p = psi2kPa(delta_p_IP)*1000 #convert 'psi' to 'Pa'
        return delta_p
    
        
    def Calculate(self):
        
        # Initialize
        self.Initialize()
        # AbstractState
        AS = self.AS
        
        if self.ExpType == 'Ideal':
            #===================================================================
            # No information about expansion device is given
            #===================================================================
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
                
            # outlet state (assume h = constant)
            self.hout_r = self.hin_r #[J/kg]
            
            AS.update(CP.PQ_INPUTS, self.pout_r, 0.0)
            Tbubble_out = AS.T() #[K]
            h_l_out = AS.hmass() #[J/kg]
            s_l_out = AS.smass() #[J/kg-K]
            AS.update(CP.PQ_INPUTS, self.pout_r, 1.0)
            Tdew_out = AS.T() #[K]
            h_v_out = AS.hmass() #[J/kg]
            s_v_out = AS.smass() #[J/kg-K]
            
            # outlet state (two-phase)
            self.xout_r = (self.hout_r-h_l_out)/(h_v_out-h_l_out) #[-]
            self.Tout_r = self.xout_r*Tdew_out+(1-self.xout_r)*Tbubble_out #[K]
            self.sout_r = self.xout_r*s_v_out+(1-self.xout_r)*s_l_out #[J/kg-K]
            
            # mass flow rate 
            self.mdot_r = 'N/A'
            
            # heat losses
            self.Q_amb = 0.0 #[W]
        
        if self.ExpType == 'Linear-TXV':
            #===================================================================
            # Global Linear TxV model from Haorong Li paper (2004)
            # paper title: Modeling Adjustable throat-Area Expansion Valves
            #===================================================================
            D = self.D                      #inside diameter [m]           
            Tsh_static = self.Tsh_static    #[K]         
            Tsh_max = self.Tsh_max          #[K]
            Adj = self.Adj                  #[-]      
            C = self.C                      #[m^2/K]
            
            Tsup = self.Tsup     #superheat value (user defined)
            
            P_up = self.pin_r
            P_down = self.pout_r
            
            A = (Tsup-Tsh_static)
            if (A>Tsh_max):
                A=Tsh_max
            
            ## thermodynamic properties
            AS.update(CP.PQ_INPUTS, self.pin_r, 0.0)
            Tbubble_in = AS.T() #[K]
            h_l_in = AS.hmass() #[J/kg]
            s_l_in = AS.smass() #[J/kg-K]
            rho_l_in = AS.rhomass() #[kg/m^3]
            AS.update(CP.PQ_INPUTS, self.pin_r, 1.0)
            Tdew_in = AS.T() #[K]
            h_v_in = AS.hmass() #[J/kg]
            s_v_in = AS.smass() #[J/kg-K]
            rho_v_in = AS.rhomass() #[kg/m^3]
            
            # inlet state
            self.xin_r = (self.hin_r-h_l_in)/(h_v_in-h_l_in)
            if (self.xin_r>0.999):
                print ("ExpDev :: Upstream state in the expansion device is superheated")
                raise
            if (self.xin_r>0.0 and self.xin_r<1.0):
                # 2phase upstream state
                print ("ExpDev :: Upstream state in the expansion device is 2-phase")
                self.sin_r = self.xin_r*s_v_in+(1-self.xin_r)*s_l_in #[J/kg-K]
                self.Tin_r = self.xin_r*Tdew_in+(1-self.xin_r)*Tbubble_in #[K]
            else: # liquid state at the inlet
                AS.update(CP.HmassP_INPUTS, self.hin_r, self.pin_r)
                self.sin_r = AS.smass() #[J/kg-K]
                self.Tin_r = AS.T() #[K]
            
            # upstream saturated liquid density
            rho_up = rho_l_in
            
            # calculate mass flow rate
            mdot_r = C*A*pow(rho_up*(P_up-P_down),0.5) 
            
            # adjust the mass flow rate via adjustment factor related with geometry (tuning factor)
            self.mdot_r = mdot_r*Adj
    
            # outlet state (assume h = constant)
            self.hout_r = self.hin_r #[J/kg]
            
            AS.update(CP.PQ_INPUTS, self.pout_r, 0.0)
            Tbubble_out = AS.T() #[K]
            h_l_out = AS.hmass() #[J/kg]
            s_l_out = AS.smass() #[J/kg-K]
            AS.update(CP.PQ_INPUTS, self.pout_r, 1.0)
            Tdew_out = AS.T() #[K]
            h_v_out = AS.hmass() #[J/kg]
            s_v_out = AS.smass() #[J/kg-K]
            
            # outlet state (two-phase)
            self.xout_r = (self.hout_r-h_l_out)/(h_v_out-h_l_out) #[-]
            self.Tout_r = self.xout_r*Tdew_out+(1-self.xout_r)*Tbubble_out #[K]
            self.sout_r = self.xout_r*s_v_out+(1-self.xout_r)*s_l_out #[J/kg-K]

            # heat losses
            self.Q_amb = 0.0 #[W]

        if self.ExpType == 'Nonlinear-TXV':
            #===================================================================
            # Nonlinear TxV model from Haorong Li paper (2004)
            # paper title: Modeling Adjustable throat-Area Expansion Valves
            #===================================================================
            D = self.D                      #inside diameter [m]           
            Tsh_static = self.Tsh_static    #[K]         
            Tsh_max = self.Tsh_max          #[K]
            Adj = self.Adj                  #[-]     
            C = self.C                      #[m^2/K]
            
            Tsup = self.Tsup     #superheat value (user defined)
            
            P_up = self.pin_r
            P_down = self.pout_r
            
            A = (Tsup-Tsh_static)/Tsh_max
            if (A>1):
                A=1
            
            ## thermodynamic properties
            AS.update(CP.PQ_INPUTS, self.pin_r, 0.0)
            Tbubble_in = AS.T() #[K]
            h_l_in = AS.hmass() #[J/kg]
            s_l_in = AS.smass() #[J/kg-K]
            rho_l_in = AS.rhomass() #[kg/m^3]
            AS.update(CP.PQ_INPUTS, self.pin_r, 1.0)
            Tdew_in = AS.T() #[K]
            h_v_in = AS.hmass() #[J/kg]
            s_v_in = AS.smass() #[J/kg-K]
            rho_v_in = AS.rhomass() #[kg/m^3]
            
            # inlet state
            self.xin_r = (self.hin_r-h_l_in)/(h_v_in-h_l_in)
            if (self.xin_r>0.999):
                print ("ExpDev :: Upstream state in the expansion device is superheated")
                raise
            if (self.xin_r>0.0 and self.xin_r<1.0):
                # 2phase upstream state
                print ("ExpDev :: Upstream state in the expansion device is 2-phase")
                self.sin_r = self.xin_r*s_v_in+(1-self.xin_r)*s_l_in #[J/kg-K]
                self.Tin_r = self.xin_r*Tdew_in+(1-self.xin_r)*Tbubble_in #[K]
            else: # liquid state at the inlet
                AS.update(CP.HmassP_INPUTS, self.hin_r, self.pin_r)
                self.sin_r = AS.smass() #[J/kg-K]
                self.Tin_r = AS.T() #[K]
               
            # upstream saturated liquid density
            rho_up = rho_l_in
            
            # calculate mass flow rate
            mdot_r = C*(2*A-A*A)*pow(rho_up*(P_up-P_down),0.5) 
            
            # adjust the mass flow rate via adjustment factor related with geometry (tuning factor)
            self.mdot_r = mdot_r*Adj
    
            # outlet state (assume h = constant)
            self.hout_r = self.hin_r #[J/kg]
            
            AS.update(CP.PQ_INPUTS, self.pout_r, 0.0)
            Tbubble_out = AS.T() #[K]
            h_l_out = AS.hmass() #[J/kg]
            s_l_out = AS.smass() #[J/kg-K]
            AS.update(CP.PQ_INPUTS, self.pout_r, 1.0)
            Tdew_out = AS.T() #[K]
            h_v_out = AS.hmass() #[J/kg]
            s_v_out = AS.smass() #[J/kg-K]
            
            # outlet state (two-phase)
            self.xout_r = (self.hout_r-h_l_out)/(h_v_out-h_l_out) #[-]
            self.Tout_r = self.xout_r*Tdew_out+(1-self.xout_r)*Tbubble_out #[K]
            self.sout_r = self.xout_r*s_v_out+(1-self.xout_r)*s_l_out #[J/kg-K]

            # heat losses
            self.Q_amb = 0.0 #[W]            
            
        if self.ExpType == 'Short-tube':
            #===================================================================
            # Short tube expansion from Payne and O'Neal (2004)
            # paper title: A Mass Flowrate Correlation for Refrigerants and Refrigerant Mixtures, Journal of HVAC
            # based on empirical dimensionless PI correlation, recommended for R-12, R-134a, R-502, R-22, R-407C, and R-410A
            #===================================================================
            D = self.D                      #inside diameter of the short-tube[m]                    
            L = self.L                      #length of the short-tube [m]
            Adj = self.Adj                  #adjusting the inside diameter [-];       
            L_c = self.L_c                  #chamfered length [m]
            Ang_c = self.Ang_c              #chamfered angle [degree]
            BranNum = int(self.BranNum)     #Number of Paralelled expansion devices 
                   
            A_s = pi/4*D*D
                
            P_up = self.pin_r
            P_down = self.pout_r
            
            # critical point of refirgerant
            P_c = AS.p_critical() #[Pa]
            T_c = AS.T_critical() #[K]
        
            # orifice adjustment parameter
            C_c = Adj
            
            ## thermodynamic properties
            AS.update(CP.PQ_INPUTS, self.pin_r, 0.0)
            Tbubble_in = AS.T() #[K]
            h_l_in = AS.hmass() #[J/kg]
            s_l_in = AS.smass() #[J/kg-K]
            rho_l_in = AS.rhomass() #[kg/m^3]
            AS.update(CP.PQ_INPUTS, self.pin_r, 1.0)
            Tdew_in = AS.T() #[K]
            h_v_in = AS.hmass() #[J/kg]
            s_v_in = AS.smass() #[J/kg-K]
            rho_v_in = AS.rhomass() #[kg/m^3]
            
            # inlet state
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
            
            AS.update(CP.QT_INPUTS, 0, self.Tin_r)
            P_sat = AS.p() #P_sat corresponding to upstream temperature (liquid saturation pressure) [Pa]
            T_sat = Tbubble_in
            T_sub = T_sat - self.Tin_r
            
            # upstream saturated liquid density
            AS.update(CP.PQ_INPUTS, P_sat, 0.0)
            rho_f = AS.rhomass() #[kg/m^3]
            # upstream saturated gas density
            AS.update(CP.PQ_INPUTS, P_sat, 1.0)
            rho_g = AS.rhomass() #[kg/m^3]
    
            # non-dimensional groups
            pi_3=(P_up-P_sat)/P_c
            pi_6=rho_g/rho_f
            pi_9=T_sub/T_c
            pi_10=L/D
        
            # coeffcients
            a1=3.8811e-1
            a2=1.1427e1
            a3=-1.4194e1
            a4=1.0703e0
            a5=-9.1928e-2
            a6=2.1425e1
            a7=-5.8195e2
            
            # single-phase flow rate
            pi_1 = (a1+a2*pi_3+a3*pi_9+a4*pi_6+a5*log(pi_10))/(1+a6*pi_3+a7*pi_9*pi_9);
            G = pi_1*pow((rho_f*P_c),0.5);
            
            # mass flow rate
            mdot_r = G*A_s
            
            if(self.xin_r<0.000001): #subcooled upstream state
                mdot_r = mdot_r
            
            else: #two-phase upstream state 
                x_up = self.xin_r
                rho_mup=1/((1-x_up)/rho_f+x_up/rho_g)
        
                # non-dimensional groups
                tp6=rho_mup/rho_f
                tp35=(P_c-P_sat)/(P_c)
                tp32=(P_c-P_up)/(P_c)
                tp27=L/D
                tp34=x_up/(1-x_up)*pow((rho_f/rho_g),0.5)
                tp28=P_up/P_c
        
                # coeffcients
                b1=1.1831e0
                b2=-1.468e0
                b3=-1.5285e-1
                b4=-1.4639e1
                b5=9.8401e0
                b6=-1.9798e-2
                b7=-1.5348e0
                b8=-2.0533e0
                b9=-1.7195e1
        
                numer = (b1*tp6+b2*pow(tp6,2.0)+b3*pow(log(tp6),2.0)+b4*pow(log(tp35),2.0)+b5*pow(log(tp32),2.0)+b6*pow(log(tp27),2.0))
                deno = (1+b7*tp6+b8*tp34+b9*pow(tp28,3.0))
                C_tp= numer/deno #two-phase flow rate adjustment
                
                if(C_tp>1):
                    C_tp=1 #since C_tp>1 is not right
                    
                # correct the mass flow rate by two-phase entrance
                mdot_r = mdot_r*C_tp
            
            # adjust the mass flow rate via adjustment factor related with geometry (tuning factor)
            self.mdot_r = mdot_r*C_c
            
            # multiply mass flow rate by the number of parallel branches
            if  BranNum == 0:
                self.mdot_r = self.mdot_r
            else:     
                self.mdot_r = self.mdot_r * BranNum
            
            # outlet state (assume h = constant)
            self.hout_r = self.hin_r #[J/kg]
            
            AS.update(CP.PQ_INPUTS, self.pout_r, 0.0)
            Tbubble_out = AS.T() #[K]
            h_l_out = AS.hmass() #[J/kg]
            s_l_out = AS.smass() #[J/kg-K]
            AS.update(CP.PQ_INPUTS, self.pout_r, 1.0)
            Tdew_out = AS.T() #[K]
            h_v_out = AS.hmass() #[J/kg]
            s_v_out = AS.smass() #[J/kg-K]
            
            # outlet state (two-phase)
            self.xout_r = (self.hout_r-h_l_out)/(h_v_out-h_l_out) #[-]
            self.Tout_r = self.xout_r*Tdew_out+(1-self.xout_r)*Tbubble_out #[K]
            self.sout_r = self.xout_r*s_v_out+(1-self.xout_r)*s_l_out #[J/kg-K]

        if self.ExpType == 'Expander':
            #===================================================================
            # General expander with given isentropic efficiency
            #===================================================================
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
                
            # isentropic outlet state
            AS.update(CP.PSmass_INPUTS,self.pout_r,self.sin_r)
            self.hout_s_r = AS.hmass() #[J/kg]
            
            # outlet state (assume eta_is = given)
            self.hout_r = self.hin_r - self.eta_is*(self.hin_r-self.hout_s_r) #[J/kg]
            
            AS.update(CP.PQ_INPUTS, self.pout_r, 0.0)
            Tbubble_out = AS.T() #[K]
            h_l_out = AS.hmass() #[J/kg]
            s_l_out = AS.smass() #[J/kg-K]
            AS.update(CP.PQ_INPUTS, self.pout_r, 1.0)
            Tdew_out = AS.T() #[K]
            h_v_out = AS.hmass() #[J/kg]
            s_v_out = AS.smass() #[J/kg-K]
            
            # outlet state (two-phase)
            self.xout_r = (self.hout_r-h_l_out)/(h_v_out-h_l_out) #[-]
            self.Tout_r = self.xout_r*Tdew_out+(1-self.xout_r)*Tbubble_out #[K]
            self.sout_r = self.xout_r*s_v_out+(1-self.xout_r)*s_l_out #[J/kg-K]

            # adjust the mass flow rate via adjustment factor related with geometry (tuning factor)
            # TODO: need to add a mass flow model 
            mdot_r = self.mdot
            self.mdot_r = mdot_r*self.C_exp
            
            # heat losses
            self.Q_amb = 0.0 #[W]

        if self.ExpType == 'Viper':
            #===================================================================
            # Viper expander model
            #===================================================================
            # inlet state
            if self.pin_r > AS.p_critical(): #Supercritical
                AS.update(CP.HmassP_INPUTS, self.hin_r, self.pin_r)
                self.sin_r = AS.smass() #[J/kg-K]
                self.Tin_r = AS.T() #[K]
                self.rhoin_r = AS.rhomass() # [kg/m^3]
            else: #other refrigerants  
                AS.update(CP.PQ_INPUTS, self.pin_r, 0.0)
                Tbubble_in = AS.T() #[K]
                h_l_in = AS.hmass() #[J/kg]
                s_l_in = AS.smass() #[J/kg-K]
                rho_l_in = AS.rhomass() # [kg/m^3]
                AS.update(CP.PQ_INPUTS, self.pin_r, 1.0)
                Tdew_in = AS.T() #[K]
                h_v_in = AS.hmass() #[J/kg]
                s_v_in = AS.smass() #[J/kg-K]
                rho_v_in = AS.rhomass() # [kg/m^3]
                
                self.xin_r = (self.hin_r-h_l_in)/(h_v_in-h_l_in)
                if (self.xin_r>0.999):
                    print ("ExpDev :: Upstream state in the expansion device is superheated")
                    raise
                if (self.xin_r>0.0 and self.xin_r<1.0): #two-phase state at the inlet
                    self.sin_r = self.xin_r*s_v_in+(1-self.xin_r)*s_l_in #[J/kg-K]
                    self.Tin_r = self.xin_r*Tdew_in+(1-self.xin_r)*Tbubble_in #[K]
                    self.rhoin_r = self.xin_r*rho_v_in+(1-self.xin_r)*rho_l_in # [kg/m^3]
                else: #liquid state at the inlet
                    AS.update(CP.HmassP_INPUTS, self.hin_r, self.pin_r)
                    self.sin_r = AS.smass() #[J/kg-K]
                    self.Tin_r = AS.T() #[K]
                    self.rhoin_r = AS.rhomass() # [kg/m^3]
            
            
            # Nozzel Model based on the work of Lennart's MS Thesis    
            D_m = self.D_m  # inside diameter [m]
            D_t = self.D_t  # throat diameter [m]
    
            
            AS.update(CP.PSmass_INPUTS, self.pout_r, self.sin_r)
            h_out_s = AS.hmass() #[J/kg]
            
            W_dot_is = self.mdot_r*(self.hin_r - h_out_s)
            
            # Ambient conditions - air properties"
            AS_air = CP.AbstractState('HEOS', 'air')
            #TO DO: need to vary this based on ambient temperature input by user
            T_amb = C2K(35)   # [C]
            p_atm = 101325 # [Pa]
            AS_air.update(CP.PT_INPUTS, p_atm, T_amb)
            k_air = AS_air.conductivity() # [W/m-K]
            mu_air = AS_air.viscosity() # [Pa-s]
            rho_air = AS_air.rhomass() #[kg/m^3]
            nu = mu_air/rho_air # Kinematic viscosity
            beta = 1/T_amb
            Pr = AS_air.Prandtl() # Prandtl number
            
            AS.update(CP.PQ_INPUTS, self.pout_r, 0.0)
            T_s = AS.T() #[K]
            g = 9.81     # [m/s^2]
            
            # Natural convection
            # Sides, vertical, cold
            A_side = self.height*self.diameter
            P_side = 2*self.height + 2*self.diameter
            L_side = A_side/P_side
            g_fac_side = 1
            Nusselt_L_side = self.natlconv(g,beta,nu,Pr,T_s,T_amb,L_side,g_fac_side)
            h_side = Nusselt_L_side*k_air/L_side
            Q_dot_side = h_side*A_side*(T_s - T_amb)
            # Top, horizontal, cold
            A_top = (pi*(self.diameter**2))/4
            P_top = pi*self.diameter
            L_top = A_top/P_top
            g_fac_top = 2 
            Nusselt_L_top = self.natlconv(g,beta,nu,Pr,T_s,T_amb,L_top,g_fac_top)
            h_top = Nusselt_L_top*k_air/L_top
            Q_dot_top = h_top*A_top*(T_s - T_amb)
            # Bottom, horizontal, cold
            A_bottom = A_top
            L_bottom = L_top
            g_fac_bottom = 3
            Nusselt_L_bottom = self.natlconv(g,beta,nu,Pr,T_s,T_amb,L_bottom,g_fac_bottom)
            h_bottom = Nusselt_L_bottom*k_air/L_bottom
            Q_dot_bottom = h_bottom*A_bottom*(T_s - T_amb)
             
            # heat losses
            Q_dot_loss_tot = Q_dot_bottom + Q_dot_top + Q_dot_side
            self.Q_amb = -Q_dot_loss_tot #[W]
            
            # basic calculation to start with
            A_inlet = pi * pow(D_m, 2) / 4
            u_inlet = self.mdot_r / (A_inlet * self.rhoin_r)
            A_t = pi * pow(D_t, 2) / 4
            
            
            def ObjectiveViper(delta_p_Viper):
                
                def ObjectiveNozzle(eta_nozzle):
                     
                    P_t = self.pout_r + delta_p_Viper                
                     
                    # calculation of enthalpy
                    AS.update(CP.PSmass_INPUTS, P_t, self.sin_r)
                    h_t_is = AS.hmass() #[J/kg]
                    h_t = self.hin_r - eta_nozzle * (self.hin_r - h_t_is)
                    AS.update(CP.PQ_INPUTS, P_t, 1.0)
                    rho_g = AS.rhomass() # [kg/m^3]
                    AS.update(CP.PQ_INPUTS, P_t, 0.0)
                    rho_l = AS.rhomass() # [kg/m^3]
                     
                    AS.update(CP.HmassP_INPUTS, h_t, P_t)
                    s_t = AS.smass() #[J/kg-K]
                    # calculation of mass fraction vapor at h_t and P_t
                    x_t = AS.Q() # [-]
         
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
                        rho_t = AS.rhomass() #density at h_t and P_t
         
                    u_t = sqrt(2 * (self.hin_r - h_t) + pow(u_inlet, 2))
         
                    if alpha_t > 1:
                        alpha_t = 0
                     
                    # now check the calculation with the massflow
                    m_dot_check = A_t * u_t * rho_t
                 
                    #pass Throat nozzle parameters
                    self.eta_nozzle = eta_nozzle
                    self.pt_r = P_t
                    self.xt_r = x_t
                    self.alphat_r = alpha_t
                    self.ut_r = u_t
                    self.st_r = s_t
                    self.ht_r = h_t
                     
                    return self.mdot_r - m_dot_check
                 
                #Actual solver for eta_nozzle
                brentq(ObjectiveNozzle,0.1,0.9)
                 
                # Calculate main parameters     
                self.delta_p_nozzle = self.pin_r - self.pt_r
                self.E_dot_nozzle = self.mdot_r*(self.hin_r - self.ht_r)  #Power recovered by nozzle [W]            
                
    #             #Power losses calculation
    #             self.W_dot_mech = self.W_dot_elec/self.eta_gen 
    #             self.W_dot_fluid_tot = self.W_dot_mech/self.eta_mech
    #                 
    #             # Turbine losses
    #             self.W_dot_fluid = self.W_dot_fluid_tot + Q_dot_loss_tot
    #             self.eta_flow = self.W_dot_fluid/self.E_dot_nozzle
     
     
                  
                # Turbine losses
                self.eta_flow = -4.3354*(self.pin_r/self.pt_r)**2 + 16.3*(self.pin_r/self.pt_r) - 14.777
                self.W_dot_fluid = self.eta_flow*self.E_dot_nozzle
                self.W_dot_fluid_tot = self.W_dot_fluid - Q_dot_loss_tot
                 
                # Power losses calculation
                self.W_dot_mech = self.W_dot_fluid_tot*self.eta_mech
                self.W_dot_elec = self.W_dot_mech*self.eta_gen
                self.CycleEnergyOut = -self.W_dot_elec + self.Q_amb
                
                #save delta_p_Viper
                self.delta_p_Viper = delta_p_Viper
                
                return self.W_dot_elec - self.W_dot_elec_target
            
            #Actual solver for delta_p_viper
            brentq(ObjectiveViper,self.delta_p_viper_init-100000,self.delta_p_viper_init+100000)
            
            
            def ObjectiveSeperation(delta_x_sep):
                #Phase separation losses
                self.x_vap,self.mdot_vap,self.x_liq,self.mdot_liq = self.phasesep(self.xt_r,delta_x_sep,self.mdot_r)
                self.delta_x_sep = delta_x_sep
                
                # Valve losses
                #metering valve
                c_v_mv = 0.73    #[psi/(gal/min)]
                n_MV = 2    # [-]
                #ball valve"
                c_v_bv = 4.4    # [psi/(gal/min)]
                #Vapor line valve losses
                delta_p_mv_vap = self.mv(n_MV,c_v_mv,self.mdot_vap,self.pout_r,self.x_vap)
                delta_p_bv_vap = self.bv(c_v_bv,self.mdot_vap,self.pout_r,self.x_vap)
                #Liquid line valve losses
                delta_p_bv_liq = self.bv(c_v_bv,self.mdot_liq,self.pout_r,self.x_liq)
                
                # Outlet state (vapor)
                AS.update(CP.PQ_INPUTS, self.pout_r, self.x_vap)
                self.Tout_vap = AS.T() #[K]
                self.hout_vap = AS.hmass() #[J/kg]
                self.sout_vap = AS.smass() #[J/kg-K]
                # Outlet state (liquid)
                AS.update(CP.PQ_INPUTS, self.pout_r, self.x_liq)
                self.Tout_liq = AS.T() #[K]
                self.hout_liq = AS.hmass() #[J/kg]
                self.sout_liq = AS.smass() #[J/kg-K]
            
                return self.mdot_r*self.ht_r - (self.mdot_liq*self.hout_liq + self.mdot_vap*self.hout_vap)
            
            #Actual solver for delta_x_sep
            brentq(ObjectiveSeperation,0.0,0.5)

            
                              
if __name__=='__main__':
    #Abstract State
    Ref = 'R410A'
    Backend = 'HEOS' #choose between: 'HEOS','TTSE&HEOS','BICUBIC&HEOS','REFPROP','SRK','PR'
    AS = CP.AbstractState(Backend, Ref)
    
    print('Example for Ideal expansion device')
    params={
        'AS':AS,
        'ExpType':'Ideal',     #expansion device type
        'pin_r': PropsSI('P','T',60+273.15,'Q',0,Ref), #upsteam pressure
        'hin_r': PropsSI('H','P',PropsSI('P','T',60+273.15,'Q',0,Ref),'Q',0,Ref), #upstream enthalpy
        'pout_r': PropsSI('P','T',10+273.15,'Q',0,Ref), #downstream pressure        
    }
    Exp=ExpDevClass(**params)
    Exp.Calculate()
    print(Exp.OutputList())
    print()
    
    
    print('Example for Linear expansion device')
    params={
        'AS':AS,
        'ExpType':'Linear-TXV',     #expansion device type
        'Tsh_static':4,             #static superheat
        'Tsh_max':6,                #maximum superheat
        'D':0.0006604,              #inside diameter [m]
        'C':1.2656e-6,              #constant from manufacturer [m^2/K]
        'Adj':0.7630,               #Adjust the diameter (tuning factor)
        'Tsup':5,                   #superheat value (user defined)
        'pin_r': PropsSI('P','T',60+273.15,'Q',0,Ref), #upsteam pressure
        'hin_r': PropsSI('H','P',PropsSI('P','T',60+273.15,'Q',0,Ref),'Q',0,Ref), #upstream enthalpy
        'pout_r': PropsSI('P','T',10+273.15,'Q',0,Ref), #downstream pressure        
    }  
    Exp=ExpDevClass(**params)
    Exp.Calculate()
    print('Tout =',Exp.Tout_r,'[K]')
    print('hout =',Exp.hout_r,'[J/kg]')
    print('xout =',Exp.xout_r,'[-]')
    print('mdot_r =',Exp.mdot_r,'[kg/s]')
    print()
    
    
    print('Example for short-tube expansion device')
    params={
        'AS':AS,
        'ExpType':'Short-tube',     #expansion device type
        'D':0.0006604,              #inside diameter [m]
        'L':0.0052324,              #length of short-tube[m]
        'L_c':0.0001524,            #chamfered length [m] (P.S. not included in the solver yet)
        'Ang_c':45,                 #chamfered angle [degree] (P.S. not included in the solver yet)
        'BranNum':12,               #Number of Paralelled short-tubes (0 -- default for 1 short-tube only)
        'Adj':1.094,                #Adjust the diameter (tuning factor)
        'pin_r': PropsSI('P','T',60+273.15,'Q',0,Ref), #upsteam pressure
        'hin_r': PropsSI('H','P',PropsSI('P','T',60+273.15,'Q',0,Ref),'Q',0,Ref), #upsteam enthalpy
        'pout_r': PropsSI('P','T',10+273.15,'Q',0,Ref), #downstream pressure        
    }
    Exp.Update(**params)
    Exp.Calculate()
    print('Tout =',Exp.Tout_r,'[K]')
    print('hout =',Exp.hout_r,'[J/kg]')
    print('xout =',Exp.xout_r,'[-]')
    print('mdot_r =',Exp.mdot_r,'[kg/s]')
    print()


    print('Example for expander device')
    params={
        'AS':AS,
        'ExpType':'Expander',       #expansion device type
        'eta_is':0.8,               #isentropic efficiency [-]
        'C_exp':1,                  #flow factor [-]                  
        'mdot':0.01,                # mass flow rate [kg/s]
        'pin_r': PropsSI('P','T',60+273.15,'Q',0,Ref), #upsteam pressure
        'hin_r': PropsSI('H','P',PropsSI('P','T',60+273.15,'Q',0,Ref),'Q',0,Ref), #upsteam enthalpy
        'pout_r': PropsSI('P','T',10+273.15,'Q',0,Ref), #downstream pressure        
    }
    Exp.Update(**params)
    Exp.Calculate()
    print('Tout =',Exp.Tout_r,'[K]')
    print('hout =',Exp.hout_r,'[J/kg]')
    print('xout =',Exp.xout_r,'[-]')
    print('hout_s =',Exp.hout_s_r,'[J/kg]')
    print('mdot_r =',Exp.mdot_r,'[kg/s]')
    print()
    
    
    print('Example for Viper expander')
    params={
    #Original code from Ammar
        # 'AS':AS,
        # 'ExpType':'Viper',       #expansion device type         
        # 'mdot_r': 0.1025,                # mass flow rate [kg/s]
        # 'pin_r': 2709000, #upsteam pressure
        # 'hin_r': PropsSI('H','P',2709000,'T',42.64+ 273.15,'R410A'), #upsteam enthalpy
        # 'pout_r': 1297000, #downstream pressure   
        # 
        # 'D_m': 0.3 * 0.0254,    # inside diameter pipe [m]
        # 'D_t': 0.09 * 0.0254,   # inside diameter throat [m]  v
        # 'diameter': in2m(2.75), # diameter of viper [m]
        # 'height' : in2m(9.75),  # height of viper [m]
        # 'delta_p_Viper': 176000, #pressure drop across viper
        # 'eta_gen' : 0.88,       # generator efficiency [-]
        # 'eta_mech' : 0.9,       # shaft mechanical efficiency [-]
        # 'del_x_sep' : 0.025,    # percentage vapor/liquid separation [-]
        # 'W_dot_elec': 59,       # generator power output   [W]  
        
        #Constants over all data test points   
        
        'AS':AS,
        'ExpType':'Viper',       #expansion device type      
        'D_m': 0.3 * 0.0254,    # inside diameter pipe [m]
        'D_t': 0.09 * 0.0254,   # inside diameter throat [m]  v
        'diameter': in2m(2.75), # diameter of viper [m]
        'height' : in2m(9.75),  # height of viper [m]
        'delta_p_viper_init': 176000, #pressure drop across viper
        'eta_gen' : 0.88,       # generator efficiency [-]
        'eta_mech' : 0.9,       # shaft mechanical efficiency [-]        
           

        
        # # Hybrid_Control_12_11_2018_Viper_I752_O95
#         'mdot_r': 0.1015,                # mass flow rate [kg/s]
#         'pin_r': 2690000, #upsteam pressure
#         'hin_r': PropsSI('H','P',2690000,'T',42.5+ 273.15,'R410A'), #upsteam enthalpy
#         'W_dot_elec_target': 58.6,       # generator power output   [W]    
#         'pout_r': 1279000,
        
        # # Hybrid_Control_12_11_2018_Viper_I743_O95
#         'mdot_r': 0.1009,                # mass flow rate [kg/s]
#         'pin_r': 2680000, #upsteam pressure
#         'hin_r': PropsSI('H','P',2680000,'T',42.54+ 273.15,'R410A'), #upsteam enthalpy
#         'W_dot_elec_target': 58.4,       # generator power output   [W]    
#         'pout_r': 1270000,
          
        
        # # Hybrid_Control_12_11_2018_Viper_I80_O95
#         'mdot_r': 0.1049,                # mass flow rate [kg/s]
#         'pin_r': 2723000, #upsteam pressure
#         'hin_r': PropsSI('H','P',2723000,'T',42.32+ 273.15,'R410A'), #upsteam enthalpy
#         'W_dot_elec_target': 58,       # generator power output   [W]    
#         'pout_r': 1325000,
        
        # # Hybrid_Control_12_11_2018_Viper_I77_O1076
#         'mdot_r': 0.1049,                # mass flow rate [kg/s]
#         'pin_r': 3110000, #upsteam pressure
#         'hin_r': PropsSI('H','P',3110000,'T',49.94 + 273.15,'R410A'), #upsteam enthalpy
#         'W_dot_elec_target': 76.4,       # generator power output   [W]    
#         'pout_r': 1405000,
        
        
        # # Hybrid_Control_12_11_2018_Viper_I77_O104
#         'mdot_r': 0.1043,                # mass flow rate [kg/s]
#         'pin_r': 2991000, #upsteam pressure
#         'hin_r': PropsSI('H','P',2991000,'T',47.68+ 273.15,'R410A'), #upsteam enthalpy
#         'W_dot_elec_target': 71.7,       # generator power output   [W]    
#         'pout_r': 1377000,
        
        
        # # Hybrid_Control_12_11_2018_Viper_I77_O95
        'mdot_r': 0.1025,                # mass flow rate [kg/s]
        'pin_r': 2709000, #upsteam pressure
        'hin_r': PropsSI('H','P',2709000,'T',42.64+ 273.15,'R410A'), #upsteam enthalpy
        'W_dot_elec_target': 59,       # generator power output   [W]    
        'pout_r': 1297000,
          
        
        # # Oct. 12th 2017 0.90" straight nozzle data, no separation
#         'mdot_r': 0.09937,                # mass flow rate [kg/s]
#         'pin_r': 2629230, #upsteam pressure
#         'hin_r': PropsSI('H','P',2629230,'T',41.38 + 273.15,'R410A'), #upsteam enthalpy
#         'W_dot_elec_target': 26.4,       # generator power output   [W]    
#         'pout_r': 1154620,

    }
    Exp.Update(**params)
    Exp.Calculate()
    print('pin_r =',Exp.pin_r/1000,'[kPa]')
    print('pt_r =',Exp.pt_r/1000,'[kPa]')
    print('pout_r =',Exp.pout_r/1000,'[kPa]')
    print('xt_r =',Exp.xt_r,'[-]')
    print('ht_r =',Exp.ht_r,'[J/kg]')
    print('mdot_r =',Exp.mdot_r,'[kg/s]')
    print('eta_nozzle =',Exp.eta_nozzle,'[-]')
    print('eta_flow =',Exp.eta_flow,'[-]')
    print('W_dot_elec =',Exp.W_dot_elec,'[W]')
    print('delta_x_sep =',Exp.delta_x_sep,'[-]')
    print()