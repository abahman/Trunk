from __future__ import division
#from CoolProp.CoolProp import Props,Tsat,IsFluidType
from Refpropp_mix import SetFluids,Props,Tsat,IsFluidType
from Correlations import f_h_1phase_Tube,TrhoPhase_ph,TwoPhaseDensity,Friction_Fanning,LMPressureGradientAvg,AccelPressureDrop
from math import log,pi,exp
from ACHPTools import DataHolderFromFile, Write2CSV
import time

class LineSetClass():
    
    def __init__(self,**kwargs):
        
        self.reducers = [] #list of reducer objects
        self.len14 = 0 #length of 1/4" OD line in inches
        self.len34 = 0 #length of 3/4" OD line in inches
        self.len78 = 0 #length of 7/8" OD line in inches
        self.len138 = 0 #length of 1 3/8" line in inches
        self.T78_line = 0 #number of plain 7/8" tees with line flow path
        self.T78_branch = 0 #number of 7/8" tees with branch flow path
        self.T78_tc = 0 #number of 7/8" tees for thermocouple
        self.T78_pt = 0 #number of 7/8" tees for pressure transducer
        self.T78_cp_line = 0 #number of 7/8" tees for charge port with line flow path
        self.T78_cp_branch = 0 #number of 7/8" tees for charge port with branch flow path
        self.T78r58_oil = 0 #number of oil mixing tees
        self.T78r58_tc = 0 #number of 7/8" tees with reduction to 5/8" for thermocouple
        self.T78r58_pt = 0 #number of 7/8" tees with reduction to 5/8" for pressure transducers
        self.T78r58_gage = 0 #number of 7/8" tees with reduction to 5/8" for pressure gage
        self.T138_line = 0 #number of plain 1 3/8" tees with line flow path
        self.T138_branch = 0 #number of 1 3/8" tees with branch flow path
        self.T138_tc = 0 #number of 1 3/8" thermocouple tees (all have 5/8" reduction on one side)
        self.T138_sg = 0 #number of 1 3/8" tees with sightglass
        self.T138_gage = 0 #number of 1 3/8" tees with 5/8" branch for pressure gage
        self.T138_pt = 0 #number of 1 3/8" tees for pressure transducers (all have 5/8" reduction for standoff)
        self.T138r58_oil = 0
        self.E78_l = 0 #number of 7/8" elbows (long)
        self.E78_s = 0 #number of 7/8" short elbows (less common than long)
        self.E78_45 = 0 #number of 7/8" 45 degree elbows
        self.E138 = 0 #number of 1 3/8" elbows
        self.rec1L = 0 #number of 1 Liter receivers
        self.psuhose = 0 #number of pump suction hoses
        self.pexhose = 0 #number of pump discharge hoses
        self.mflow = 0 #number of mass flow meters
        self.oilsep = 0 #number of oil separators
        #additional tube lengths and elements that are needed to compute volume but not for pressure drop (no flow through these elements)
        self.reducers_add = []
        self.len34_add = 0
        self.len78_add = 0
        self.len138_add = 0
        self.T78_add = 0
        self.T78_pt_add = 0
        self.T78_tc_add = 0
        self.T78r58_gage_add = 0
        self.T138_add = 0
        self.T138_tc_add = 0
        self.E138_add = 0
        self.ID = {
                    '158':1.5*0.0254,
                    '138':1.25*0.0254, #ID of a 1 3/8" tube [m]
                    '118':1*0.0254,
                    '78':0.75*0.0254, #ID of a 7/8" tube [m]
                    '34':0.625*0.0254, #ID of a 3/4" tube [m]
                    '12':0.375*0.0254, #ID of a 1/2" tube [m]
                    '14NPT':0.25*0.0254 #ID of a 1/4" NPT fitting [m]
                    }
        #define all the loss coefficients in a dictionary
        #loss coefficients are based on the Darcy friction factor
        self.K = {
                    'E_l':0.2,
                    'E_s':0.3,
                    'E_45':0.2,
                    'T_line':0.2,
                    'T_branch':1.0, #TODO: I could add values for ball valves and sight glasses
                    'mflowF050':11.76, #micromotion F050 mass flow meter, based on the velocity in a 7/8" OD tube
                    }
        self.K_redu = {
                    '78x12':0.4, #TODO: sudden contraction and expansion are assumed and read from a figure in Munson, Young, and Okiishi. This could be improved.
                    'rec_su':0.1 + 0.8, #contraction to 1/4"NPT opening and expansion inside
                    'rec_ex':0.5 + 0.1, #contraction to 1/4"NPT exit and expansion to 1/2" tube
                    '12x78':0.5,
                    '78x158':0.3,
                    '158x78':0.25,
                    '78x118':0.1,
                    '118x78':0.1,
                    '78x34':0.05,
                    '34x78':0.05,
                    '78x138':0.16,
                    '138x78':0.18,
                    'os_su':0.2+0.5, #internal elbow plus sudden expansion (neglecting filter mesh)
                    'os_ex':0.4, #sudden contraction
                    }
        #Load the parameters passed in
        # using the dictionary

        self.__dict__.update(kwargs)

    def OutputList(self):
        """
        Return a list of parameters for this component for further output
        It is a list of tuples, and each tuple is formed of items:
        [0] Description of value
        [1] Units of value
        [2] The value itself
        """
        return [
                ('T_ex','C',self.T_ex),
                ('P_ex','kPa',self.p_ex),
                ('delta_P','kPa',self.p_su - self.p_ex)
                ]

    def Re_1p_calc(self):
        """
        This method creates a dictionary property of the LineSet object with a
        Reynolds number for each diameter of tube in the set that involves flow
        (only inlet conditions are used to determine the Reynolds number)
        single-phase fluids only
        """
        self.Re_1p = {}
        self.V_1p = {}
        
        for size in self.ID:
            self.V_1p[size] = self.mdot/(self.rho*pi*self.ID[size]**2/4)
            self.Re_1p[size] = self.rho*self.V_1p[size]*self.ID[size]/self.mu_su
    
    def Re_f_calc(self):
        """
        Reynolds number for saturated liquid if it filled the entire length of tube
        """
        self.Re_f = {}
        V_f = {}
        
        for size in self.ID:
            V_f[size] = self.mdot/(self.rho_f*pi*self.ID[size]**2/4)
            self.Re_f[size] = self.rho_f*V_f[size]*self.ID[size]/self.mu_f
    
    def f_1p_calc(self):
        """
        find a single-phase Fanning friction factor for each diameter tube
        (two-phase condition returns liquid only value)
        """
        self.f_1p = {}
        
        if self.x_su < 0.0 or self.x_su > 1.0:

            self.Re_1p_calc()
            Re = self.Re_1p
        else:
            self.Re_f_calc()
            Re = self.Re_f
        
        for OD in Re:
            self.f_1p[OD] = Friction_Fanning(Re[OD], e_D=0)

    def Leq_calc(self):
        """
        This method creates a dictionary property of the LineSet object with an
        equivalent length for each diameter of tube in the set that involves flow
        (the equivalent length corresponds to use of the Fanning friction factor)
        """
        in_to_m = 0.0254
        self.Leq = {
                    '14NPT':0,
                    '12':0,
                    '34':self.len34*in_to_m, #start with the lengths of the straight tubes in [m]
                    '78':self.len78*in_to_m,
                    '118':0,
                    '138':self.len138*in_to_m,
                    '158':0
                    }
        self.f_1p_calc() #get the single-phase or liquid only friction factor
        
        #add elbow and tee equivalent lengths
        self.Leq['78'] += self.K['E_l']/4*self.ID['78']/self.f_1p['78']*self.E78_l
        self.Leq['138'] += self.K['E_l']/4*self.ID['138']/self.f_1p['138']*self.E138
        self.Leq['78'] += self.K['E_s']/4*self.ID['78']/self.f_1p['78']*self.E78_s
        self.Leq['78'] += self.K['E_45']/4*self.ID['78']/self.f_1p['78']*self.E78_45
        self.Leq['78'] += self.K['T_line']/4*self.ID['78']/self.f_1p['78']*(self.T78_line + self.T78_cp_line + self.T78_pt + self.T78r58_gage + self.T78r58_pt)
        self.Leq['138'] += self.K['T_line']/4*self.ID['138']/self.f_1p['138']*(self.T138_line + self.T138_gage + self.T138_pt + self.T138_sg)
        self.Leq['78'] += self.K['T_branch']/4*self.ID['78']/self.f_1p['78']*(self.T78_branch + self.T78_cp_branch + self.T78_tc + self.T78r58_oil + self.T78r58_tc)
        self.Leq['138'] += self.K['T_branch']/4*self.ID['138']/self.f_1p['138']*(self.T138_branch + self.T138_tc + self.T138r58_oil)
        #currently neglect additional losses due to sight glasses or open ball valves
        #add losses due to liquid receiver, mass flow meter, and oil separator (neglect pump suction and discharge hoses because pressure measurements are made outside them)
        if self.rec1L > 0:
            self.Leq['14NPT'] += (self.K_redu['rec_su'] + self.K_redu['rec_ex'])/4*self.ID['14NPT']/self.f_1p['14NPT']
        if self.mflow > 0:
            self.Leq['78'] += self.K['mflowF050']/4*self.ID['78']/self.f_1p['78']
        if self.oilsep > 0:
            self.Leq['138'] += (self.K_redu['os_su'] + self.K_redu['os_ex'])/4*self.ID['138']/self.f_1p['138']
        
        for reducer in self.reducers: #add reducer equivalent lengths
            if reducer.D1 == 0.5:
                self.Leq['12'] += reducer.D1length*in_to_m
                if reducer.D2==0.875:
                    self.Leq['12'] += self.K_redu['12x78']/4*self.ID['12']/self.f_1p['12']
                    self.Leq['78'] += reducer.D2length*in_to_m
            elif reducer.D1 == 0.75:
                self.Leq['34'] += reducer.D1length*in_to_m
                if reducer.D2==0.875:
                    self.Leq['34'] += self.K_redu['34x78']/4*self.ID['34']/self.f_1p['34']
                    self.Leq['78'] += reducer.D2length*in_to_m
            elif reducer.D1 == 0.875:
                self.Leq['78'] += reducer.D1length*in_to_m
                if reducer.D2==0.5:
                    self.Leq['12'] += self.K_redu['78x12']/4*self.ID['12']/self.f_1p['12']
                    self.Leq['12'] += reducer.D2length*in_to_m
                elif reducer.D2==0.75:
                    self.Leq['34'] += self.K_redu['78x34']/4*self.ID['34']/self.f_1p['34']
                    self.Leq['34'] += reducer.D2length*in_to_m
                elif reducer.D2==1.125:
                    self.Leq['78'] += self.K_redu['78x118']/4*self.ID['78']/self.f_1p['78']
                    self.Leq['118'] += reducer.D2length*in_to_m
                elif reducer.D2==1.375:
                    self.Leq['78'] += self.K_redu['78x138']/4*self.ID['78']/self.f_1p['78']
                    self.Leq['138'] += reducer.D2length*in_to_m
                elif reducer.D2==1.625:
                    self.Leq['78'] += self.K_redu['78x158']/4*self.ID['78']/self.f_1p['78']
                    self.Leq['158'] += reducer.D2length*in_to_m
            elif reducer.D1 == 1.125:
                self.Leq['118'] += reducer.D1length*in_to_m
                if reducer.D2==0.875:
                    self.Leq['78'] += self.K_redu['118x78']/4*self.ID['78']/self.f_1p['78']
                    self.Leq['78'] += reducer.D2length*in_to_m
            elif reducer.D1 == 1.375:
                self.Leq['138'] += reducer.D1length*in_to_m
                if reducer.D2==0.875:
                    self.Leq['78'] += self.K_redu['138x78']/4*self.ID['78']/self.f_1p['78']
                    self.Leq['78'] += reducer.D2length*in_to_m
            elif reducer.D1 == 1.625:
                self.Leq['158'] += reducer.D1length*in_to_m
                if reducer.D2==0.875:
                    self.Leq['78'] += self.K_redu['158x78']/4*self.ID['78']/self.f_1p['78']
                    self.Leq['78'] += reducer.D2length*in_to_m

    def Volume(self):
        ID138 = 1.25 #ID of a 1 3/8" tube
        ID78 = 0.75 #ID of a 7/8" tube
        ID34 = 0.625 #ID of a 3/4" tube
        ID58 = 0.5 #ID of a 5/8" tube
        ID14 = 0.125 #ID of a 1/4" tube
        
        A138 = pi*ID138**2/4
        A78 = pi*ID78**2/4 #volume per unit length of 7/8" tube
        A34 = pi*ID34**2/4 #volume per unit length of 3/4" tube
        A58 = pi*ID58**2/4 #volume per unit length of 5/8" tube
        A14 = pi*ID14**2/4 #volume per unit length of 1/4" tube
        
        vT78 = A78*(2.5 + 0.75) #volume of a 7/8" tee (lines go all the way inside for most of the tee so we assume ID is the same as ID of the line
        vT78_tc = vT78 + A78*1.5 #volume of a 7/8" thermocouple tee
        vT78_pt = vT78 + A78*1.75 #volume of a 7/8" pressure transducer tee
        vT78_cp = vT78 + A78*1.25 #voluem of a 7/8" charge port tee
        vT78r58_oil = A78*(2 + 0.75) + A58*0.75
        vT78r58_tc = A78*(2 + 0.75) + A58*(0.75 + 1.75) #volume of this reducing tee for thermocouple, including extension (abrupt reducion assumed for simplicity)
        vT78r58_pt = A78*2 + A58*(0.5 + 4.75) #volume of reducing tee for pressure transducer, including standoff
        vT78r58_gage = A78*2 + A58*(0.5 + 1.75) #shorter standoff than pressure transducer tee
        vT138 = A138*(3.375 + 1) #volume of 1 3/8" tee
        vT138_sg = vT138 + A138*0.375 #volume of a 1 3/8" tee with sightglass
        vT138_gage = A138*2.875 + A58*3.25 #(includes standoff) all 1 3/8" tees used for measurements have a reduction to 5/8" on one side
        vT138_pt = A138*2.875 + A58*(0.5 + 4.75) #(includes standoff) in 5/8" line
        vT138r58_oil = A138*(2.8 + 1) + A58*0.625 + pi/3*1/4*0.625*(ID138**2 + ID138*ID58 + ID58**2) #volume of the reducing tee that connects to the oil pump bypass
        vT138_tc = vT138r58_oil + A58*1.75 #same as reducing tee for oil junction, additional length for thermocouple
        vE78_l = A78*3.5 #volume of a long 7/8" elbow
        vE78_s = A78*3 #volume of a short 7/8" elbow
        vE78_45 = A78*2.5 #volume of a 7/8" 45 degree elbow
        vE138 = A138*5.5 #volume of a 1 3/8" elbow
        vrec1L = 61.0237440947 #volume of a 1 liter receiver in in^3
        vpsuhose = pi*1.125**2/4*35.6
        vpexhose = pi*0.5**2/4*72
        vmflow = 5.5671 #based on 0.0241 gal reported by micromotion tech support
        voilsep = pi*4**2/4*27.25 #internal volume of oil separator (neglecting wall thickness, internal tubes, baffles, and filter
        
        #add additional volumes that are not involved in flow or pressure drop
        reducers = self.reducers + self.reducers_add
        len34 = self.len34 + self.len34_add
        len78 = self.len78 + self.len78_add
        len138 = self.len138 + self.len138_add
        T78 = self.T78_line + self.T78_branch + self.T78_add
        T78_pt = self.T78_pt + self.T78_pt_add
        T78_tc = self.T78_tc + self.T78_tc_add
        T78_cp = self.T78_cp_line + self.T78_cp_branch
        T78r58_gage = self.T78r58_gage + self.T78r58_gage_add
        T138 = self.T138_line + self.T138_branch + self.T138_add
        T138_tc = self.T138_tc + self.T138_tc_add
        E138 = self.E138 + self.E138_add
        
        vol_reducers = 0
        for piece in reducers:
            vol_reducers += piece.volume() #add up the volume of any reducers
        #end for
        
        volume = (self.len14*A14 + len34*A34 + len78*A78 + len138*A138 + #straight tube lengths
                T78*vT78 + T78_tc*vT78_tc + T78_pt*vT78_pt + T78_cp*vT78_cp + #7/8" tees
                self.T78r58_oil*vT78r58_oil + self.T78r58_tc*vT78r58_tc + self.T78r58_pt*vT78r58_pt + T78r58_gage*vT78r58_gage +
                #7/8" to 5/8" reducing tees
                T138*vT138 + self.T138_sg*vT138_sg + self.T138_gage*vT138_gage + self.T138r58_oil*vT138r58_oil + T138_tc*vT138_tc
                + self.T138_pt*vT138_pt + #1 3/8" tees
                self.E78_l*vE78_l + self.E78_s*vE78_s + self.E78_45*vE78_45 + E138*vE138 + #all elbows
                self.rec1L*vrec1L + self.psuhose*vpsuhose + self.pexhose*vpexhose + self.mflow*vmflow + self.oilsep*voilsep +
                #miscellaneous
                vol_reducers) #all reducers
        
        return volume*1/39.3700787402**3 #convert from in^3 to m^3

    def Update(self,**kwargs):
        #Load the parameters passed in
        # using the dictionary
        self.__dict__.update(kwargs)
        
    # def OutputList(self):
    # """
    # Return a list of parameters for this component for further output
    #
    # It is a list of tuples, and each tuple is formed of items:
    # [0] Description of value
    # [1] Units of value
    # [2] The value itself
    # """
    # return [
    # ('Length of tube','m',self.L),
    # ('Supply line OD','m',self.OD),
    # ('Supply line ID','m',self.ID),
    # ('Tube Conductivity','W/m-K',self.k_tube),
    # ('Insulation thickness','m',self.t_insul),
    # ('Insulation conductivity','W/m-K',self.k_insul),
    # ('Air overall HTC','W/m^2-K',self.h_air),
    # ('Air Temperature','K',self.T_air),
    # ('Q Total','W',self.Q),
    # ('Pressure drop ','Pa',self.DP),
    # ('Reynolds # Fluid','-',self.Re_fluid),
    # ('Mean HTC Fluid','W/m^2-K',self.h_fluid),
    # ('Charge','kg',self.Charge),
    # ('Inlet Temperature','K',self.Tin),
    # ('Outlet Temperature','K',self.Tout)
    # ]

    def Calculate(self):
        #The LineSet must have its inlet pressure and enthalpy specified before the Calculate method will work
        #find inlet temperature
        self.T_su = Props('T','P',self.p_su, 'H',self.h_su/1000, self.Ref) - 273.15
        #find vapor quality so we know what density to use
        self.x_su = Props('Q','P',self.p_su, 'H',self.h_su/1000, self.Ref)
        
        #Heat Transfer Calculations
        #assume no heat transfer or pressure drop for first cut
        self.h_ex = self.h_su
        # self.f_fluid, self.h_fluid, self.Re_fluid=f_h_1phase_Tube(self.mdot, self.ID, self.Tin, self.pin, self.Ref)
        # # Specific heat capacity [J/kg-K]
        # cp=Props('C','T',self.Tin,'P',self.pin,self.Ref)*1000
        # #Thermal resistance of tube
        # R_tube=log(self.OD/self.ID)/(2*pi*self.L*self.k_tube)
        # #Thermal resistance of insulation
        # R_insul=log((self.OD+2.0*self.t_insul)/self.OD)/(2*pi*self.L*self.k_insul);
        # #Convective UA for inside the tube
        # UA_i=pi*self.ID*self.L*self.h_fluid;
        # #Convective UA for the air-side
        # UA_o=pi*(self.OD+2*self.t_insul)*self.L*self.h_air;
        #
        # #Avoid the possibility of division by zero if h_air is zero
        # if UA_o<1e-12:
        # UA_o=1e-12
        #
        # #Overall UA value
        # UA=1/(1/UA_i+R_tube+R_insul+1/UA_o)
        #
        # #Outlet fluid temperature [C]
        ## self.Tout=self.T_air-exp(-UA/(self.mdot*cp))*(self.T_air-self.Tin)
        # #first, assume no temperature drop/rise in lines
        # self.Tout = self.Tin
        # #Overall heat transfer rate [W]
        # self.Q=self.mdot*cp*(self.Tout-self.Tin)
        # self.hout=self.hin+self.Q/self.mdot
        #
        # #Pressure drop calculations for superheated refrigerant
        # v=1./rho
        # G=self.mdot/(pi*self.ID**2/4.0)
        # #Pressure gradient using Darcy friction factor
        # dpdz=-self.f_fluid*v*G**2/(2.*self.ID) #Pressure gradient
        # self.DP=dpdz*self.L
        
        #find exit temperature and quality after heat transfer without pressure drop
        self.T_ex = Props('T','P',self.p_su, 'H',self.h_ex/1000, self.Ref) - 273.15
        self.x_ex = Props('Q','P',self.p_su, 'H',self.h_ex/1000, self.Ref)
        
        # Density for charge calculation [kg/m^3]
        T_mean = (self.T_su + self.T_ex)/2
        
        if (self.x_su >= 1) or (self.x_su <= 0): #we have a single-phase state
            self.rho = Props('D','T',T_mean+273.15, 'P',self.p_su, self.Ref)
            self.mu_su = Props('V','T',T_mean+273.15, 'D',self.rho, self.Ref) #kg/m-s
            self.Leq_calc() #compute all the equivalent lengths we need for the pressure drop calculation
            #compute the pressure drop for incompressible single-phase flow using the Fanning friction factor
            DP = 0
            for size in self.Leq:
                DP += self.f_1p[size]*self.Leq[size]/self.ID[size]*2*self.rho*self.V_1p[size]**2
        else:
            xmin = min(self.x_su, self.x_ex)
            xmax = max(self.x_su, self.x_ex)
            if xmin - xmax < 1e-5:
                xmax = xmin
            
            Tdew = Props('T','P',self.p_su, 'Q',1.0, self.Ref)
            Tbubble = Props('T','P',self.p_su, 'Q',0.0, self.Ref)
            
            #Calculate the transport properties once
            satTransport={}
            self.rho_f = Props('D','T',Tbubble,'Q',0.0,self.Ref)
            self.mu_f = Props('V','T',Tbubble,'D',self.rho_f,self.Ref)
            satTransport['v_f']=1/self.rho_f
            satTransport['v_g']=1/Props('D','T',Tdew,'Q',1.0,self.Ref)
            satTransport['mu_f']=self.mu_f
            satTransport['mu_g']=Props('V','T',Tdew,'Q',1.0,self.Ref)
            
            self.Leq_calc() #compute all the equivalent lengths we need for the pressure drop calculation
            #(equivalent length is the length that would give the same pressure drop for liquid only in the line.
            #We then use that length with the Lockhart-Martinelli pressure gradient)
            
            G = {}
            DPfrict = 0
            DPaccel_dict = {}
            DPaccel_avg_num = 0
            DPaccel_avg_den = 0
            for size in self.Leq:
                #if I compute heat transfer first and have multiple diameters in the LineSet,
                #this is a bit weird because it assumes that each portion at each diameter experiences the entire quality change.
                G[size] = self.mdot/(pi*self.ID[size]**2/4)
                DPfrict += LMPressureGradientAvg(xmin,xmax,self.Ref,G[size],self.ID[size],Tbubble,Tdew,C=None,satTransport=satTransport)*self.Leq[size]
                DPaccel_dict[size] = AccelPressureDrop(xmin,xmax,self.Ref,G[size],Tbubble,Tdew,rhosatL=1/satTransport['v_f'],rhosatV=1/satTransport['v_g'],slipModel='Zivi')
        #do a weird weighted average of the acceleration pressure drop here since we also compute it for each size of line in the LineSet as though it had the entire quality change.
                DPaccel_avg_num += DPaccel_dict[size]*self.Leq[size]
                DPaccel_avg_den += self.Leq[size]
            
            DPaccel = DPaccel_avg_num/DPaccel_avg_den
            DP = DPfrict + DPaccel
        
        self.p_ex = self.p_su - DP/1000 #convert DP from Pa to kPa
        
        #compute some final exit properties after temperature and pressure change in the line
        self.T_ex = Props('T','P',self.p_ex, 'H',self.h_ex/1000, self.Ref) - 273.15
        self.x_ex = Props('Q','P',self.p_ex, 'H',self.h_ex/1000, self.Ref)
        
        #update density using arithmetic mean pressure if it is vapor
        if (self.x_su >= 1) or (self.x_su <= 0): #we have a single-phase state
            p_mean = (self.p_su + self.p_ex)/2 #not sure if using this really matters or is consistent
            self.rho = Props('D','T',T_mean+273.15, 'P',p_mean, self.Ref)
        else:
            self.rho = TwoPhaseDensity(self.Ref, xmin,xmax, Tdew,Tbubble, slipModel='Zivi')
        
        #Charge in Line set [kg]
        self.Charge = self.rho*self.vol_int

class reducer(object):
    def __init__(self, D1,D1length, D2,D2length, length_trans, thick):
        self.D1 = D1
        self.D1length = D1length #length for which the reducer has this diameter
        self.D2 = D2
        self.D2length = D2length #length for which the reducer has this diameter
        self.length_trans = length_trans #length of diameter transition
        self.thick = thick #tube wall thickness
    def volume(self):
        ID1 = self.D1 - 2*self.thick
        ID2 = self.D2 - 2*self.thick
        volume = pi/4*(ID1**2*self.D1length + ID2**2*self.D2length) + pi/3*1/4*self.length_trans*(ID1**2 + ID1*ID2 + ID2**2)
        return volume

def SampleLineSets():
    
    ExpEx_RegVapSu = LineSetClass()
    RegVapEx_CondSu = LineSetClass()
    CondEx_PumpSu = LineSetClass()
    PumpEx_RegLiqSu = LineSetClass()
    RegLiqEx_EvapSu = LineSetClass()
    EvapEx_ExpSu = LineSetClass()

    #-----------------------------------------
    #-----------------------------------------
    # Lineset from exp_ex through oilsep to regen_vap_su
    #-----------------------------------------
    #-----------------------------------------

    reducerlist = [
                    reducer(D1=7/8,D1length=0.875, D2=(1+3/8),D2length=1, length_trans=0.25, thick=1/16)
                    ]
    reducerlist_add = [
                        reducer(D1=3/4,D1length=0, D2=7/8,D2length=0, length_trans=1.75, thick=1/16),
                        reducer(D1=3/4,D1length=0, D2=7/8,D2length=0, length_trans=1.75, thick=1/16),
                        reducer(D1=7/8,D1length=0.875, D2=(1+3/8),D2length=1, length_trans=0.25, thick=1/16)
                    ]
    params={
            'len138':31.1875 + 35.375,
            'len78':0.375,
            'len14':37.0 + 19.6050880621,
            'E138':2+1,
            'E78_l':3,
            'T138_line':2+1,
            'T138_gage':1,
            'T138_pt':1,
            'T138_sg':1+1,
            'T138_tc':1,
            'T138r58_oil':1,
            'T78_pt':1,
            'oilsep':1,
            'reducers':reducerlist,
            'len138_add':4.625 + 53,
            'len78_add':41.875,
            'len34_add':3.5,
            'T78_tc_add':1,
            'E138_add':2,
            'T138_add':1,
            'T138_tc_add':1,
            'reducers_add':reducerlist_add,
            }
    ExpEx_RegVapSu.Update(**params)
    #compute the internal volume of the lineset since it is a constant
    ExpEx_RegVapSu.vol_int = ExpEx_RegVapSu.Volume()
    #-----------------------------------------
    #-----------------------------------------
    # Lineset from regen_vap_ex to cond_su
    #-----------------------------------------
    #-----------------------------------------
    reducerlist = [
                    reducer(D1=(1+3/8),D1length=1, D2=7/8,D2length=0.875, length_trans=0.25, thick=1/16),
                    reducer(D1=7/8,D1length=0, D2=(1+1/8),D2length=1.125, length_trans=2, thick=1/16)
]
    params={
            'len138':55.5625,
            'len78':7.875,
            'E138':1,
            'T138_line':1,
            'T138_pt':1,
            'T138_sg':1,
            'T138_tc':2,
            'T78_pt':2,
            'reducers':reducerlist,
            'len138_add':2.375,
            }
    RegVapEx_CondSu.Update(**params)
    #compute the internal volume of the lineset since it is a constant
    RegVapEx_CondSu.vol_int = RegVapEx_CondSu.Volume()
    #-----------------------------------------
    #-----------------------------------------
    # Lineset from cond_ex to pump_su
    #-----------------------------------------
    #-----------------------------------------
    reducerlist = [
                    reducer(D1=7/8,D1length=0, D2=1/2,D2length=0, length_trans=4, thick=1/16),
                    reducer(D1=1/2,D1length=0, D2=7/8,D2length=0, length_trans=6.75, thick=1/16)
                ]
    params={
            'len78':32.9375,
            'E78_l':4,
            'E78_s':1,
            'T78_cp_line':1,
            'T78_pt':2,
            'T78_tc':2,
            'psuhose':1,
            'rec1L':1,
            'reducers':reducerlist,
            }
    CondEx_PumpSu.Update(**params)
    #compute the internal volume of the lineset since it is a constant
    CondEx_PumpSu.vol_int = CondEx_PumpSu.Volume()
    #-----------------------------------------
    #-----------------------------------------
    # Lineset from pump_ex to regen_liq_su
    #-----------------------------------------
    #-----------------------------------------
    reducerlist = [
                    reducer(D1=7/8,D1length=0, D2=1+5/8,D2length=0, length_trans=2.75, thick=1/16)
                ]
    params={
            'len78':95.5625,
            'E78_l':4,
            'E78_s':1,
            'T78_cp_branch':1,
            'T78_pt':4,
            'T78_tc':1,
            'T78r58_pt':1,
            'T78r58_tc':1,
            'pexhose':1,
            'mflow':1,
            'reducers':reducerlist,
            }
    PumpEx_RegLiqSu.Update(**params)
    #compute the internal volume of the lineset since it is a constant
    PumpEx_RegLiqSu.vol_int = PumpEx_RegLiqSu.Volume()
    #-----------------------------------------
    #-----------------------------------------
    # Lineset from regen_liq_ex to evap_su
    #-----------------------------------------
    #-----------------------------------------
    reducerlist = [
                    reducer(D1=1+5/8,D1length=0, D2=7/8,D2length=0, length_trans=2.75, thick=1/16),
                    reducer(D1=7/8,D1length=0, D2=(1+1/8),D2length=0, length_trans=3.25, thick=1/16)
                    ]
    params={
            'len78':79.625,
            'len14':42.6050880621,
            'E78_45':1,
            'E78_l':2,
            'E78_s':1,
            'T78_pt':1,
            'T78_tc':1,
            'T78r58_pt':1,
            'T78r58_tc':1,
            'reducers':reducerlist,
            }
    RegLiqEx_EvapSu.Update(**params)
    #compute the internal volume of the lineset since it is a constant
    RegLiqEx_EvapSu.vol_int = RegLiqEx_EvapSu.Volume()
    #-----------------------------------------
    #-----------------------------------------
    # Lineset from evap_ex to exp_su
    #-----------------------------------------
    #-----------------------------------------
    reducerlist = [
                    reducer(D1=(1+1/8),D1length=0, D2=7/8,D2length=0, length_trans=3.25, thick=1/16)
                    ]
    reducerlist_add = [
                        reducer(D1=7/8,D1length=0, D2=3/4,D2length=0, length_trans=1.75, thick=1/16)
                    ]
    params={
            'len78':71,
            'len14':66.25,
            'E78_l':3,
            'T78_pt':2,
            'T78_tc':2,
            'T78r58_oil':1,
            'reducers':reducerlist,
            'len78_add':10.75,
            'len34_add':3.25,
            'T78_add':1,
            'T78_pt_add':1,
            'T78r58_gage_add':1,
            'reducers_add':reducerlist_add,
            }
    EvapEx_ExpSu.Update(**params)
    #compute the internal volume of the lineset since it is a constant
    EvapEx_ExpSu.vol_int = EvapEx_ExpSu.Volume()
    return (PumpEx_RegLiqSu, ExpEx_RegVapSu, RegLiqEx_EvapSu, EvapEx_ExpSu, RegVapEx_CondSu, CondEx_PumpSu)

def Solve_LineSets(Inputs, PumpEx_RegLiqSu, ExpEx_RegVapSu, RegLiqEx_EvapSu, EvapEx_ExpSu, RegVapEx_CondSu, CondEx_PumpSu):
    params={
            'p_su':Inputs['P_pump_ex'],
            'h_su':Props('H','T',Inputs['T_pump_ex']+273.15,'P',Inputs['P_pump_ex'],Inputs['Ref'])*1000,
            'mdot':Inputs['mdot_ref'],
            'Ref':Inputs['Ref']
            }
    PumpEx_RegLiqSu.Update(**params)
    PumpEx_RegLiqSu.Calculate()
    params={
            'p_su':Inputs['P_pump_ex'],
            'h_su':Props('H','T',Inputs['T_pump_ex']+273.15,'P',Inputs['P_pump_ex'],Inputs['Ref'])*1000,
            'mdot':Inputs['mdot_ref'],
            'Ref':Inputs['Ref']
            }
    ExpEx_RegVapSu.Update(**params)
    ExpEx_RegVapSu.Calculate()
    params={
            'p_su':Inputs['P_pump_ex'],
            'h_su':Props('H','T',Inputs['T_pump_ex']+273.15,'P',Inputs['P_pump_ex'],Inputs['Ref'])*1000,
            'mdot':Inputs['mdot_ref'],
            'Ref':Inputs['Ref']
            }
    RegLiqEx_EvapSu.Update(**params)
    RegLiqEx_EvapSu.Calculate()
    params={
            'p_su':Inputs['P_pump_ex'],
            'h_su':Props('H','T',Inputs['T_pump_ex']+273.15,'P',Inputs['P_pump_ex'],Inputs['Ref'])*1000,
            'mdot':Inputs['mdot_ref'],
            'Ref':Inputs['Ref']
            }
    EvapEx_ExpSu.Update(**params)
    EvapEx_ExpSu.Calculate()
    params={
            'p_su':Inputs['P_pump_ex'],
            'h_su':Props('H','T',Inputs['T_pump_ex']+273.15,'P',Inputs['P_pump_ex'],Inputs['Ref'])*1000,
            'mdot':Inputs['mdot_ref'],
            'Ref':Inputs['Ref']
            }
    RegVapEx_CondSu.Update(**params)
    RegVapEx_CondSu.Calculate()
    params={
            'p_su':Inputs['P_cond_ex'],
            'h_su':Props('H','T',Inputs['T_cond_ex']+273.15,'P',Inputs['P_cond_ex'],Inputs['Ref'])*1000,
            'mdot':Inputs['mdot_ref'],
            'Ref':Inputs['Ref']
            }
    CondEx_PumpSu.Update(**params)
    CondEx_PumpSu.Calculate()

if __name__=='__main__':
    # Ref = 'r134a.fld'
    # mdot = 0.14 #[kg/s]
    # p_cond_ex = 600 #[kPa]
    # T_cond_ex = 13 #[C]
    # h_cond_ex = Props('H','T',T_cond_ex+273.15,'P',p_cond_ex,Ref)*1000
    #
    # p_regen_vap_ex = 610 #[kPa]
    # x_regen_vap_ex = 0.9
    # h_regen_vap_ex = Props('H','P',p_regen_vap_ex,'Q',x_regen_vap_ex,Ref)*1000
    #
    # p_evap_ex = 2000 #[kPa]
    # T_evap_ex = 100 #[C]
    # h_evap_ex = Props('H','T',T_evap_ex+273.15,'P',p_evap_ex,Ref)*1000
    #Set Fluids
    old = ''
    SetFluids(['r134a.fld|'+old+'r245fa.fld','water.fld'])
    directory = '../Python Cycle Thesis/'
    # grouplist = ['R134a','R245fa','ZRC']
    # filtername = ''
    # filtername = '_qualfilter_evap'
    # filtername = '_qualfilter_cond'
    # filtername = '_qualfilter_regen'
    # filtername = '_qualfilter_unc_evap'
    # filtername = '_qualfilter_unc_cond'
    # filtername = '_qualfilter_uncSH_cond'
    # filtername = '_qualfilter_uncvap_regen'
    # filtername = '_qualfilter_uncvapSH_regen'
    # filtername = '_qualfilter_uncliq_regen'
    # store_134a = DataHolderFromFile(directory+'store_'+grouplist[0]+filtername+'.csv')
    # store_245fa = DataHolderFromFile(directory+'store_'+grouplist[1]+filtername+'.csv')
    # store_ZRC = DataHolderFromFile(directory+'store_'+grouplist[2]+filtername+'.csv')
    # storelist = [store_134a, store_245fa, store_ZRC]
    grouplist = ['R245fa']
    filtername = '_CondEx_PumpSu'
    store_245fa = DataHolderFromFile('store_'+grouplist[0]+filtername+'.csv')
    storelist = [store_245fa]
    startpos = 0
    stoppos = None
    # storelist = [store_134a]
    # grouplist = ['R134a']
    # startpos = 0
    # stoppos = 1
    # storelist = [store_245fa]
    # grouplist = ['R245fa']
    # startpos = 0
    # stoppos = None
    # storelist = [store_ZRC]
    # grouplist = ['ZRC']
    # startpos = 0
    # stoppos = None
    #Generate the cycle class with all constant parameters
    LineSets = SampleLineSets()
    time0 = time.time()
    i = -1
    for store in storelist:
        i += 1
        numpoints = len(store['fluid'][startpos:stoppos])
        ziplist = [store['fluid'][startpos:stoppos],
        store['mdot_ref'][startpos:stoppos],
        store['P_pump_ex'][startpos:stoppos],
        store['T_pump_ex'][startpos:stoppos],
        store['P_cond_ex'][startpos:stoppos],
        store['T_cond_ex'][startpos:stoppos]]
        run = startpos - 1 #counts the number of run we are on
        for Ref,mdot_ref,P_pump_ex,T_pump_ex,P_cond_ex,T_cond_ex in zip(*ziplist):
            print Ref,mdot_ref,P_pump_ex,T_pump_ex,P_cond_ex,T_cond_ex
            run += 1
            Inputs={
                    'Ref':old+Ref, #working fluid
                    'mdot_ref':mdot_ref,
                    'P_pump_ex':P_pump_ex,
                    'T_pump_ex':T_pump_ex,
                    'P_cond_ex':P_cond_ex,
                    'T_cond_ex':T_cond_ex
                    }
            print run
            label = 'CondEx_PumpSu'
            qualifier = ''
            Solve_LineSets(Inputs, *LineSets)
            if numpoints > 1:
                Write2CSV(LineSets[5],'LineSet'+'_'+label+'_'+grouplist[i]+'.csv', append=run>0)
            else:
                print LineSets[5].OutputList()
    time1 = time.time()
    print 'All runs executed in', (time1 - time0)/60., 'minutes.'