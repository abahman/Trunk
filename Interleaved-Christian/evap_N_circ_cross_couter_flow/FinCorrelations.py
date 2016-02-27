from CoolProp.HumidAirProp import HAProps#,UseVirialCorrelations,UseIsothermCompressCorrelation,UseIdealGasEnthalpyCorrelations
from math import sqrt,pi,log,tanh,exp,cos,log
from ACHPTools import ValidateFields
#from math import isnan
#Turn on virial correlations for air and water for speed in Humid Air routines
#UseVirialCorrelations(1)
#UseIsothermCompressCorrelation(1)
#UseIdealGasEnthalpyCorrelations(1)

def IsFinsClass(Fins):
    '''
    Returns the Fins class if Fins is an instance of the FinInputs class, False otherwise
    
    Convenience function for the Validator 
    '''
    if isinstance(Fins,FinInputs):
        return Fins
    else:
        return False
    
class FinsVals:
    pass    
class TubesVals:
    pass
class AirVals:
    pass
        
class FinInputs():
    """ 
    Empty Class for fin data
    """
    def __init__(self):
        #Don't do anything
        self.Tubes=TubesVals()
        self.Air=AirVals()
        self.Fins=FinsVals()
    def __repr__(self):
        string="Tubes::\n"
        for field in self.Tubes.__dict__.keys():
            string+=field+": "+repr(self.Tubes.__dict__[field])+"\n"
        string+="Fins::\n"
        for field in self.Fins.__dict__.keys():
            string+=field+": "+repr(self.Fins.__dict__[field])+"\n"
        string+="Air::\n"
        for field in self.Air.__dict__.keys():
            string+=field+": "+repr(self.Air.__dict__[field])+"\n"
        return string
    def Validate(self):
        """
        Check that all required fields are included, and no extra fields not listed here are added
        
        This is quite strict, but for the best to avoid typos
        """
        reqFields=[
            ('RH',float,0,1),
            ('Tdb',float,-80+273.15,200+273.15),
            ('FanPower',float,0,4000),
            ('p',float,0.01,10000),
            ('Vdot_ha',float,0.001,10)
        ]
        optFields=['RHmean','Tmean']
        d=dict(self.Air.__dict__) #The current values
        ValidateFields(d,reqFields,optFields)
        
        reqFields=[
            ('FPI',float,0.1,100),
            ('Pd',float,0.000001,0.1),
            ('xf',float,0.000001,0.1),
            ('t',float,0.00001,0.01),
            ('k_fin',float,0.01,10000)
        ]
        optFields=None
        d=dict(self.Fins.__dict__) #The current values
        ValidateFields(d,reqFields,optFields)
        
        reqFields=[
            ('NTubes_per_bank',int,0.1,100),
            ('Ncircuits',int,1,50),
            ('Nbank',float,1,50),
            ('Ltube',float,0.001,10),
            ('ID',float,0.0001,1),
            ('OD',float,0.0001,1),
            ('Pl',float,0.0001,1),
            ('Pt',float,0.0001,1)
        ]
        optFields=None
        d=dict(self.Tubes.__dict__) #The current values
        ValidateFields(d,reqFields,optFields)
        
def WavyLouveredFins(Inputs):
    """
    # Correlations from:
    # Chi-Chuan Wang and Yu-Min Tsai and Ding-Chong Lu, 1998, "Comprehensive
    # Study of Convex-Louver and Wavy Fin-and-Tube Heat Exchangers", Journal 
    # of Thermophysics and Heat Transfer
    
    || -    xf    ->
    ^              ==                          ==
    |           ==  |  ==                  ==
    Pd       ==     |     ==            ==
    |     ==        |        ==     ==
    =  ==           s             ==  
                    |
                    |
                    |
                   ==                        ==
                ==     ==                 ==
             ==           ==           ==
          ==                 ==     ==
       ==                        ==  
    
     t: thickness of fin plate
     Pf: fin pitch (centerline-to-centerline distance between fins)
     Pd: indentation for waviness (not including fin thickness)
     s: fin spacing (free space between fins) = Pf-t
    
    
    
                 |--       Pl      -|
                ___                 |
              /     \               |
       =     |       |              |
       |      \ ___ /               |
       |                            |
       |                           ___      
       |                         /     \      |
      Pt                        |       |     D
       |                         \ ___ /      |
       |    
       |        ___
       |      /     \
       =     |       |
              \ ___ /
    """
    
    Ntubes_bank = Inputs.Tubes.NTubes_per_bank #tubes per bank
    Nbank =       Inputs.Tubes.Nbank       #Number of banks
    Ltube =       Inputs.Tubes.Ltube       #length of a single tube
    D =           Inputs.Tubes.OD          #Outer diameter of tube
    Pl =          Inputs.Tubes.Pl          #Horizontal spacing between banks (center to center)
    Pt =          Inputs.Tubes.Pt          #Vertical spacing between tubes in a bank (center to center)

    FPI =         Inputs.Fins.FPI
    pd =          Inputs.Fins.Pd  #2 * amplitude of fin
    xf =          Inputs.Fins.xf  #1/2 period of fin
    t =           Inputs.Fins.t    #thicknes of fin material
    k_fin =       Inputs.Fins.k_fin

    Vdot_ha =     Inputs.Air.Vdot_ha
    p =           Inputs.Air.p
    
    if hasattr(Inputs,'Tmean'):
        Temp = Inputs.Air.Tmean
    else:
        Temp = Inputs.Air.Tdb
    
    if hasattr(Inputs,'RHmean'):
        RHin = Inputs.Air.RHmean
    else:
        RHin = Inputs.Air.RH

    # Check that cs_cp is defined, if so, set it to the value passed in
    if (hasattr(Inputs,'cs_cp') and Inputs.cs_cp>0) or (hasattr(Inputs,'WetDry') and Inputs.WetDry=='Wet'):
        isWet=True
        cs_cp=Inputs.Air.cs_cp
    else:
        isWet=False
        cs_cp=1.0

    #Film temperature [K]
    Tfilm = Temp
    #Fins per meter [1/m]
    FPM = FPI / 0.0254
    #Fin pitch (distance between centerlines of fins)
    pf = 1 / FPM
    #Spacing between fins
    s = 1 / FPM - t

    #Height of heat exchanger [m]
    Height = Pt * (Ntubes_bank)  #assuming that fin extends 1/2 pt above/below last tube in bundle
    #A_duct is the face area [m^2] equivalent to the duct cross-section
    A_duct = Height * Ltube  #neglecting the additional height of the fins above/below the last tubes in the bundle
    #Number of fins in the tube sheet [-]
    Nfin = Ltube * FPM
    #Secant of theta is the area enhancement factor [-]
    #  It captures the increase in area due to the waviness of the fins 
    sec_theta = sqrt(xf*xf + pd*pd) / xf
    # Duct cross-sectional area that is not fin or tube [m^2]
    Ac = A_duct - t * Nfin * (Height-D*Ntubes_bank) - Ntubes_bank * D * Ltube
    # Total outer area of the tubes [m^2]
    Atube = Ntubes_bank * Nbank * pi * D * Ltube
    #Wetted Area of a single fin [m^2]
    A_1fin = 2.0 * (Height * Pl * (Nbank+1) * sec_theta  - Ntubes_bank*Nbank * pi*D*D/4) #assuming that fin extends 1/2 pt in front/after last tube in bundle
    # Total wetted area of the fins [m^2]
    Af = Nfin * A_1fin
    #Total area including tube and fins [m^2]
    A = Af + Ntubes_bank * Nbank * pi * D * (Ltube-Nfin*t)

    #Evaluate the mass flow rate based on inlet conditions
    # To convert a parameter from per kg_{humid air} to per kg_{dry air}, divide by (1+W)
    W=HAProps('W','T',Inputs.Air.Tdb,'P',p,'R',Inputs.Air.RH)
    v_da=HAProps('V','T',Inputs.Air.Tdb,'P',p,'W',W)
    h_da=HAProps('H','T',Inputs.Air.Tdb,'P',p,'W',W)*1000
    rho_ha = 1 / v_da*(1+W) #[m^3/kg_ha]
    rho_da = 1 / v_da #[m^3/kg_da]
    mdot_ha = Vdot_ha * rho_ha #[kg_ha/s]
    mdot_da = Vdot_ha * rho_da #[kg_da/s]

    umax = mdot_ha / (rho_ha * Ac) #[m/s]

    #Use a forward difference to calculate cp from cp=dh/dT
    dT=0.0001 #[K]
    cp_da=(HAProps('H','T',Inputs.Air.Tdb+dT,'P', p, 'W',W)*1000-h_da)/dT #[J/kg_da/K]
    #if isnan(cp_da):
    #    print "error in Fin Correlations, cp_da is a 'nan'","iputs are :",Inputs.Air.Tdb,p,W,h_da,Inputs.Air.RH,"result of property function call is ",HAProps('H','T',Inputs.Air.Tdb+dT,'P', p, 'W',W)
    cp_ha=cp_da/(1+W) #[J/kg_ha/K]
    
    #Transport properties of humid air from CoolProp
    mu_ha=HAProps('M','T',Inputs.Air.Tdb,'P',p,'W',W)
    k_ha=HAProps('K','T',Inputs.Air.Tdb,'P',p,'W',W)*10**3
    
    #Dimensionless Groups
    Pr = cp_ha * mu_ha / k_ha
    Re_D = rho_ha * umax * D / mu_ha

    #Heat transfer
    j = 16.06 * pow(Re_D,-1.02 * (pf / D) - 0.256) * pow(A / Atube, -0.601) * pow(Nbank,-0.069) * pow(pf / D,0.84) #Colburn j-Factor
    h_a = j * rho_ha * umax * cp_ha / pow(Pr,2.0/3.0) #air side mean heat transfer coefficient
    h_a =h_a*Inputs.h_tp_tuning; #print 'wlfins - Inputs.h_tp_tuning ',Inputs.h_tp_tuning
    if h_a<0.00001:
        print "warning, h_a in FinCorrelations, Wavy Louvered fins was smaller 0: ",h_a," set to 0.00001"
        h_a=0.00001
    #air side pressure drop correlations
    if (Re_D<1e3):
        fa_total=0.264*(0.105+0.708*exp(-Re_D/225.0))*pow(Re_D,-0.637)*pow(A/Atube,0.263)*pow(pf/D,-0.317)
    else:
        fa_total=0.768*(0.0494+0.142*exp(-Re_D/1180.0))*pow(A/Atube,0.0195)*pow(pf/D,-0.121)
    
    #calcs needed for specific fin types
    r = D / 2
    X_D = sqrt(Pl*Pl + Pt*Pt / 4) / 2
    X_T = Pt / 2
    rf_r = 1.27 * X_T / r * sqrt(X_D / X_T - 0.3)
    m = sqrt(2 * h_a * cs_cp / (k_fin * t)) #cs_cp is the correction for heat/mass transfer
    
    #Using the circular fin correlation of Schmidt
    phi = (rf_r - 1) * (1 + 0.35 * log(rf_r))
    eta_f = tanh(m * r * phi) / (m * r * phi)
    
    # Fin efficiency based on analysis in 
    # "FIN EFFICIENCY CALCULATION IN ENHANCED FIN-AND-TUBE HEAT EXCHANGERS IN DRY CONDITIONS"
    # by Thomas PERROTIN, Denis CLODIC, International Congress of Refrigeration 2006
    # In the paper, there is no 0.1 in the cosine term, but if the cosine term is used without 
    # the correction, the results are garbage for wet analysis
    # Using the offset fins correlation
    phi = (rf_r - 1) * (1 + (0.3+pow(m*(rf_r*r-r)/2.5,1.5-rf_r/12.0)*(0.26*pow(rf_r,0.3)-0.3)) * log(rf_r))
    
    #finned surface efficiency
    eta_f = tanh(m * r * phi) / (m * r * phi)*cos(0.1 * m * r * phi)
    
    #overall surface efficiency
    eta_o = 1 - Af / A * (1 - eta_f)

    G_c=mdot_ha/Ac #air mass flux
    DeltaP_air=A/Atube/rho_ha*G_c**2/2.0*fa_total #airside pressure drop
    
    #write necessary values back into the given structure
    Inputs.A_a=A;
    Inputs.cp_da=cp_da
    Inputs.cp_ha=cp_ha
    if isWet==True:
        Inputs.eta_a_wet=eta_o
    else:
        Inputs.eta_a=eta_o
    Inputs.h_a=h_a
    Inputs.mdot_ha=mdot_ha
    Inputs.mdot_da=mdot_da
    Inputs.f_a=fa_total
    Inputs.dP_a=DeltaP_air
    Inputs.Re=Re_D
        

    #f_tube = 4 / pi * (0.25 + 0.118 / (St / D - 1) ^ 1.08 * Re_D ^ -0.16) * (St / D - 1)
    #f_fin = 1.455 * Re_D ^ -0.656 * (St / Sl) ^ -0.347 * (s / D) ^ -0.134 * (St / D) ^ 1.23
    #f_total = f_fin * (Af / A) + f_tube * (1 - Af / A) * (1 - t / pf)
    
def HerringboneFins(Inputs):
    #Source:
    #Empirical correlations for heat transfer and flow friction characteristics of herringbone wavy fin-and-tube heat exchangers
    #Chi-Chuan Wang, Young-Ming Hwang, Yur-Tsai Lin
    #International Journal of Refrigeration, 25, 2002, 637-680
    
    #Properties:
    p =  Inputs.Air.p #air pressure
    W=HAProps('W','T',Inputs.Air.Tdb,'P',p,'R',Inputs.Air.RH)
    #Transport properties of humid air from CoolProp
    mu_ha=HAProps('M','T',Inputs.Air.Tdb,'P',p,'W',W) 
    k_ha=HAProps('K','T',Inputs.Air.Tdb,'P',p,'W',W)*10**3
    #Evaluate the mass flow rate based on inlet conditions
    Vdot_ha =     Inputs.Air.Vdot_ha
    # To convert a parameter from per kg_{humid air} to per kg_{dry air}, divide by (1+W)
    W=HAProps('W','T',Inputs.Air.Tdb,'P',p,'R',Inputs.Air.RH)
    v_da=HAProps('V','T',Inputs.Air.Tdb,'P',p,'W',W)
    h_da=HAProps('H','T',Inputs.Air.Tdb,'P',p,'W',W)*1000
    rho_ha = 1 / v_da*(1+W) #[m^3/kg_ha]
    rho_da = 1 / v_da #[m^3/kg_da]
    mdot_ha = Vdot_ha * rho_ha #[kg_ha/s]
    mdot_da = Vdot_ha * rho_da #[kg_da/s]
    #Use a forward difference to calculate cp from cp=dh/dT
    dT=0.0001 #[K] 
    cp_da=(HAProps('H','T',Inputs.Air.Tdb+dT,'P', p, 'W',W)*1000-h_da)/dT #[J/kg_da/K]
    cp_ha=cp_da/(1+W) #[J/kg_ha/K]
    
    # Check that cs_cp is defined, if so, set it to the value passed in
    if (hasattr(Inputs,'cs_cp') and Inputs.cs_cp>0) or (hasattr(Inputs,'WetDry') and Inputs.WetDry=='Wet'):
        isWet=True
        cs_cp=Inputs.Air.cs_cp
    else:
        isWet=False
        cs_cp=1.0
    
    #Dimensions and values used for both Reynolds number ranges:
    delta_f = Inputs.Fins.t     #fin thickness(m)
    FPI = Inputs.Fins.FPI       #fins per inch
    FPM = FPI / 0.0254         #Fins per meter [1/m]
    pf = 1 / FPM                  #Fin pitch (distance between centerlines of fins)
    F_s = 1 / FPM - delta_f       #Fin spacing(m)
    
    P_t=Inputs.Tubes.Pt   #transverse tube pitch (m)
    P_L=Inputs.Tubes.Pl  #longitudinal tube pitch (m)
    
    D_o = Inputs.Tubes.OD      #Outer diameter of tube (m)
    D_c=D_o+2*delta_f      #fin collar outside diameter (m) (tube + 2*fin thickness)
    
    Ltube = Inputs.Tubes.Ltube       #length of a single tube [m]
    Nfin = Ltube * FPM #Number of fins in the tube sheet [-]
    Ntubes_bank = Inputs.Tubes.NTubes_per_bank #tubes per bank
    Height = P_t * (Ntubes_bank+1)  #Height of heat exchanger [m] # assuming that fin extends 1/2 pt above/below last tube in bundle
    A_duct = Height * Ltube  #A_duct is the face area [m^2] - equivalent to the duct cross-section #neglecting the additional height of the fins above/below the last tubes in the bundle
    Ac = A_duct - delta_f * Nfin * (Height-D_c*Ntubes_bank) - Ntubes_bank * D_c * Ltube #Minimum duct cross-sectional area that is not fin or tube(-collar) [m^2]
    u_max = mdot_ha / (rho_ha * Ac) #maximum airside velocity [m/s]
    Re_Dc=rho_ha*u_max*D_c/mu_ha #from reference #4 of Wang et all. Slightly different notation used in [4]
    
    P_d =Inputs.Fins.Pd  #wave height
    X_f =Inputs.Fins.xf   #projected fin length (m)
    tanTheta=P_d/X_f   #tangens of corrugation angle
    
    secTheta=sqrt(X_f*X_f + P_d*P_d) / X_f ###!!used wavy louvered fins definition - there seems to be a bug in the paper !!
    beta=(pi*D_c**2)/(4.0*P_t*P_L)
    oneMbeta=1.0-beta #save one operation
    D_h=2.0*F_s*oneMbeta/(oneMbeta*secTheta+2*F_s*beta/D_c)# hydraulic diameter (m)
    
    
    N=Inputs.Tubes.Nbank  #number of lomgitudunal tube rows (#Number of banks in ACHP-notation)
    
    if Re_Dc<1000.0:
        #Heat transfer
        J1=0.0045-0.491*pow(Re_Dc,-0.0316-0.0171*log(N*tanTheta))*pow(P_L/P_t,-0.109*log(N*tanTheta))*pow(D_c/D_h,0.542+0.0471*N)*pow(F_s/D_c,0.984)*pow(F_s/P_t,-0.349)
        J2=-2.72+6.84*tanTheta
        J3=2.66*tanTheta
        j=0.882*pow(Re_Dc,J1)*pow(D_c/D_h,J2)*pow(F_s/P_t,J3)*pow(F_s/D_c,-1.58)*pow(tanTheta,-0.2)
        #Friction
        F1=-0.574-0.137*pow(log(Re_Dc)-5.26,0.245)*pow(P_t/D_c,-0.765)*pow(D_c/D_h,-0.243)*pow(F_s/D_h,-0.474)*pow(tanTheta,-0.217)*pow(N,0.035)
        F2=-3.05*tanTheta
        F3=-0.192*N
        F4=-0.646*tanTheta
        f=4.37*pow(Re_Dc,F1)*pow(F_s/D_h,F2)*pow(P_L/P_t,F3)*pow(D_c/D_h,0.2054)*pow(N,F4)
    else:
        #Heat transfer
        j1=-0.0545-0.0538*tanTheta-0.302*pow(N,-0.24)*pow(F_s/P_L,-1.3)*pow(P_L/P_t,0.379)*pow(P_L/D_h,-1.35)*pow(tanTheta,-0.256)
        j2=-1.29*pow(P_L/P_t,1.77-9.43*tanTheta)*pow(D_c/D_h,0.229-1.43*tanTheta)*pow(N,-0.166-1.08*tanTheta)*pow(F_s/P_t,-0.174*log(0.5*N))
        j=0.0646*pow(Re_Dc,j1)*pow(D_c/D_h,j2)*pow(F_s/P_t,-1.03)*pow(P_L/D_c,0.432)*pow(tanTheta,-0.692)*pow(N,-0.737)
        #Friction
        f1=-0.141*pow(F_s/P_L,0.0512)*pow(tanTheta,-0.472)*pow(P_L/P_t,0.35)*pow(P_t/D_h,0.449*tanTheta)*pow(N,-0.049+0.237*tanTheta)
        f2=-0.562*pow(log(Re_Dc),-0.0923)*pow(N,0.013)
        f3=0.302*pow(Re_Dc,0.03)*pow(P_t/D_c,0.026)
        f4=-0.306+3.63*tanTheta
        f=0.228*pow(Re_Dc,f1)*pow(tanTheta,f2)*pow(F_s/P_L,f3)*pow(P_L/D_c,f4)*pow(D_c/D_h,0.383)*pow(P_L/P_t,-0.247)
    
    Pr = cp_ha * mu_ha / k_ha
    h_a = j * rho_ha * u_max * cp_ha / pow(Pr,2.0/3.0) #air side mean heat transfer coefficient using colborn j-factor
    
    #calcs needed for specific fin types
    #additional parameters needed
    k_fin =  Inputs.Fins.k_fin
    Nbank =       Inputs.Tubes.Nbank       #Number of banks
    secTheta=sqrt(X_f*X_f + P_d*P_d) / X_f  #secTheta
    #Wetted Area of a single fin [m^2]
    A_1fin = 2.0 * (Height * P_L * (Nbank+1) * secTheta  - Ntubes_bank*Nbank * pi*D_o*D_o/4) #assuming that fin extends 1/2 pt in front/after last tube in bundle
    # Total wetted area of the fins [m^2]
    Af = Nfin * A_1fin
    #Total area including tube and fins [m^2]
    A = Af + Ntubes_bank * Nbank * pi * D_o * (Ltube-Nfin*delta_f)
    
    
    r = D_o / 2
    X_D = sqrt(P_L*P_L + P_t*P_t / 4) / 2
    X_T = P_t / 2
    rf_r = 1.27 * X_T / r * sqrt(X_D / X_T - 0.3)
    m = sqrt(2 * h_a * cs_cp / (k_fin * delta_f)) #cs_cp is the correction for heat/mass transfer
    
    #Using the circular fin correlation of Schmidt
    phi = (rf_r - 1) * (1 + 0.35 * log(rf_r))
    eta_f = tanh(m * r * phi) / (m * r * phi)
    
    # Fin efficiency based on analysis in 
    # "FIN EFFICIENCY CALCULATION IN ENHANCED FIN-AND-TUBE HEAT EXCHANGERS IN DRY CONDITIONS"
    # by Thomas PERROTIN, Denis CLODIC, International Congress of Refrigeration 2006
    # In the paper, there is no 0.1 in the cosine term, but if the cosine term is used without 
    # the correction, the results are garbage for wet analysis
    # Using the offset fins correlation
    phi = (rf_r - 1) * (1 + (0.3+pow(m*(rf_r*r-r)/2.5,1.5-rf_r/12.0)*(0.26*pow(rf_r,0.3)-0.3)) * log(rf_r))
    
    #finned surface efficiency
    eta_f = tanh(m * r * phi) / (m * r * phi)*cos(0.1 * m * r * phi)
    
    #overall surface efficiency
    eta_o = 1 - Af / A * (1 - eta_f)

    G_c=mdot_ha/Ac #air mass flux
    Atube = Ntubes_bank * Nbank * pi * D_o * Ltube# Total outer area of the tubes [m^2]
    DeltaP_air=A/Atube/rho_ha*G_c**2/2.0*f #airside pressure drop
    
    #write necessary values back into the given structure
    Inputs.A_a=A;
    Inputs.cp_da=cp_da
    Inputs.cp_ha=cp_ha
    if isWet==True:
        Inputs.eta_a_wet=eta_o
    else:
        Inputs.eta_a=eta_o
    Inputs.h_a=h_a
    Inputs.mdot_ha=mdot_ha
    Inputs.mdot_da=mdot_da
    Inputs.f_a=f
    Inputs.dP_a=DeltaP_air
    Inputs.Re=Re_Dc
    
    
def PlainFins(Inputs):
    #Source:
    #Heat transfer and friction characteristics of plain fin-and-tube heat exchangers, part II: Correlation
    #Chi-Chuan Wang, Kuan-Yu Chi, Chun-Jung Chang
    
        #Properties:
    p =  Inputs.Air.p #air pressure
    W=HAProps('W','T',Inputs.Air.Tdb,'P',p,'R',Inputs.Air.RH)
    #Transport properties of humid air from CoolProp
    mu_ha=HAProps('M','T',Inputs.Air.Tdb,'P',p,'W',W) 
    k_ha=HAProps('K','T',Inputs.Air.Tdb,'P',p,'W',W)*10**3
    #Evaluate the mass flow rate based on inlet conditions
    Vdot_ha =     Inputs.Air.Vdot_ha
    # To convert a parameter from per kg_{humid air} to per kg_{dry air}, divide by (1+W)
    W=HAProps('W','T',Inputs.Air.Tdb,'P',p,'R',Inputs.Air.RH)
    v_da=HAProps('V','T',Inputs.Air.Tdb,'P',p,'W',W)
    h_da=HAProps('H','T',Inputs.Air.Tdb,'P',p,'W',W)*1000
    rho_ha = 1 / v_da*(1+W) #[m^3/kg_ha]
    rho_da = 1 / v_da #[m^3/kg_da]
    mdot_ha = Vdot_ha * rho_ha #[kg_ha/s]
    mdot_da = Vdot_ha * rho_da #[kg_da/s]
    #Use a forward difference to calculate cp from cp=dh/dT
    dT=0.0001 #[K] 
    cp_da=(HAProps('H','T',Inputs.Air.Tdb+dT,'P', p, 'W',W)*1000-h_da)/dT #[J/kg_da/K]
    cp_ha=cp_da/(1+W) #[J/kg_ha/K]
    
    # Check that cs_cp is defined, if so, set it to the value passed in
    if (hasattr(Inputs,'cs_cp') and Inputs.cs_cp>0) or (hasattr(Inputs,'WetDry') and Inputs.WetDry=='Wet'):
        isWet=True
        cs_cp=Inputs.Air.cs_cp
    else:
        isWet=False
        cs_cp=1.0
    
    #Dimensions and values used for both Reynolds number ranges:
    delta_f = Inputs.Fins.t     #fin thickness(m)
    FPI = Inputs.Fins.FPI       #fins per inch
    FPM = FPI / 0.0254         #Fins per meter [1/m]
    F_p = 1 / FPM                  #Fin pitch (distance between centerlines of fins)
    F_s = 1 / FPM - delta_f       #Fin spacing(m)
    
    P_t=Inputs.Tubes.Pt   #transverse tube pitch (m)
    P_L=Inputs.Tubes.Pl  #longitudinal tube pitch (m)
    
    D_o = Inputs.Tubes.OD      #Outer diameter of tube (m)
    D_c=D_o+2*delta_f      #fin collar outside diameter (m) (tube + 2*fin thickness)
    
    Ltube = Inputs.Tubes.Ltube       #length of a single tube [m]
    Nfin = Ltube * FPM #Number of fins in the tube sheet [-]
    Ntubes_bank = Inputs.Tubes.NTubes_per_bank #tubes per bank
    Height = P_t * (Ntubes_bank+1)  #Height of heat exchanger [m] # assuming that fin extends 1/2 pt above/below last tube in bundle
    A_duct = Height * Ltube  #A_duct is the face area [m^2] - equivalent to the duct cross-section #neglecting the additional height of the fins above/below the last tubes in the bundle
    A_c = A_duct - delta_f * Nfin * (Height-D_c*Ntubes_bank) - Ntubes_bank * D_c * Ltube #Minimum duct cross-sectional area that is not fin or tube(-collar) [m^2]
    u_max = mdot_ha / (rho_ha * A_c) #maximum airside velocity [m/s]
    Re_Dc=rho_ha*u_max*D_c/mu_ha #from reference #4 of Wang et all. Slightly different notation used in [4]
    
    N=Inputs.Tubes.Nbank  #number of lomgitudunal tube rows (#Number of banks in ACHP-notation)
    L=(N+1)*P_L #depth of heat exchanger in airflow direction, assuming fins extends 1/2 bank oer the first and last tube
    
    Nbank=N #to be able to reuse equations
    #Wetted Area of a single fin [m^2]
    A_1fin = 2.0 * (Height * P_L * (Nbank+1)   - Ntubes_bank*Nbank * pi*D_c*D_c/4) #assuming that fin extends 1/2 pt in front/after last tube in bundle
    # Total wetted area of the fins [m^2]
    Af = Nfin * A_1fin
    #Total area including tube and fins [m^2]
    A_o = Af + Ntubes_bank * Nbank * pi * D_c * (Ltube-Nfin*delta_f)
    
    D_h=4*A_c*L/A_o #Hydraulic diameter
    
    #heat transfer
    if N==1:
        P1=1.9-0.23*log(Re_Dc)
        P2=-0.236+0.126*log(Re_Dc)
        j=0.108*pow(Re_Dc,-0.29)*pow(P_t/P_L,P1)*pow(F_p/D_h,-1.084)*pow(F_p/D_h,-0.786)*pow(F_p/P_t,P2)
    if N>=2:
        P3=-0.361-0.042*N/log(Re_Dc)+0.158*log(N*(F_p/D_c)**0.41)
        P4=-1.224-0.076*pow(P_L/D_h,1.42)/log(Re_Dc)
        P5=-0.083+0.058*N/log(Re_Dc)
        P6=-5.735+1.21*log(Re_Dc/N)
        j=0.086*pow(Re_Dc,P3)*pow(N,P4)*pow(F_p/D_c,P5)*pow(F_p/D_h,P6)*pow(F_p/P_t,-0.93)
    #pressure drop
    F1=-0.764+0.739*P_t/P_L+0.177*F_p/D_c-0.00758/N
    F2=-15.689+64.021/log(Re_Dc)
    F3=1.696-15.695*log(Re_Dc)
    f=0.0267*pow(Re_Dc,F1)*pow(P_t/P_L,F2)*pow(F_p/D_c,F3)
    
    
    Pr = cp_ha * mu_ha / k_ha
    h_a = j * rho_ha * u_max * cp_ha / pow(Pr,2.0/3.0) #air side mean heat transfer coefficient using colborn j-factor
    
    #calcs needed for specific fin types
    #additional parameters needed
    k_fin =  Inputs.Fins.k_fin

    #Total area including tube and fins [m^2]
    A = A_o
    
    r = D_o / 2
    X_D = sqrt(P_L*P_L + P_t*P_t / 4) / 2
    X_T = P_t / 2
    rf_r = 1.27 * X_T / r * sqrt(X_D / X_T - 0.3)
    m = sqrt(2 * h_a * cs_cp / (k_fin * delta_f)) #cs_cp is the correction for heat/mass transfer
    
    #Using the circular fin correlation of Schmidt
    phi = (rf_r - 1) * (1 + 0.35 * log(rf_r))
    eta_f = tanh(m * r * phi) / (m * r * phi)
    
    # Fin efficiency based on analysis in 
    # "FIN EFFICIENCY CALCULATION IN ENHANCED FIN-AND-TUBE HEAT EXCHANGERS IN DRY CONDITIONS"
    # by Thomas PERROTIN, Denis CLODIC, International Congress of Refrigeration 2006
    # In the paper, there is no 0.1 in the cosine term, but if the cosine term is used without 
    # the correction, the results are garbage for wet analysis
    # Using the offset fins correlation
    phi = (rf_r - 1) * (1 + (0.3+pow(m*(rf_r*r-r)/2.5,1.5-rf_r/12.0)*(0.26*pow(rf_r,0.3)-0.3)) * log(rf_r))
    
    #finned surface efficiency
    eta_f = tanh(m * r * phi) / (m * r * phi)*cos(0.1 * m * r * phi)
    
    #overall surface efficiency
    eta_o = 1 - Af / A * (1 - eta_f)

    G_c=mdot_ha/A_c #air mass flux
    Atube = Ntubes_bank * Nbank * pi * D_o * Ltube# Total outer area of the tubes [m^2]
    DeltaP_air=A/Atube/rho_ha*G_c**2/2.0*f #airside pressure drop
    
    #write necessary values back into the given structure
    Inputs.A_a=A;
    Inputs.cp_da=cp_da
    Inputs.cp_ha=cp_ha
    if isWet==True:
        Inputs.eta_a_wet=eta_o
    else:
        Inputs.eta_a=eta_o
    Inputs.h_a=h_a
    Inputs.mdot_ha=mdot_ha
    Inputs.mdot_da=mdot_da
    Inputs.f_a=f
    Inputs.dP_a=DeltaP_air
    Inputs.Re=Re_Dc
    
    print "Warning! The PlainFins seem to be buggy, better use HerringboneFins with small waveiness"
    
    
    
if __name__=='__main__':
    
    FinsTubes=FinInputs()
    
    FinsTubes.Tubes.NTubes_per_bank=32
    FinsTubes.Tubes.Ncircuits=5
    FinsTubes.Tubes.Nbank=3
    FinsTubes.Tubes.Ltube=0.452
    FinsTubes.Tubes.OD=0.009525
    FinsTubes.Tubes.ID=0.0089154
    FinsTubes.Tubes.Pl=0.0254
    FinsTubes.Tubes.Pt=0.0219964
    
    FinsTubes.Fins.FPI=14.5
    FinsTubes.Fins.Pd=0.001
    FinsTubes.Fins.xf=0.001
    FinsTubes.Fins.t=0.00011
    FinsTubes.Fins.k_fin=237
    
    FinsTubes.Air.Vdot_ha=0.5663
    FinsTubes.Air.Tmean=299.8
    FinsTubes.Air.Tdb=299.8
    FinsTubes.Air.p=101.325
    FinsTubes.Air.RH=0.51
    FinsTubes.Air.RHmean=0.51
    FinsTubes.Air.FanPower=438
    
    FinsTubes.h_a_tuning=1.0
    
    FinsTubes.Validate()
    
    print FinsTubes  #just print our inputs
    WavyLouveredFins(FinsTubes)  #calculate
    #print "Wavy-Louvered fins:","eta_a is:"+str(FinsTubes.eta_a)+", dP_a is:"+str(FinsTubes.dP_a) #print some of the results
    #HerringboneFins(FinsTubes) 
    #print "Herringbone Fins fins:","eta_a is:"+str(FinsTubes.eta_a)+", dP_a is:"+str(FinsTubes.dP_a) #print some of the results
    #PlainFins(FinsTubes) 
    #print "Plain Fins fins:","eta_a is:"+str(FinsTubes.eta_a)+", dP_a is:"+str(FinsTubes.dP_a) #print some of the results
    #print "a graph for the fin correlations can be found here: "+r"\achp\trunk\Documentation\Web\MPLPlots"
    
    from convert_units import *
    import numpy as np
    T_in_air=C2K(np.array([21.91156557,35.00974256,34.62973333,1.887537563,11.93058636,22.03684101,34.88511447,34.21912972,46.13087575,21.98885524,34.89140381,46.04504935,21.83105245,34.93460259,46.07872677,35.00406406,35.00746031,35.05650785,12.07902318,21.96276869,34.96930278,45.99610218,45.82536445,22.01319628,34.97739709,46.03228081,22.04558986,34.98846756,46.01794557,45.82686894]))
    print T_in_air
    T_out_air=C2K(np.array([28.78985745,41.63781052,41.19573548,12.24567424,19.31507296,29.20484101,41.71311264,40.90295761,52.47790469,28.15125957,40.85095145,51.76475844,27.25191332,40.37750598,51.39640202,41.58042512,41.71669149,40.55688695,19.55247111,29.31516076,41.91559772,52.51207449,52.51724838,29.00278808,41.73992965,52.36669423,28.77633948,41.45586776,52.11011609,52.07796145]))
    DP_exp=C2K(np.array([14.12111675,23.84673016,22.65304301,-9.797957071,-5.566755153,-2.126132584,1.898849817,1.112952015,3.738750499,-1.306441764,1.744882033,3.978293506,-1.287296739,2.026808765,5.683573737,1.532792013,1.371234043,3.427513552,-9.776282454,-3.906539948,-2.388584879,-1.171195008,0.483224026,2.029724911,-3.886246512,-1.179884555,-5.122534562,-3.385596806,-1.262127168,0.084246696]))
    #inlet conditions are oulet conditions from compressor (to be consistent with missing discharge line in simulation)
    T_in_r=C2K(np.array([67.10147377,78.35730357,78.13384301,47.76928283,52.69020692,62.56512135,74.46998535,75.19187332,84.83142515,62.3836173,76.68006171,87.08338961,60.65594239,81.14687251,88.76194141,75.00402163,74.39679051,77.43017832,51.95799572,60.37359794,72.87872468,83.68284711,85.47171104,59.47620425,73.16597907,84.02673479,59.87238863,73.02381038,83.83112139,85.7876652]))
    P_cond=np.array([1513.262724,2009.839399,1994.279004,1268.777314,1246.863589,1555.81102,2037.348499,2001.989459,2552.56487,1486.460171,1968.017922,2489.210709,1435.17637,1924.820183,2451.952428,2025.5761,2036.18638,1937.269656,1299.060294,1604.281575,2097.121365,2602.785332,2588.160422,1574.733287,2079.54988,2587.621396,1557.400931,2054.880705,2567.618647,2552.901938])
    m_dot_exp=1.0/1000.0*np.array([79.60695986,80.56359025,79.67354002,89.65517973,87.30242866,86.56622185,86.10408624,83.57227382,85.2149884,73.54629467,72.04151777,71.93786696,64.34677464,62.63416426,65.15477488,82.4560853,84.8675524,64.8760222,92.72211183,92.84027134,91.48381753,90.41708854,87.22565273,87.44858877,87.66102364,87.00625237,83.24811552,83.2667997,83.29687891,80.69227919])
    RH_in_air=np.zeros(len(m_dot_exp))
    for i in range(0,len(m_dot_exp)):
        RH_in_air[i]=HAProps('R','T',float(T_in_air[i]),'P',101.325,'D',max(float(DP_exp[i]),273.5))
        if RH_in_air[i]>0.999:
            print "error in run #",i,"calculated RH",RH_in_air[i],"inputs" 'T',T_in_air[i],'P',101.325,'D',DP_exp[i]
    print "using relative humidity calculated with formulas above, since it does not work in LRCS_condenser directly"
    print RH_in_air
    
    
    