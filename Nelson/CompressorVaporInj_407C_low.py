# -*- coding: utf-8 -*-
"""
Created on Thu Feb 06 19:05:48 2014
Modified on Tue Jan 17 14:51:00 2017
 
RefProp returns [J/kg]

Pressure in kPa
Temperature in K

Solver for Semi-Emperical Model
Basline case assuming homogeneous flow through 

@author: Nelson
"""


"""
Change log..better late than never

5/7/15
-motor inefficiencies is (1-eta)*W_comp not (1-eta)*(W_comp-W_loss_friction)
-In Wbal, compressor work includes W_loss_fic and then motor efficiency applied

1/17/2017
change PropsSI to PropsSISI with RefProp units changed from [kJ/kg] to [J/kg] 

"""




"need to be locally imported....yeah I know right..."
#from CoolProp.CoolProp import PropsSI
import CoolProp as CP
import numpy as np
import math
#import DataIO
import time
import warnings

warnings.simplefilter("ignore",RuntimeWarning)
from random import randint, random


numTol = 1e-6 #numerical tolerance



#--------------------------------------------------------------------------
def Import(start,end,filename):
    "import experimental data"
    
    [data,rownum] = DataIO.ParameterImport(start,end,filename)
    
    i = 0  
    "initialize arrays"
    Tamb = float(data[i][0])
    Tsuc = float(data[i][1])
    TsucOil = float(data[i][2])
    Psuc = float(data[i][3])
    Tex = float(data[i][4])
    Pex = float(data[i][5])
    mgas = float(data[i][6])
    mliq = float(data[i][7])
    Nexp = float(data[i][8])
    tau = float(data[i][9])
    i=i+1
    
    while i < (end - start+1):
        Tamb = np.append(Tamb,float(data[i][0]))
        Tsuc = np.append(Tsuc,float(data[i][1]))
        TsucOil = np.append(Tsuc,float(data[i][2]))
        Psuc = np.append(Psuc,float(data[i][3]))
        Tex = np.append(Tex,float(data[i][4]))
        Pex = np.append(Pex,float(data[i][5]))
        mgas = np.append(mgas,float(data[i][6]))
        mliq = np.append(mliq,float(data[i][7]))
        Nexp = np.append(Nexp,float(data[i][8]))
        tau = np.append(tau,float(data[i][9]))
#        print "i: ",i
        i=i+1
        Data = [Tamb,Tsuc,TsucOil,Psuc,Tex,Pex,mgas,mliq,Nexp,tau]
    
    return Data
    
    

#=============================================================================
"The Fluid Properties"

"The Fluid Properties"

def c_l(T):
    "specific heat [J/kg-K] of Duratherm LT given T in K"
#    c = (2300)/1000 #Copeland 32-3MAF POE Oil Ian Thesis
    
    #correlations from Sugi Testing?
    c0= 1.93709
    c1= - 0.00214
    c2 = 6.58646e-6
    
    c = c0 + c1*(T) + c2*(T)**2
    return c*1000 #[J/kg-K]
    
def u_l(T):
    "internal energy [J/kg] of Duratherm LT given T in K"
#    u = (2300*T)/1000  #Copeland 32-3MAF POE Oil Ian Thesis

    #correlations from Sugi Testing?
    c0= 1.93709
    c1= - 0.00214
    c2 = 6.58646e-6
    u = c0*(T) + c1/2 *(T**2 ) + c2/3 *(T**3)

    return u*1000 #[J/kg]

def rho_l(T):
    "density [kg/m^3] of Duratherm LT given T in K"
    rho = (-0.00074351165*(T-273.15) + 0.9924395)*1000  #Copeland 32-3MAF POE Oil Ian Thesis
    return rho

def s_l(T):
    "specific entropy [J/kg-K] of Duratherm LT given T in K"  
#    s = 2300*np.log(T/298.0)/1000 #Copeland 32-3MAF POE Oil Ian Thesis

    #correlations from Sugi Testing?
    c0= 1.93709
    c1= - 0.00214
    c2 = 6.58646e-6
    
    s = c0*np.log((T)/298) + c1*((T)-298) + c2/2*((T)**2 - 298**2)
    
    return s*1000 #[J/kg-K]

def h_l(T,P):
    "the specific enthalpy of Duratherm LT [J/kg-k]"
    h = u_l(T) + P/rho_l(T)
    return h



def T_l(h,P):
    "find the liquid temperature [K] given h and P"
    #initial guesses, functions fairly linear so guesses not too important
    T = [300, 400]
    h_check = h_l(T[0],P)
    f = [h_check - h] #function to converge
    
    i=0 #array index
    while abs(f[i])> numTol:
        i=i+1 #update index
        h_check = h_l(T[i],P)
        f = np.append(f,h_check - h)
        T = np.append(T , T[i]-(f[i]*(T[i]-T[i-1]))/(f[i]-f[i-1])) #secant method
        
        if i>100:
            raise Exception("T_l not converging after 100x")
        
    return T[i]
    

def P_hv(AS,h,v):
    P = [1000000,2000000]
    AS.update(CP.HmassP_INPUTS, h, P[0])
    v_check = 1.0/AS.rhomass() #[m^3/kg]
    
    f = [v_check - v] #function to converge
    
    i=0 #array index
    while abs(f[i])> numTol:
        i=i+1 #update index
        
        "ride herd on Pressure"
        if P[i]>10000000:
            P[i] = 100000+ 500*random()
        if P[i] < 100000:
            P[i] = 100000+ 500*random()
        
        AS.update(CP.HmassP_INPUTS, h, P[i])
        v_check = 1.0/AS.rhomass() #[m^3/kg]
        f = np.append(f,v_check - v)
        P = np.append(P , P[i]-(f[i]*(P[i]-P[i-1]))/(f[i]-f[i-1])) #secant method
    
        if i>100:
#            print "v_check - v: ",v_check - v 
#            print "P: ",P[i]
            raise Exception("P_hv not converging after 100x")
        
    return P[i]
    


def vol_l(T):
    "return specific volume of liq"
    v = 1.0/rho_l(T)
    return v

def vol_g(AS,T,P):
    "returns the specific volume so inverse of density doesn't need to be manually entered"
    AS.update(CP.PT_INPUTS, P, T)
    v = 1.0/AS.rhomass() #[m^3/kg]
    return v
        
#==========================================================================    
    
    
"Mixture Function Calls"




def T_mix_h(AS,h,P,y,Tguess):
    "find the mixture temperature given its specific enthalpy"
    T=[Tguess+5, Tguess-5]
    AS.update(CP.PT_INPUTS, P, T[0])
    h0 = AS.hmass() #[J/kg]
    AS.update(CP.PT_INPUTS, P, T[1])
    h1 = AS.hmass() #[J/kg]
    hgas = [h0,h1]
    
    AS.update(CP.HmassP_INPUTS, hgas[0], P)
    Tgas = AS.T() #[K]
    
    h_check = (hgas[0]+ y*h_l(Tgas,P))/(1+y)
    f =[h_check - h]
    
    i=0 #array index
    while abs(f[i])> 1e-5: #numTol:
        i=i+1 #update index
#        if T[i]< Tguess/2: #ride herd
#            T[i] = Tguess/2 + random()*(Tguess - Tguess/2)
        
        AS.update(CP.HmassP_INPUTS, hgas[i], P)
        Tgas = AS.T() #[K]
        h_check = (hgas[i]+ y*h_l(Tgas,P))/(1+y)
        
        f = np.append(f,h_check - h)
        hgas = np.append(hgas , hgas[i]-(f[i]*(hgas[i]-hgas[i-1]))/(f[i]-f[i-1])) #secant method
        
        if i>150:
            Tgas = T_mix_h_bisect(gas,h,P,y,Tguess)
            break
    
    return Tgas



  
    

def T_mix_h_bisect(AS,h,P,y,Tguess):
    "find the mixture temperature given its specific enthalpy"
       
    THigh = 450
    TLow = 100       
    
    AS.update(CP.PT_INPUTS, P, THigh)
    h_h = AS.hmass() #[J/kg]
    AS.update(CP.PT_INPUTS, P, TLow)
    h_l = AS.hmass() #[J/kg]
    
    h_high = (h_h + y*h_l(THigh,P))/(1+y)
    h_low = (h_l + y*h_l(TLow,P))/(1+y) 
    
    FHigh = h_high - h
    FLow = h_low - h
    
    if FLow>FHigh:
        signChange = -1.0
    else:
        signChange=1.0
    
    FHigh = FHigh*signChange
    FLow = FLow*signChange
    
#    print "Fhigh: ",FHigh
#    print "FLow: ",FLow
    
    NumSections = 0
    FMid = 10
    
    while abs(FMid) > 1e-5:
        
        NumSections = NumSections+1
        TMid = (THigh+TLow)/2.0
        
        AS.update(CP.PT_INPUTS, P, TMid)
        h_m = AS.hmass() #[J/kg]
        FMid = (h_m + y*h_l(TMid,P))/(1+y) - h
        
        if FMid>0:
            THigh = TMid
        else:
            TLow = TMid
        
#        print "Tmid: ",TMid
        if NumSections>1000:
#            print "Phigh: ",PHigh
#            print "Plow: ",PLow
#            print "PMid: ",PMid
#            print "Fmid: ",FMid
            raise Exception("T_mix_h converging after 100x")
    
    T2 = TMid    


    return T2



def T_mix_s(AS,s,P,y,Tguess):
    "find the mixture temperature given its specific enthalpy"
    T=[Tguess+5, Tguess-5]
    
    AS.update(CP.PT_INPUTS, P, T[0])
    s0 = AS.smass() #[J/kg-K]
    AS.update(CP.PT_INPUTS, P, T[1])
    s1 = AS.smass() #[J/kg-K]
    sgas = [s0,s1]
    
    AS.update(CP.PSmass_INPUTS, P, sgas[0])
    Tgas = AS.T() #[K]
    
    s_check = (sgas[0]+ y*s_l(Tgas))/(1+y)
    f =[s_check - s]
    
    i=0 #array index
    while abs(f[i])> 1e-5: #numTol:
        i=i+1 #update index
#        if T[i]< Tguess/2: #ride herd
#            T[i] = Tguess/2 + random()*(Tguess - Tguess/2)
        
        AS.update(CP.PSmass_INPUTS, P, sgas[i])
        Tgas = AS.T() #[K]
        s_check = (sgas[i]+ y*s_l(Tgas))/(1+y)
        
        f = np.append(f,s_check - s)
        sgas = np.append(sgas , sgas[i]-(f[i]*(sgas[i]-sgas[i-1]))/(f[i]-f[i-1])) #secant method
        
        if i>150:
            raise Exception("T_mix_s not converging after 100x")
#            Tgas = T_mix_h_bisect(gas,h,P,y,Tguess)
#            break
    
    return Tgas


    
def gamma_mix(AS,T,P,y):
    AS.update(CP.PT_INPUTS, P, T)
    Cp0 = AS.cpmass() #[J/kg-K]
    Cv0 = AS.cvmass() #[J/kg-K]
    Cp = (Cp0 + y*c_l(T))/(1+y)
    Cv = (Cv0 + y*c_l(T))/(1+y)
    gamma = Cp/Cv
    return gamma



"Ian Helper Functions"
def cK_e(vl,vg,x,psi):
    
    K_e = psi+(1.0-psi)*math.sqrt(( vg / vl + psi*(1.0 -x)/x) /(1.0+ psi*(1.0 -x)/x));
    
    return K_e
    
    
def cV_e(vl,vg,Ke,x,psi):
    
    Kc =1.0/( psi +((1.0 - psi) *(1.0 - psi))/( Ke -psi));
    ve =(x* vg + Ke *(1.0 - x)* vl )*(x +(1.0 - x)/Kc);
    
    return ve
    
def dvdP_m(AS,T,P,xL):
    
    delta =0.001;
    
    return (1.0/ rho_m(AS,T,P+delta ,xL) - 1.0/ rho_m(AS,T,P,xL))/ delta

    
def rho_m(AS,T,P,xL):
    if (xL ==0):
        return 1.0/vol_g(AS,T,P)
    if (xL ==1):        
        return rho_l(T);
    vG =vol_g(AS,T,P);
    vL =1.0/rho_l(T);
    x=1- xL;
    rhom =1.0/(vG*x+vL*(1 - x))
    
    return rhom

#======================================================================   

"Selected Modelling steps in functions for overall clarity"




def Chisholm_leak(AS,Area,P1,P2,T1,y):
    psi = 0.4 #recommended by chisholm?
    x = 1.0/(1.0+y)
    N = 20 #Ian used 20 divisions
    sigma = 0 #nozzle area ratio, taken as zero in Ian's thesis
    Cd = 0.77
    
    I=0
    gamma = gamma_mix(AS,T1,P1,0) #gas only in Ian's
    T=T1
    dP = (P1 - P2)/N
    P = P1
    vl = vol_l(T)
    vg = vol_g(AS,T,P)
    vg0 = vg
    xg = x
    Ke = cK_e(vl,vg,xg,psi)
    ve1 = cV_e(vl,vg,Ke,xg,psi)
    ve_high = ve1
    
    gamma_mixture = gamma_mix(AS,T1,P1,y)
    P_Crit = P1*pow((2/(gamma_mixture+1)),(gamma_mixture/(gamma_mixture-1)))
    
    for kk in xrange(1,N):
        P = P - dP  #kPa
        vg = vol_g(AS,T,P)
        Ke = cK_e(vl,vg,xg,psi)
        ve2 = cV_e(vl,vg,Ke,xg,psi)
        dI = dP/2.0 * (ve1 + ve2)
        I = I + dI
        ve1 = ve2
        
        if P < P_Crit:
            break

    "Two phase discharge coefficient from Morris"        
    beta = math.sqrt(sigma)
    ve_thr = ve2
    G_thr = math.sqrt (2.0* I/( pow ( ve_thr ,2.0) -pow (beta ,4.0) * pow (ve1 ,2.0) ))
    
#    "two phase choking"  
#    G_max = math.sqrt ( -1000.0/( xg* dvdP_m(gas,T,P2 ,0) +(1 - xg)* dvdP_m (gas,T,P2,1) ))
#    M= G_thr*math.sqrt((-xg* dvdP_m(gas,T,P2,0) -(1-xg)*dvdP_m (gas,T,P2 ,1)) /1000.0)
#    
#    if (M >1):
#        G_thr = G_max 
    
    T2 = T1
    mdot = Cd*G_thr *Area
#    Ma=M
    
    return mdot



def leakage(AS,T_preleak,P_preleak,P_postleak,A_leak):
    
    y_star = 0
    gamma_leak = gamma_mix(AS,T_preleak,P_preleak,y_star)
    P_leak_crit = P_preleak*pow((2.0/(gamma_leak+1)),gamma_leak/(gamma_leak-1))
    P_leak_thr = max(P_postleak,P_leak_crit) #choke flow in leakage
    
    AS.update(CP.PT_INPUTS, P_preleak, T_preleak)
    s_p = AS.smass() #[J/kg-K]
    h_p = AS.hmass() #[J/kg]
    s_preleak = (s_p + y_star*s_l(T_preleak))/(1+y_star)
    h_preleak = (h_p + y_star*h_l(T_preleak,P_preleak))/(1+y_star)
    
    AS.update(CP.PSmass_INPUTS, P_leak_thr, s_preleak)
    v_leak_thr = 1.0/AS.rhomass() #[m^3/kg]
    h_leak_thr = AS.hmass() #[J/kg]
    vel_leak_thr = math.sqrt(2.0*(h_preleak - h_leak_thr)) #J/kg
    m_leak = 1.0/v_leak_thr*vel_leak_thr*A_leak
    
    return m_leak
    
    

def Chisholm_suc(AS,Area,mg,ml,P1,T1):
    
    psi = 0.4 #recommended by chisholm?
    x = mg/(ml+mg)
    y = ml/mg
    mDot = mg+ml
    Cd = 0.77
    sigma=0
    
    gamma = gamma_mix(AS,T1,P1,y)
    P_Crit = P1*pow((2/(gamma+1)),(gamma/(gamma-1)))  #serve as bound on itntegration
    dP = 2 #pressure step [kPa]
    
    "inlet conditions"
    AS.update(CP.PT_INPUTS, P1, T1)
    s0 = AS.smass() #[J/kg-K]
    h0 = AS.hmass() #[J/kg]
    s1 = (s0 + y*s_l(T1))/(1+y)
    h1 = (h0 + y*h_l(T1,P1))/(1+y)
    
    I=0
    mChis = 0
    
    gamma = gamma_mix(AS,T1,P1,0) #gas only in Ian's
    T=T1
    P = P1
    vl = vol_l(T)
    vg = vol_g(AS,T,P)
    vg0 = vg
    xg = x
    Ke = cK_e(vl,vg,xg,psi)
    ve1 = cV_e(vl,vg,Ke,xg,psi)
    ve_high = ve1

    P2 = P1

    
    while mChis < mDot and P2 > P_Crit:
        P2 = P2-dP
        
        
        T2 = T1#T_mix_s(gas,s1,P2,y,T1) #isentropic.
        
        vg = vol_g(AS,T2,P2)
        Ke = cK_e(vl,vg,xg,psi)
        ve2 = cV_e(vl,vg,Ke,xg,psi)
        dI = dP/2.0 * (ve1 + ve2)
        I = I + dI       
        ve1 = ve2
        
        beta = math.sqrt(sigma)
        ve_thr = ve2
        G_thr = math.sqrt (2.0* I/( pow ( ve_thr ,2.0) -pow (beta ,4.0) * pow (ve1 ,2.0) ))
        mChis = Cd*G_thr *Area
        
        
    if P2<P_Crit:
        raise Exception("Chisholm choke")
        
    T2 = T_mix_h(AS,h1,P2,y,T1) #isobaric diffuser
        
    
    return [P2,T2]  
    
    
def Chisholm_ex(AS,Area,mg,ml,P2,T2):
    
    psi = 0.4 #recommended by chisholm?
    x = mg/(ml+mg)
    y = ml/mg
    mDot = mg+ml
    Cd = 0.77
    sigma=0
    
    gamma = gamma_mix(AS,T2,P2,y)

    dP = 2 #pressure step [kPa]
    
#    "inlet conditions"
#    s2 = (PropsSI('S','T',T2,'P',P2,gas)+y*s_l(T2))/(1+y)
#    h2 = (PropsSI('H','T',T2,'P',P2,gas)+y*h_l(T2,P2))/(1+y)
   
    
    I=0
    mChis = 0
    
    gamma = gamma_mix(AS,T2,P2,0) #gas only in Ian's
    
    T=T2
    P = P2
    vl = vol_l(T)
    vg = vol_g(AS,T,P)
    vg0 = vg
    xg = x
    Ke = cK_e(vl,vg,xg,psi)
    ve2 = cV_e(vl,vg,Ke,xg,psi)
#    ve_high = ve1

    P1 = P2
    

    while mChis < mDot:
        P1 = P1 + dP
        
        T1 = T2
        
        vg = vol_g(AS,T1,P1)
        Ke = cK_e(vl,vg,xg,psi)
        ve1 = cV_e(vl,vg,Ke,xg,psi)
        dI = dP/2.0 * (ve1 + ve2)
        I = I + dI       
        ve2 = ve1
        
        beta = math.sqrt(sigma)
        ve_thr = ve2
        G_thr = math.sqrt (2.0* I/( pow ( ve_thr ,2.0) -pow (beta ,4.0) * pow (ve1 ,2.0) ))
        mChis = Cd*G_thr *Area
        
        
    if P1*pow((2/(gamma+1)),(gamma/(gamma-1))) > P2:
        raise Exception("Chisholm choke")        
        

    
    return [P1,T1]    







def SuctionNozzle(AS,Area,mg,P1,T1,h1,s1):
    
    
    y = 0

    
    gamma = gamma_mix(AS,T1,P1,y)
    P_Crit = P1*pow((2/(gamma+1)),(gamma/(gamma-1)))  #serve as bound on iterations
    
    
    
    PHigh = P1-1
    PLow = P_Crit
    
    FHigh = Suction_helper(AS,h1,s1,T1,PHigh,Area,mg)
    FLow = Suction_helper(AS,h1,s1,T1,PLow,Area,mg)  
   
    if FLow>FHigh:
        signChange = -1.0
    else:
        signChange=1.0
    
    FHigh = FHigh*signChange
    FLow = FLow*signChange
    
#    print "Fhigh: ",FHigh
#    print "FLow: ",FLow
    
    NumSections = 0
    FMid = 10
    
    while abs(FMid) > numTol:
       
        NumSections = NumSections+1
        PMid = (PHigh+PLow)/2.0
        FMid = signChange*Suction_helper(AS,h1,s1,T1,PMid,Area,mg) 
        
        if FMid>0:
            PHigh = PMid
        else:
            PLow = PMid
        
        
        if NumSections>100:
#            print "Phigh: ",PHigh
#            print "Plow: ",PLow
#            print "PMid: ",PMid
#            print "Fmid: ",FMid
            raise Exception("Nozzle not converging after 100x")
    
    P2 = PMid    
    
    "recover static enthalpy in isobaric diffuser"
    AS.update(CP.HmassP_INPUTS, h1, P2)
    T2 = AS.T() #[K] #T_mix_h(gas,h1,P2,y,T1)
#    print "Pnoz numSecs: ",NumSections
    
    return [T2,P2]
    
    
def Suction_helper(gas,h1,s1,T1,P2,Area,mg):
    AS.update(CP.PSmass_INPUTS, P2, s1)
    T2 = AS.T() #[K] #T_mix_s(gas,s1,P2,y,T1) #isentropic nozzle
    h2 = AS.hmass() #[J/kg]
    AS.update(CP.HmassP_INPUTS, h2, P2)
    v2 = 1.0/AS.rhomass() #[m^3/kg] #(vol_g(gas,T2,P2) +y*vol_l(T2))/(1+y)
    vel2 = (mg)/(Area/v2)
    KE2 = 0.5*vel2**2.0
    Ebal = h1 - (h2 + KE2)        
    
    return Ebal  




def IsentropicCompression(AS,s1,T1,P1,v2,v_ratio,P_ex):
    "Perform isentropic compression along the volume ratio of the machine"
    "Assume oil and gas not perfectly mixed immediately after injection but mixed after compression"
    
    
    y=0
#    s1 = (PropsSI('S','T',T1,'P',P1,gas)+y*s_l(T1_oil))/(1+y)
    
    "guess outlet pressure"
    P= [P1*v_ratio,0.95*P1*v_ratio]
    
    i=0 #secant array index for pressure
    f = [10.0000] #initialize convergence function (unlikely to be a naturally occuring soln)
            
    while abs(f[i]) > numTol:        
        if f[i] != 10.0000: #only increase index after 1st iteration
            i=i+1
        
        "ride herd on Pressure"
        if P[i] < P1:
            P[i] = P_ex - random()*(P1-P_ex)
#        if P[i] < P1/(v_ratio*2.0):
#            P[i] = P_ex + random()*(P1-P_ex)
        
        "guess outlet temperature"
        if y > 1:
            T = [T1+5,T1+15]
        else:
            T= [T1+20,T1+40]
        
        j = 0 #secant array index for temperature
        g = [10.0000] #initialize convergence function (unlikely to be a naturally occuring soln)
        
        
        AS.update(CP.PSmass_INPUTS, P[i], s1)
        T_s = AS.T() #[K]    
        
        "Mass (volume) Balance"
        v_out = 1.0/AS.rhomass() #[m^3/kg]
        Mbal = v2 - v_out
        
        if i > 0:
            f = np.append(f, Mbal)              
            P = np.append(P, P[i] - (f[i]*(P[i]-P[i-1]))/(f[i]-f[i-1]))            
        else:                    
            f = [Mbal]         
        if i>50:  
#            print "Mbal: isenExp: ",Mbal                                 
#            raise Exception("IsenExp-P not converging after 50x")
            break
            
    P = P[i]
    T = T_s
#    print "P: ",P
#    print "T: ",T

    return [T,P]
    



def ExhaustNozzle(AS,Area,mg,P2,T2,P_suc):

    
    "Calculate Nozzle inlet pressure"
    AS.update(CP.PT_INPUTS, P2, T2)
    h2 = AS.hmass() #[J/kg]
    s2 = AS.smass() #[J/kg-K]
    v2 = vol_g(AS,T2,P2)
    vel2 = (mg)/(Area/v2)
    KE = 0.5*vel2**2
    h1 = h2 + KE
    s1=s2
    
    P1 = [P2+10000,P2+50000]  #guess nozzle inlet pressure"
    
    j=0
    g=[10.0000]
    
    while abs(g[j])> numTol:
        if g[j] != 10.0000: #only increase index after 1st iteration
            j=j+1   
        
        "ride herd on Pressure"
        if P1[j]>10000000:
            P1[j] = P_suc + 100000*random()
        if P1[j] < 100:
            P1[j] = P_suc + 100000*random()
        
        AS.update(CP.PSmass_INPUTS, P1[j], s1)    
        h1_check = AS.hmass() #[J/kg]
        Ebal = h1 - h1_check
        
        if j > 0:
            g = np.append(g, Ebal)              
            P1 = np.append(P1, P1[j] - (g[j]*(P1[j]-P1[j-1]))/(g[j]-g[j-1]))            
        else:                    
            g = [Ebal]         
        if j>30:    
#            print "Ebal: ", Ebal
#            print "P: ",P1
            raise Exception("Exhaust Nozzle P not converging after 30x") 
    
    AS.update(CP.PSmass_INPUTS, P1[j], s1)
    T1 = AS.T() #[K] #isentropic nozzle
        
        
    return [T1,P1[j]]



    
    
    


# Main Solver ==============================================================


def Compressor(AS,T_suc,T_inj,P_suc,P_inj,P_dis,N_comp,VR1,VR2,A_suc,A_inj,A_leak,A_dis,V_suc_comp,T_loss,eta_m):
    "Solution method for semi-emperical scroll expander with oil Injection"
    "suction port dry refrigerant until oil mixed"
    
    
    
    N_comp = N_comp*1.0 #convert int to double

    "Friction Losses"
    W_loss_fric =  2.0*math.pi*(N_comp/60)*T_loss #W 
    
    
    "generate a starting guess for massflow iterations"
    mguess = V_suc_comp*(N_comp/60)/(vol_g(AS,T_suc,P_suc))
    mLeak_guess = leakage(AS,T_suc+100,P_dis,P_suc,A_leak)

    mguess = mguess - mLeak_guess
    
    
    
    "Guess massflow rate depening on rotational speed"
    if N_comp > 1000:
        M_dot = [mguess,0.9*mguess]
    if N_comp < 1000:
        M_dot = [mguess,0.6*mguess] #at slower speeds more leakage relatively so assume greater than expected flow
    
    i=0 #secant array index for massflow rate
    f = [10.0000] #initialize convergence function (unlikely to be a naturally occuring soln)
    

    while abs(f[i]) > numTol:        

        if f[i] != 10.0000: #only increase index after 1st iteration
            i=i+1
        
        "ride herd on massflow"
        if M_dot[i] < 0:
            M_dot[i] = mguess - random()*(mguess/4)

   
        
        "(1) Suction Inlet Conditions"
        AS.update(CP.PT_INPUTS, P_suc, T_suc)
        h_suc = AS.hmass() #[J/kg]
        s_suc = AS.smass() #[J/kg-K]

        "(1) Injection Inlet Conditions"
        AS.update(CP.PT_INPUTS, P_inj, T_inj)
        h_inj = AS.hmass() #[J/kg]
        s_inj = AS.smass() #[J/kg-K]
        
        
        
        
        "(2) suction Nozzle pressure drop"        
        nozzle_start_time = time.time()
        nozzlePass=False
        nozReduce=-1
        while nozzlePass == False:   
            nozReduce = nozReduce+1
            
            "gas and liquid massflow rates"
            m_ref = M_dot[i]
            
      
            if time.time() - nozzle_start_time > 5.0: #kill runs taking too long (s)"
                raise Exception("Time's up")            
#            [P_suc1,T_suc1] = Chisholm_suc(gas,A_suc,m_g,m_l,P_suc,T_suc)
            try:  
                "Using Isentropic nozzle since no liquid at suction" 
                [T_suc1,P_suc1] = SuctionNozzle(AS,A_suc,m_ref,P_suc,T_suc,h_suc,s_suc)
                AS.update(CP.PT_INPUTS, P_suc1, T_suc1)
                h_suc1 = AS.hmass() #[J/kg]
                s_suc1 = AS.smass() #[J/kg-K]
                nozzlePass = True
                
            except:
                M_dot[i] = M_dot[i]*(0.95-nozReduce*0.02) #gradually decrease till a sufficient mass flow range is found
        
        
        
#        "Actual UA Values"
#        UA_amb = UA_amb_n/1000 #W --> kW

        
        "Generate starting guesses for Work"
        
        "guess isentropic step"
        V_dot_suc_comp = V_suc_comp*(N_comp/60.0)
        v_dis3 = (V_dot_suc_comp/(VR1*VR2))/(M_dot[i]+mLeak_guess)
        pseudo_s1 = s_suc1
        [T_dis3,P_dis3] = IsentropicCompression(AS,pseudo_s1,T_suc1,P_suc1,v_dis3,(VR1*VR2),P_dis)
        AS.update(CP.PT_INPUTS, P_dis3, T_dis3)
        h_dis3 = AS.hmass() #[J/kg]
        pseudo_hsuc1 = h_suc1
        w_comp_int_s = h_dis3 - pseudo_hsuc1        
        
        "guess const volume step"
        P_dis2 = P_dis 
        w_comp_v = v_dis3*(P_dis2 - P_dis3)
        
        Wcomp_guess = (M_dot[i]+mLeak_guess)*(w_comp_int_s+w_comp_v)/eta_m + W_loss_fric
#        print "WcompGuess: ", Wcomp_guess
        W_comp = [Wcomp_guess,0.9*Wcomp_guess]
        
        k = 0 #index for work guesses
        L = [10.0000]

        while abs(L[k]) > 1e-4:
        
            if L[k] != 10.0000:
                k=k+1
            
            "ride herd on work"
            if W_comp[k] < 0:
                W_comp[k] = 0.75*Wcomp_guess + random()*(0.5*Wcomp_guess)

           
            
            
            "Guess injection flow rate"
            m_inj_guess = [0.1*m_ref, 0.3*m_ref] #assumes Q_suc ~ Q_ex

    
            j=0 #secant array index for massflow rate
            g = [10.0000] #initialize convergence function (unlikely to be a naturally occuring soln)
                
            while abs(g[j]) > numTol:        
    #            print "g[j]: ",g[j]
                if g[j] != 10.0000: #only increase index after 1st iteration
                    j=j+1       
                    
                
                "ride herd on m_inj1"
                if m_inj_guess[j] > 1.5*m_ref:
                    m_inj_guess[j] = random()*m_ref
                if m_inj_guess[j] < -1.5*m_ref:
                    m_inj_guess[j] = random()*m_ref
               
                
                "Shell Heat Loss - Energy Balance"
                Q_dot_amb = W_loss_fric + W_comp[k]*(1 - eta_m)
                
                "Envelope Energy Balance"
                h_dis = (h_suc*m_ref + h_inj*m_inj_guess[j] + W_comp[k] - Q_dot_amb  ) / (m_ref + m_inj_guess[j])
#                print  W_comp[k]/Wcomp_guess
                
                AS.update(CP.HmassP_INPUTS, h_dis, P_dis)
                T_dis = AS.T() #[K]
                
                
#                print "inj ratio: ", m_inj_guess[j]/m_ref
                [T_dis1,P_dis1] = ExhaustNozzle(AS,A_dis,(m_ref + m_inj_guess[j]),P_dis,T_dis,P_suc)
                AS.update(CP.PT_INPUTS, P_dis1, T_dis1)
                h_dis1 = AS.hmass() #[J/kg]


                
                
                
                "Leakage mass flow"
                m_Leak = leakage(AS,T_dis1,P_dis1,P_suc1,A_leak)
                
                
                
                "leakage mixing at inlet"
                h_leak = h_dis1
                m_ref_postLeak = m_ref + m_Leak
                h_suc2= (m_ref*h_suc1 + m_Leak*h_leak) / m_ref_postLeak 
                P_suc2 = P_suc1
                
                
                "1st Isentropic Compression Stage"
                v_int1 = (V_dot_suc_comp/VR1)/(m_ref_postLeak)
                AS.update(CP.HmassP_INPUTS, h_suc2, P_suc2)
                s_suc2 = AS.smass() #[J/kg-K]
                T_suc2 = AS.T() #[K]
                v_suc2 = 1.0/AS.rhomass() #[m^3/kg]
                
                [T_int1,P_int1] = IsentropicCompression(AS,s_suc2,T_suc2,P_suc2,v_int1,VR1,P_suc2*VR1)
                AS.update(CP.PT_INPUTS, P_int1, T_int1)
                h_int1 = As.hmass() #[J/kg]
                
                
                
                
                "1st Injection Mass Flow"
                "Determine the average pressure between the inital pressure before"
                "the ports open and the final pressure after the injection mass enters the "
                "finite volume"
                
                "guess average pressure for injection"
                P_inj1_guess = [P_int1 + (P_inj-P_int1)*0.25, P_int1 + (P_inj-P_int1)*0.75]
                
                s=0 #secant array index for massflow rate
                E = [10.0000] #initialize convergence function (unlikely to be a naturally occuring soln)
                    
                while abs(E[s]) > 0.01:        

                    if E[s] != 10.0000: #only increase index after 1st iteration
                        s=s+1
                    
                    "Inflow scenario"
                    if P_inj > P_int1:
                        
                        s_inj1 = s_inj
                        
                        gamma = gamma_mix(AS,T_inj,P_inj,0)
                        P_Crit_inj = P_inj*pow((2/(gamma+1)),(gamma/(gamma-1)))
                        P_thr = max(P_inj1_guess[s],P_Crit_inj)
                        
                        AS.update(CP.PSmass_INPUTS, P_thr, s_inj1)
                        rho_inj1 = AS.rhomass() #[kg/m^3]
                        h_inj1 = AS.hmass() #[J/kg]
                        
                        KE = (h_inj - h_inj1)                
                        vel_inj1 = np.sqrt(KE/(0.5))
                        
                        m_inj = rho_inj1*vel_inj1*A_inj#*(1.0/(N_comp/3600.0))
                        
                        "Injection Mixing"    
                        h_int2 = (m_ref_postLeak*h_int1 + m_inj*h_inj)/(m_ref_postLeak + m_inj)
                        v_int2 = (V_dot_suc_comp/VR1) / (m_inj + m_ref_postLeak)
                        P_int2 = P_hv(AS,h_int2,v_int2)  #iteratively solves since refprop crashes otherwise
                        AS.update(CP.HmassP_INPUTS, h_int2, P_int2)
                        s_int2 = AS.smass() #[J/kg-K]
                        
                        
                        "Average Inj Pressure"
                        P_avg = P_int1 + (P_int2 - P_int1)*1.0
                        Pinj1_bal = P_avg - P_inj1_guess[s]
                        
                        if s > 0:
                            E = np.append(E, Pinj1_bal)              
                            P_inj1_guess = np.append(P_inj1_guess, P_inj1_guess[s] - (E[s]*(P_inj1_guess[s]-P_inj1_guess[s-1]))/(E[s]-E[s-1]))            
                        else:                    
                            E = [Pinj1_bal]
                        
                        if s>50:
                            raise Exception("Injection1 avg Pressure not converging afer 50x")     

                       
                    
                    "Outflow scenario"
                    if P_inj < P_int1:
                        raise Exception("Injection Pressure too Low")
          


                "2nd Isentropic Compression Stage"
                v_int3 = (V_dot_suc_comp/(VR1*VR2))/(m_ref_postLeak + m_inj)
                AS.update(CP.HmassP_INPUTS, h_int2, P_int2)
                T_int2 = AS.T() #[K]
                
                
                [T_int3,P_int3] = IsentropicCompression(AS,s_int2,T_int2,P_int2,v_int3,VR2,P_int2*VR2)
                AS.update(CP.PT_INPUTS, P_int3, T_int3)
                h_int3 = AS.hmass() #[J/kg]    

                
                
                "Injection MassFlow Balance"
                Inj_bal = m_inj - m_inj_guess[j]
                if j > 0:
                    g = np.append(g, Inj_bal)              
                    m_inj_guess = np.append(m_inj_guess, m_inj_guess[j] - (g[j]*(m_inj_guess[j]-m_inj_guess[j-1]))/(g[j]-g[j-1]))            
                else:                    
                    g = [Inj_bal]
                
                if j>50:
#                    print "Inj_bal: ",Inj_bal
                    raise Exception("Injection Flow not converging after 50x")
                    break



            "const volume step"
            w_comp_v = v_int3*(P_dis1 - P_int3)

            
            
            "Work guess Balance"
            Wbal = (m_ref_postLeak*(h_int1 - h_suc2) + (m_ref_postLeak + m_inj)*(h_int3 - h_int2) + (m_ref_postLeak + m_inj)*w_comp_v + W_loss_fric)/eta_m  - W_comp[k]
          
            
#            print "Wbal: ",Wbal
#            print "Wcomp[k]: ",W_comp[k]
            
            if k > 0:
                L = np.append(L, Wbal)              
                W_comp = np.append(W_comp, W_comp[k] - (L[k]*(W_comp[k]-W_comp[k-1]))/(L[k]-L[k-1]))            
            else:                    
                L = [Wbal]
            
            if k>30:
#                print "Wbal: ",Wbal
#                print "W_comp: ",W_comp
#                print "L: ",L
                raise Exception("Wbal not converging after 30x")
                break
        
        
        "Ambient Heat Loss"
#        T_wall = Q_dot_amb/UA_amb + T_amb
        
        
    
        "from preInjection state"
        M_dot_int_calc = V_dot_suc_comp/v_suc2 #determined from machine volume and speed
        
        "Mass Balance"
        Mbal = (m_ref + m_Leak) - M_dot_int_calc
        if i > 0:
            f = np.append(f, Mbal)              
            M_dot = np.append(M_dot, M_dot[i] - (f[i]*(M_dot[i]-M_dot[i-1]))/(f[i]-f[i-1]))            
        else:                    
            f = [Mbal] 
        if i>50:
            raise Exception("M_dot not converging after 50x")
    
    
    
    
    "model results"
    W_comp = W_comp[k] #W
    
#    print "Pint1: ",P_int1
    
#    print "m_suc: ",i
#    print "w_comp: ",k
#    print "m_inj: ",j
#    print "Pinj iter: ",s
#    
#    print "P_int1: ",P_int1
#    print "P_int2: ",P_int2
    
    return [W_comp,m_ref,m_inj,T_dis]




"========================================================================="


"Sample Condidtions to test model"
"""

start_time = time.time()

UA_amb_n = 5
VR1 = 1.5
VR2 = 1.25
A_suc = 0.00015
A_inj = 0.000003
A_dis = 0.0005583
A_leak = 0.0000009793
V_suc_comp = 0.000065
T_loss = 0.2
eta_m = 0.85
x_Pinj = 0.9


gas = 'REFPROP-R407C'
T_amb = 293
T_suc = -10 + 273
P_suc = PropsSI('P','T',T_suc - 5,'Q',0.5,gas)
T_inj = 25 + 273
P_inj = PropsSI('P','T',T_inj - 5,'Q',0.5,gas)
P_dis =  P_suc*6
N_comp = 3600



#[W,m_ref,m_inj,T] = Compressor(gas,T_amb,T_suc,T_inj,P_suc,P_inj,P_dis,N_comp,UA_amb_n,Vratio_preInj,Vratio_postInj,x_avg_inj,A_suc,A_inj,A_leak,A_dis,V_suc_comp,T_loss,eta_m)

[W,m_ref,m_inj,T] = Compressor(gas,T_amb,T_suc,T_inj,P_suc,P_inj,P_dis,N_comp,VR1,VR2,A_suc,A_inj,A_leak,A_dis,V_suc_comp,T_loss,eta_m,x_Pinj)


print "Time: ", time.time() - start_time
print "W: ",W
print "m_ref: ",m_ref
print "m_inj: ", m_inj
print "T_dis: ",T
"""