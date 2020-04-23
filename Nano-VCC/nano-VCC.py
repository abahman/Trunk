'''This code is for Direct Expansion in Cooling Mode'''
from __future__ import division, absolute_import, print_function
from ACHP.Cycle import DXCycleClass 
from ACHP.Plots import PlotsClass
from ACHP.SecondLawAnalysis import SecondLawAnalysisClass
from ACHP.convert_units import cfm2cms, C2K, W2BTUh

# Instantiate the cycle class
Cycle=DXCycleClass()

#--------------------------------------
#         Cycle parameters
#--------------------------------------
Cycle.Verbosity = 1 #the idea here is to have different levels of debug output 
Cycle.ImposedVariable = 'Subcooling'
Cycle.CycleType = 'DX'
#Cycle.Charge_target = 4.32 #kg #uncomment for use with imposed 'Charge'
Cycle.DT_sc_target = 5.7
Cycle.Mode='AC'
Cycle.Ref='R410A'
Cycle.Backend='HEOS' #Backend for refrigerant properties calculation: 'HEOS','TTSE&HEOS','BICUBIC&HEOS','REFPROP','SRK','PR'
Cycle.Oil = 'POE32'
Cycle.shell_pressure = 'low-pressure'
Cycle.EvapSolver = 'Moving-Boundary' #choose the type of Evaporator solver scheme ('Moving-Boundary' or 'Finite-Element')
Cycle.EvapType = 'Fin-tube' #if EvapSolver = 'Moving-Boundary', choose the type of evaporator ('Fin-tube' or 'Micro-channel')
Cycle.CondSolver = 'Moving-Boundary' #choose the type of Condenser solver scheme ('Moving-Boundary' or 'Finite-Element')
Cycle.CondType = 'Fin-tube' #if CondSolver = 'Moving-Boundary', choose the type of condenser ('Fin-tube' or 'Micro-channel')
Cycle.Update()

#--------------------------------------
#     Charge correction parameters (activate by setting Cycle.ImposedVariable to 'Charge' and Cycle.ChargeMethod to either 'One-point' or 'Two-point')
#--------------------------------------
Cycle.C = 0 #[kg]
Cycle.K = 0 #[kg]
Cycle.w_ref = 0 #[-]

#--------------------------------------
#       Compressor parameters
#--------------------------------------
# A 5 ton cooling capacity compressor map (Model: ZP61KCE-TFD 50Hz)
M=[227.786090375967,6.10564450069768,2.58098314158866,0.0562813131727362,-6.23385510899473E-4,-0.0145254762680343,6.15445078376047E-4,-3.92364697795786E-4,2.70574999280108E-4,-5.52227038499527E-5]
P=[-1135.36217441004,8.86411096612509,70.3931074036509,0.0883166966838579,-0.23656279540721,-0.447497536866393,-1.25297449463428E-4,-0.00118070778028288,9.28462197659111E-4,0.00214872885582762]

params={
        'M':M,
        'P':P,
        'Ref':Cycle.Ref, #Refrigerant
        'Oil':Cycle.Oil, #Compressor lubricant oil
        'shell_pressure':Cycle.shell_pressure, #Compressor shell pressure
        'fp':0.1, #Fraction of electrical power lost as heat to ambient 
        'Vdot_ratio': 1.0, #Displacement Scale factor
        'V_oil_sump':0.001685, #Volume of oil in the sump [m^3]
        'Verbosity': 0, # How verbose should the debugging be [0-10]
        }

Cycle.Compressor.Update(**params)

# T1 : Indoor temperature: 27C(DB), 19C(WB) 46.3%; Outdoor temperature: 35C(DB), 24C(WB)  39.6%
# T3 : Indoor temperature: 29C(DB), 19C(WB) 37.8%; Outdoor temperature: 46C(DB), 24C(WB)  14.6%
#--------------------------------------
#      Condenser parameters
#--------------------------------------
Cycle.Condenser.Fins.Tubes.NTubes_per_bank=30       #**[check]number of tubes per bank=row 
Cycle.Condenser.Fins.Tubes.Nbank=2                  #number of banks/rows 
Cycle.Condenser.Fins.Tubes.Ncircuits=8 
Cycle.Condenser.Fins.Tubes.Ltube=2.125              #**[check]This might be a bit shorter 
Cycle.Condenser.Fins.Tubes.OD=7/1000
Cycle.Condenser.Fins.Tubes.ID=7/1000-0.002          #**Assumed thickness = 1mm
Cycle.Condenser.Fins.Tubes.Pl=13.37/1000            #distance between center of tubes in flow direction 
Cycle.Condenser.Fins.Tubes.Pt=21/1000               #distance between center of tubes orthogonal to flow direction
Cycle.Condenser.Fins.Tubes.kw=385                   #wall thermal conductivity (i.e pipe material)

Cycle.Condenser.Fins.Fins.FPI=19.5385               #*Number of fins per inch (calculated from the manual, fin distance = 1.3mm)
Cycle.Condenser.Fins.Fins.Pd=0.001                  #*[assumed check]2* amplitude of wavy fin
Cycle.Condenser.Fins.Fins.xf=0.001                  #*[assumed check]1/2 period of fin
Cycle.Condenser.Fins.Fins.t=0.00011                 #*[assumed check]Thickness of fin material
Cycle.Condenser.Fins.Fins.k_fin=205                 #Thermal conductivity of fin material

Cycle.Condenser.Fins.Air.Vdot_ha=cfm2cms(4029)             #rated volumetric flowrate
Cycle.Condenser.Fins.Air.Tmean=C2K(35)#35
Cycle.Condenser.Fins.Air.Tdb=C2K(35)#35
Cycle.Condenser.Fins.Air.p=101325                   #Condenser Air pressure in Pa
Cycle.Condenser.Fins.Air.RH=0.396
Cycle.Condenser.Fins.Air.RHmean=0.396
Cycle.Condenser.Fins.Air.FanPower=320

Cycle.Condenser.FinsType = 'WavyLouveredFins'        #WavyLouveredFins, HerringboneFins, PlainFins
Cycle.Condenser.Verbosity=0

#--------------------------------------
# Evaporator Parameters 
#--------------------------------------
Cycle.Evaporator.Fins.Tubes.NTubes_per_bank=12      #**[check]number of tubes per bank=row
Cycle.Evaporator.Fins.Tubes.Nbank=4
Cycle.Evaporator.Fins.Tubes.Ltube=0.996             #*[check] could be a bit shorter
Cycle.Evaporator.Fins.Tubes.OD=9.52/1000
Cycle.Evaporator.Fins.Tubes.ID=9.52/1000-0.002      #*[assumed thickness of 1mm]
Cycle.Evaporator.Fins.Tubes.Pl=22/1000
Cycle.Evaporator.Fins.Tubes.Pt=25.4/1000
Cycle.Evaporator.Fins.Tubes.Ncircuits=7
Cycle.Evaporator.Fins.Tubes.kw=385                   #wall thermal conductivity (i.e pipe material)

Cycle.Evaporator.Fins.Fins.FPI=15.875
Cycle.Evaporator.Fins.Fins.Pd=0.001                 #*[assumed check]2* amplitude of wavy fin
Cycle.Evaporator.Fins.Fins.xf=0.001                 #*[assumed check]1/2 period of fin
Cycle.Evaporator.Fins.Fins.t=0.00011                #*[assumed check]Thickness of fin material
Cycle.Evaporator.Fins.Fins.k_fin=205

Cycle.Evaporator.Fins.Air.Vdot_ha=cfm2cms(1948)         #* Highest speed
Cycle.Evaporator.Fins.Air.Tmean=C2K(27)#27
Cycle.Evaporator.Fins.Air.Tdb=C2K(27)#27
Cycle.Evaporator.Fins.Air.p=101325                                              #Evaporator Air pressure in Pa
Cycle.Evaporator.Fins.Air.RH=0.463
Cycle.Evaporator.Fins.Air.RHmean=0.463
Cycle.Evaporator.Fins.Air.FanPower=740                  #* Highest speed

Cycle.Evaporator.FinsType = 'WavyLouveredFins'        #WavyLouveredFins, HerringboneFins, PlainFins
Cycle.Evaporator.Verbosity=0
Cycle.Evaporator.DT_sh=5                    #target superheat

# ----------------------------------
#       Expanison device Parameters
# ----------------------------------
params={
        'ExpType':'Ideal',     #expansion device type
#         'Tsh_static':4,             #static superheat
#         'Tsh_max':6,                #maximum superheat
#         'D':0.0006604,              #inside diameter [m]
#         'C':1.2656e-6,              #constant from manufacturer [m^2/K]
#         'Adj':0.7630,               #Adjust the diameter (tuning factor)
    }
Cycle.ExpDev.Update(**params)

# ----------------------------------
#       Line Set Parameters
# ----------------------------------
params={
        'L':7.6,
        'k_tube':0.19,
        't_insul':0.02,
        'k_insul':0.036,
        'T_air':297,
        'h_air':0.0000000001,
        'LineSetOption': 'Off'
        }

Cycle.LineSetLiquid.Update(**params)
Cycle.LineSetSuction.Update(**params)
Cycle.LineSetLiquid.OD=0.009525
Cycle.LineSetLiquid.ID=0.007986
Cycle.LineSetSuction.OD=0.01905
Cycle.LineSetSuction.ID=0.017526

# ----------------------------------
# ----------------------------------
#       Line Set Discharge Parameters
# ----------------------------------
# ----------------------------------
params={
        'L':0.3,                #tube length in m
        'k_tube':0.19,
        't_insul':0, #no insulation
        'k_insul':0.036,
        'T_air':297,
        'h_air':0.0000000001,
        'LineSetOption': 'Off'
        }
  
Cycle.LineSetDischarge.Update(**params)
Cycle.LineSetDischarge.OD=0.009525
Cycle.LineSetDischarge.ID=0.007986


# Now solve
from time import time
t1=time()
Cycle.PreconditionedSolve()
print ('Took '+str(time()-t1)+' seconds to run Cycle model')
print ('Cycle Capacity is '+str(W2BTUh(Cycle.Capacity))+' Btu/hr')
print ('Cycle COP is '+str(Cycle.COSP))
print ('Cycle refrigerant charge is '+str(Cycle.Charge)+' kg')

# Now run Second Law analysis
SecondLaw = SecondLawAnalysisClass()
SecondLaw.DXCycle(Cycle)
print ('Cycle Second Law is '+str(SecondLaw.epsilon_sys))

# Now do cycle plotting
plot = PlotsClass()
plot.TSOverlay(Cycle)
plot.PHOverlay(Cycle)