from __future__ import print_function
import CoolProp
import CoolProp.CoolProp as CP
from CoolProp.Plots import PropertyPlot
import numpy as np


print ("I'm testing python")
print (' ')

rho = CP.PropsSI('D', 'T', 298.15, 'P', 101325, 'Nitrogen')
print ('rho of Nirogen= ', rho)
print (' ')

ph_plot = PropertyPlot('Water', 'PH')
ph_plot.calc_isolines()
#ph_plot.show()
#ph_plot.savefig('images/enthalpy_pressure_graph_for_Water.pdf')

fluid = 'Water'
pressure_at_critical_point = CP.PropsSI(fluid,'pcrit')
# Massic volume (in m^3/kg) is the inverse of density
# (or volumic mass in kg/m^3). Let's compute the massic volume of liquid
# at 1bar (1e5 Pa) of pressure
vL = 1/CP.PropsSI('D','P',1e5,'Q',0,fluid)
# Same for saturated vapor
vG = 1/CP.PropsSI('D','P',1e5,'Q',1,fluid)

print ('vL = ', vL)
print ('vG = ', vG)

print(' ')

import sys; print (sys.path[0:4])
print('CoolProp version: ', CoolProp.__version__)
print('CoolProp gitrevision: ', CoolProp.__gitrevision__)
print('CoolProp fluids: ', CoolProp.__fluids__)
print('CoolProp path: ', CoolProp.__path__)

print(' ')
print('************ USING HEOS *************')  #This backend is the backend that provides properties using the CoolProp code
print(' ')
print('FLUID STATE INDEPENDENT INPUTS')
print('Critical Density Propane:', CP.PropsSI('HEOS::Propane', 'rhocrit'), 'kg/m^3')
print(' ')

print('TWO PHASE INPUTS (Pressure)')
print('Density of saturated liquid Propane at 101.325 kPa:', CP.PropsSI('D', 'P', 101325, 'Q', 0, 'HEOS::Propane'), 'kg/m^3')
print('Density of saturated vapor R290 at 101.325 kPa:', CP.PropsSI('D', 'P', 101325, 'Q', 1, 'HEOS::R290'), 'kg/m^3')
print(' ')
print('TWO PHASE INPUTS (Temperature)')
print('Density of saturated liquid Propane at 300 K:', CP.PropsSI('D', 'T', 300, 'Q', 0, 'HEOS::Propane'), 'kg/m^3')
print('Density of saturated vapor R290 at 300 K:', CP.PropsSI('D', 'T', 300, 'Q', 1, 'HEOS::R290'), 'kg/m^3')
      
p = CP.PropsSI('P', 'T', 300, 'D', 1, 'HEOS::Propane')
h = CP.PropsSI('H', 'T', 300, 'D', 1, 'HEOS::Propane')
T = CP.PropsSI('T', 'P', p, 'H', h, 'HEOS::Propane')
D = CP.PropsSI('D', 'P', p, 'H', h, 'HEOS::Propane')
print(' ')
print('SINGLE PHASE CYCLE (propane)')
print('T,D -> P,H', 300, ',', 1, '-->', p, ',', h)
print('P,H -> T,D', p, ',', h, '-->', T, ',', D)
      
#CP.enable_TTSE_LUT('Propane') #This is removed in CoolProp v5

try:
    print(' ')
    print('************ USING REFPROP ***************') #This backend provides a clean interface between CoolProp and REFPROP
    print(' ')
    print('TWO PHASE INPUTS (Pressure)')
    print('Density of saturated liquid Propane at 101.325 kPa:', CP.PropsSI('D', 'P', 101325, 'Q', 0, 'REFPROP::Propane'), 'kg/m^3')
    print('Density of saturated vapor Propane at 101.325 kPa:', CP.PropsSI('D', 'P', 101325, 'Q', 1, 'REFPROP::Propane'), 'kg/m^3')
    print('TWO PHASE INPUTS (Temperature)')
    print('Density of saturated liquid Propane at 300 K:', CP.PropsSI('D', 'T', 300, 'Q', 0, 'REFPROP::Propane'), 'kg/m^3')
    print('Density of saturated vapor Propane at 300 K:', CP.PropsSI('D', 'T', 300, 'Q', 1, 'REFPROP::Propane'), 'kg/m^3')
          
    p = CP.PropsSI('P', 'T', 300, 'D', 1, 'REFPROP::Propane')
    h = CP.PropsSI('H', 'T', 300, 'D', 1, 'REFPROP::Propane')
    T = CP.PropsSI('T', 'P', p, 'H', h, 'REFPROP::Propane')
    D = CP.PropsSI('D', 'P', p, 'H', h, 'REFPROP::Propane')
    print('SINGLE PHASE CYCLE (propane)')
    print('T,D -> P,H', 300, ',', 1, '-->', p, ',', h)
    print('P,H -> T,D', p, ',', h, '-->', T, ',', D)
except:
    print(' ')
    print('************ CANT USE REFPROP ************')
    print(' ')

print(' ')
print('************ CHANGE UNIT SYSTEM (default is SI) *************')
print(' ')
#CP.set_standard_unit_system(CoolProp.UNIT_SYSTEM_SI)
print('Vapor pressure of water at 373.15 K in SI units (Pa):', CP.PropsSI('P', 'T', 373.15, 'Q', 0, 'Water'))
#CP.set_standard_unit_system(CoolProp.UNIT_SYSTEM_KSI)
print('Vapor pressure of water at 373.15 K in kSI units (kPa):', CP.PropsSI('P', 'T', 373.15, 'Q', 0, 'Water'))

print(' ')
print('************ BRINES AND SECONDARY WORKING FLUIDS *************') #This backend provides the thermophysical properties for incompressible pure fluids, incompressible mixtures, and brines
print(' ')
print('Density of 50% (mass) ethylene glycol/water at 300 K, 101.325 kPa:', CP.PropsSI('D', 'T', 300, 'P', 101325, 'INCOMP::MEG-50%'), 'kg/m^3') #'EG-50%' is updated to 'INCOMP::MEG-50%' for CoolProp v5.x
print('Viscosity of Therminol D12 at 350 K, 101.325 kPa:', CP.PropsSI('V', 'T', 350, 'P', 101325, 'INCOMP::TD12'), 'Pa-s') #'TD12' is updated to 'INCOMP::TD12'

print(' ')
print('************ HUMID AIR PROPERTIES *************')
print(' ')
print('Humidity ratio of 50% rel. hum. air at 300 K, 101.325 kPa:', CP.HAPropsSI('W', 'T', 300, 'P', 101325, 'R', 0.5), 'kg_w/kg_da')
print('Relative humidity from last calculation:', CP.HAPropsSI('R', 'T', 300, 'P', 101325, 'W', CP.HAPropsSI('W', 'T',300, 'P', 101325, 'R', 0.5)), '(fractional)')


from CoolProp.CoolProp import cair_sat, HAPropsSI, PropsSI, get_fluid_param_string #,saturation_ancillary
Tin = 0+273.16
print('C = ', cair_sat(Tin)*1000)
print(PropsSI('C','T',Tin,'Q',0,'Water'))
print(HAPropsSI('C','T',Tin,'P', 101325,'R',1.0))
print(' ')


from scipy.optimize import newton                                               #see example page 31
print('newton method =',newton(lambda x: x**2-0.4, 1.5))
print(' ')
print('Exact method =', 0.4**0.5)
print(' ')

from scipy.optimize import bisect                                               #see example page 32
print('Bisection method =', bisect(lambda x: x**2-0.4, 0,4))
print(' ')

from scipy.optimize import fsolve                                               #see example page 33
func = lambda x: [x[0]**2-2*x[1]-2, x[0]+x[1]**2-1]
x =fsolve(func, [0,0])
print('fsolve x =',x)
print('fsolve f(x) = ', func(x))
print(' ')

print(1e-5)
print(1E-5)
print(' ')
print ('Molar = ', PropsSI('M','R410A')) #,'T',298.15,'P',101325

print ('surface tension = ', PropsSI('I','P',250000,'Q',0,'R407C')) #[N/m]
print ('surface tension = ', PropsSI('I','P',250000,'Q',0.5,'R407C')) #[N/m]
print ('surface tension = ', PropsSI('I','P',250000,'Q',1,'R407C')) #[N/m]
print('fluid string:',get_fluid_param_string('water','pure'))

#print('new surface tension method:',saturation_ancillary('R407C','I',1,'T', 250))
print ('T_HR = ', HAPropsSI('T','P',101325.0,'H',37972.967209365510,'R',1)-273.15,'degree C') 
print ('P_sat = ', PropsSI('Q','P', 445100,'H',244044.447331,'R404A'),'Pa') 
print ('P_sat = ', PropsSI('P','T', 323.15,'Q',0,'R407C'),'Pa') 
AS = CoolProp.AbstractState("INCOMP", "MEG")
#print (AS.get_mass_fractions())
AS.set_mass_fractions([0.2])
AS.update(CP.PT_INPUTS,120000,300)
print (AS.hmass())
print ('h_props = ', PropsSI('H','P', 120000,'T',300,'INCOMP::MEG-20%'),'J/kg')
print ('IncompressibleBackend' in AS.backend_name())