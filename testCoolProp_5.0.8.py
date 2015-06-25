'''Examples from documentation of CoolProps 5.0.8'''

from CoolProp.CoolProp import PropsSI
import CoolProp; print CoolProp.__version__, CoolProp.__gitrevision__
print PropsSI('D', 'T', 298.15, 'P', 100e5, 'CO2')
print PropsSI('H', 'T', 298.15, 'Q', 1, 'R134a')
print ' '

from CoolProp.HumidAirProp import HAPropsSI
# Enthalpy (J per kg dry air) as a function of temperature, pressure,
#    and relative humidity at STP
h = HAPropsSI('H','T',298.15,'P',101325,'R',0.5); print 'h =', h

# Temperature of saturated air at the previous enthalpy
T = HAPropsSI('T','P',101325,'H',h,'R',1.0); print 'T =', T

# Temperature of saturated air - order of inputs doesn't matter
T = HAPropsSI('T','H',h,'R',1.0,'P',101325); print 'T =', T
print ' '


import CoolProp as CP
print CP.__version__

import CoolProp.CoolProp as CP
fluid = 'Water'
pressure_at_critical_point = CP.PropsSI(fluid,'pcrit')
# Massic volume (in m^3/kg) is the inverse of density
# (or volumic mass in kg/m^3). Let's compute the massic volume of liquid
# at 1bar (1e5 Pa) of pressure
vL = 1/CP.PropsSI('D','P',1e5,'Q',0,fluid)
print 'vL =', vL
# Same for saturated vapor
vG = 1/CP.PropsSI('D','P',1e5,'Q',1,fluid)
print 'vG =', vG



'''plots examples'''
'''Warning: for each set of examples take out the comment #
    and run indivually otherwise the plots will MISSED up'''
import CoolProp.Plots as CPP
from CoolProp.Plots import PropsPlot
from matplotlib import pyplot
from CoolProp.Plots import Ts, drawIsoLines


# Ref = 'n-Pentane'
# ax = Ts(Ref)
# ax.set_xlim([-0.5, 1.5])
# ax.set_ylim([300, 530])
# quality = drawIsoLines(Ref, 'Ts', 'Q', [0.3, 0.5, 0.7, 0.8], axis=ax)
# isobars = drawIsoLines(Ref, 'Ts', 'P', [100, 2000], num=5, axis=ax)
# isochores = drawIsoLines(Ref, 'Ts', 'D', [2, 600], num=7, axis=ax)
# pyplot.show()

#   
# ph_plot_water = CPP.PropsPlot('Water','Ph')
# ph_plot_water.savefig('images/enthalpy_pressure_graph_for_Water2.pdf')

# ts_plot_R290 = PropsPlot('R290', 'Ts')
# ts_plot_R290.savefig('images/plot_Ts_R290.pdf')

# ph_plot_R410A = PropsPlot('R410A', 'Ph')
# ph_plot_R410A.savefig('images/plot_pH_R410.pdf')

# ref_fluid = 'n-Pentane'
# ts_plot = PropsPlot(ref_fluid, 'Ts')
# ts_plot.draw_isolines('Q', [0.3, 0.5, 0.7, 0.8])
# ts_plot.draw_isolines('P', [100, 2000], num=5)
# ts_plot.draw_isolines('D', [2, 600], num=7)
# ts_plot.set_axis_limits([-2, 1.5, 200, 500])
# ts_plot.savefig('images/n-pentane.pdf')


# ts_plot_water = PropsPlot('Water', 'Ts')
# ts_plot_water.title('Ts Graph for Water')
# ts_plot_water.xlabel(r's $[{kJ}/{kg K}]$')
# ts_plot_water.ylabel(r'T $[K]$')
# ts_plot_water.grid()
# ts_plot_water.savefig('images/Water_Ts.pdf')

# ph_plot_water = PropsPlot('Water', 'Ph')
# ax = ph_plot_water.axis
# ax.set_yscale('log')
# ax.text(400, 5500, 'Saturated Liquid', fontsize=15, rotation=40)
# ax.text(2700, 3500, 'Saturated Vapour', fontsize=15, rotation=-100)
# ph_plot_water.savefig('images/Water_Ph.pdf')

ref_fluid = 'R600a'
fig = pyplot.figure(1, figsize=(10, 10), dpi=100)
for i, gtype in enumerate(['PT', 'PD', 'PS', 'PH', 'TD', 'TS', 'HS']):
    ax = pyplot.subplot(4, 2, i+1)
    if gtype.startswith('P'):
        ax.set_yscale('log')
    props_plot = PropsPlot(ref_fluid, gtype, axis=ax)
    props_plot.title(gtype)
    props_plot._draw_graph()
fig.set_tight_layout(True) #pyplot.tight_layout()
fig.savefig('images/comined_R600a.pdf') #pyplot.savefig('images/comined_R600a.pdf')
