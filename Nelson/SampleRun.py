import CompressorVaporInj_407C




"System Conditions"
P_evap=445.1 #Evap. pressure [kPa]
P_cond=2221.1 #Cond. pressure [kPa]
P_inj=859.1 #Inj. pressure [kPa]
T_suc=283.2 #Suction temp [K]
T_inj=298.2 #Injection temp [K]

N_c = 3600 #compressor speed [RPM]


"Optimized Compressor Parameters"
VR1 = 1.400305991  #volume ratio before injection
VR2 = 1.431429505 #volume ratio after injection
A_suc =8.99E-05 #main suction port area [m^2]
A_inj = 6.57E-06 #injection port area [m^2]
A_dis = 1.30E-05 #discharge port area [m^2]
A_leak = 4.62E-07 #leakage area [m^2]
V_suc_comp = 6.68E-05 #suction volume [m^3]
T_loss = 0.616731207 #frictional torque loss [N-m]
eta_m = 0.99 # motor efficiency


[W_dot,m_dot_ref,m_dot_inj,T_dis] = CompressorVaporInj_407C.Compressor('REFPROP-R407C',T_suc,T_inj,P_evap,P_inj,P_cond,N_c,VR1,VR2,A_suc,A_inj,A_leak,A_dis,V_suc_comp,T_loss,eta_m)

print "Work [W]: ",W_dot
print "Suction massFlow [kg/s]: ",m_dot_ref
print "Injection massFlow [kg/s]: ",m_dot_inj
print "Discharge Temperature [K]: ",T_dis

