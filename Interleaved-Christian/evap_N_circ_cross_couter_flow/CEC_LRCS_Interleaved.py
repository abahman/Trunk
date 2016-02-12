import sys
#sys.path.append("C:\Users\bachc\Desktop\achp\trunk\PyACHP")  #LRCS-folder is already appended, no need to add again
print sys.path
from MultiCircuitEvaporator_w_circuitry import MultiCircuitEvaporator_w_circuitryClass
from ACHPTools import simple_write_to_file,Write2CSV
from LRCS_evaporator import Evaporator_LRCS
from DOE_HP_evaporator import get_sim_flowrate_SH
from CoolProp.CoolProp import Props
from convert_units import *
import numpy as np
import copy
print ">>> done with imports in CEC_LRCS_Interleaved"

def CEC_LRCS_recircuit(mode=0,plotit=False,maldistrib=0,type='wCircuitry',equalflow=False):
    #using simple definition that  only allows for 1 group in first row and 1 group in second row
    #mode: 0-normal layout, 1-interleaved layout
    from plot_circuitry import plot_circuitry
    
    #get LRCS Fins-class
    Evap=Evaporator_LRCS()
    print "orginal Evap.Fins.Air.Vdot_ha from LRCS Evaporator",Evap.Fins.Air.Vdot_ha
    Fins=copy.deepcopy(Evap.Fins)
    #swap the evaporator against a multi circuited one, if applicable
    if type=='MCE':
        from MultiCircuitEvaporator import MultiCircuitEvaporatorClass
        Evap=MultiCircuitEvaporatorClass() #use multi circuit evaporator
    elif type=='wCircuitry':
        Evap=MultiCircuitEvaporator_w_circuitryClass() #use multi circuit evaporator
    elif type=='standard':
        Evap=Evaporator_LRCS()  #no change to previous
        print 'keeping standard evaporator"'
    else:
        print "unsupported evaporator type"
        raise()
    Fins.Air.RH=0.1
    Fins.Air.RHmean=0.1
    params={
         'Verbosity':0,
         'Ref':'R404a',
         'Fins':Fins
    }
    Evap.Update(**params)
    if type=='wCircuitry':
        #need to modify some parts
        Evap.Fins.Air.Vdot_ha=Evap.Fins.Air.Vdot_ha*2.0 #need to double flowrate comopared to "normal" evaporator since we have same flow for first and second tube sheet
        N_tubes_total=Evap.Fins.Tubes.Nbank*Evap.Fins.Tubes.NTubes_per_bank
        N_tubes_cell=N_tubes_total/(Evap.Fins.Tubes.Ncircuits*2)
        Evap.Circs_tubes_per_bank=np.array([N_tubes_cell]*Evap.Fins.Tubes.Ncircuits*2)  #custom circuitry
        if np.sum(Evap.Circs_tubes_per_bank)!=N_tubes_total:
            print "something wrtong with circuitry - needs manual attention", np.sum(Evap.Circs_tubes_per_bank),N_tubes_total
        Evap.Fins.Tubes.Ncircuits=16
        Evap.Custom_circuitry=True  #let evaporator know that it should calculate the custom circuitry
    #determine maldistribution
    if maldistrib==1:
        print "applying type 1 rectangular airflow maldistribution",
        typeA=np.array(([0.9]*4+[1.1]*4)*2)/16.0
        print typeA,typeA.shape,typeA.sum()-1.0
    if maldistrib==2:
        print "applying type 2 rectangular airflow maldistribution",
        typeA=np.array(([0.8]*4+[1.2]*4)*2)/16.0
        print typeA,typeA.shape,typeA.sum()-1.0
    if maldistrib==3:
        print "applying type 3 rectangular airflow maldistribution",
        typeA=np.array(([0.7]*4+[1.3]*4)*2)/16.0
        print typeA,typeA.shape,typeA.sum()-1.0
        
    if maldistrib!=0:  #normalize airflow distribution factors
        Evap.Vdot_ha_coeffs= Evap.Circs_tubes_per_bank*1.0/np.sum(Evap.Circs_tubes_per_bank)  #normalized value to begin with
        typeA=Evap.Vdot_ha_coeffs* typeA                 #consider nominal per circuit flowrate
        Evap.Vdot_ha_coeffs=typeA/np.sum(typeA)  #normalize
    
    #air inlet is same for all circuits
    air_pre=[-1,-1,-1,-1,-1,-1,-1,-1,0 , 1 , 2 , 3 , 4 , 5 , 6 , 7]  #define air inlet to circuits
    #define ref inlet to circuits
    if mode==0:
         ref_pre=[0+8 , 1+8 , 2+8 , 3+8 , 4+8 , 5+8 , 6+8 , 7+8,-1,-1,-1,-1,-1,-1,-1,-1] #non-interleaved circuits, model 1
         layoutname='non-interleaved'
    if mode==1:
        ref_pre=[4,5,6,7,0,1,2,3]+[-1]*8  #interleaved circuits, model 1
        layoutname='interleaved'
    kwargs={'air_pre':air_pre,  #define air inlet to circuits
            'ref_pre':ref_pre,  #define ref inlet to circuits
            'Custom_circuitry':True   #this attribute has to be set for the calculate function
            }
    Evap.Update(**kwargs)
    Evap.TestDescription=layoutname+" maldistribution-type "+str(maldistrib)
    
    #adjust refrigerant flow factors, if applicable
    if equalflow: #same refrigerant and airside flowrates, as for hybrid control (only makes sense for parallel circuitry
        if mode!=0:
            print "this doen't make sense unless the circuitry is a standard normal one"
            raise()
        else:
            Evap.mdot_r_coeffs=Evap.Vdot_ha_coeffs
            
    if plotit: plot_circuitry(air_pre,ref_pre[8:]+ref_pre[0:8],2,layoutname=layoutname)  #direction swapped for plotting
    return Evap

def CEC_LRCS_recircuit_debug(mode=0,plotit=False,maldistrib=0,type='wCircuitry'):
    #debug version with reduced number of circuits
    
    #using simple definition that  only allows for 1 group in first row and 1 group in second row
    #mode: 0-normal layout, 1-interleaved layout
    from plot_circuitry import plot_circuitry
    
    #get LRCS Fins-class
    Evap=Evaporator_LRCS()
    Fins=copy.deepcopy(Evap.Fins)
    #swap the evaporator against a multi circuited one, if applicable
    if type=='MCE':
        from MultiCircuitEvaporator import MultiCircuitEvaporatorClass
        Evap=MultiCircuitEvaporatorClass() #use multi circuit evaporator
    elif type=='wCircuitry':
        Evap=MultiCircuitEvaporator_w_circuitryClass() #use multi circuit evaporator
    elif type=='standard':
        Evap=Evaporator_LRCS()  #no change to previous
        print 'keeping standard evaporator"'
    else:
        print "unsupported evaporator type"
        raise
    Fins.Air.RH=0.1
    Fins.Air.RHmean=0.1
    params={
         'Verbosity':0,
         'Ref':'R404a',
         'Fins':Fins
    }
    Evap.Update(**params)
    if type=='wCircuitry':
        #need to modify some parts
        Evap.Fins.Air.Vdot_ha=(Evap.Fins.Air.Vdot_ha*2.0)/20.0 #need to double flowrate comopared to "normal" evaporator since we have same flow for first and second tube sheet
        N_tubes_total=Evap.Fins.Tubes.Nbank*Evap.Fins.Tubes.NTubes_per_bank
        N_tubes_cell=N_tubes_total/(2)/8.0
        Evap.Circs_tubes_per_bank=np.array([N_tubes_cell]*2)  #custom circuitry
        if np.sum(Evap.Circs_tubes_per_bank)!=N_tubes_total:
            print "something wrtong with circuitry - needs manual attention", np.sum(Evap.Circs_tubes_per_bank),N_tubes_total
        Evap.Fins.Tubes.Ncircuits=2
        Evap.Custom_circuitry=True  #let evaporator know that it should calculate the custom circuitry
    #determine maldistribution (do not use at this point
    if maldistrib==1:
        print "applying type 1 rectangular airflow maldistribution",
        typeA=np.array(([0.9]*4+[1.1]*4)*2)/16.0
        print typeA,typeA.shape,typeA.sum()-1.0
    if maldistrib==2:
        print "applying type 2 rectangular airflow maldistribution",
        typeA=np.array(([0.8]*4+[1.2]*4)*2)/16.0
        print typeA,typeA.shape,typeA.sum()-1.0
    if maldistrib==3:
        print "applying type 3 rectangular airflow maldistribution",
        typeA=np.array(([0.7]*4+[1.3]*4)*2)/16.0
        print typeA,typeA.shape,typeA.sum()-1.0
        
    if maldistrib!=0:  #normalize airflow distribution factors
        print "applying refrigerant side maldistribution"
        typeA=Evap.Vdot_ha_coeffs* typeA                 #consider nominal per circuit flowrate
        Evap.Vdot_ha_coeffs=typeA/np.sum(typeA)  #normalize
    
    #air inlet is same for all circuits
    air_pre=[-1,0]  #define air inlet to circuits
    #define ref inlet to circuits
    if mode==0:
         ref_pre=[1,-1]#non-interleaved circuits, model 1
         layoutname='non-interleaved'
    if mode==1:
        ref_pre=[1,-1]  #interleaved circuits, model 1 (doesn't make a difference for a single circuit)
        layoutname='interleaved'
    kwargs={'air_pre':air_pre,  #define air inlet to circuits
            'ref_pre':ref_pre,  #define ref inlet to circuits
            'Custom_circuitry':True   #this attribute has to be set for the calculate function
            }
    Evap.Update(**kwargs)
    Evap.TestDescription=layoutname+" maldistribution-type "+str(maldistrib)
    
    if plotit: plot_circuitry(air_pre,ref_pre[8:]+ref_pre[0:8],2,layoutname=layoutname)  #direction swapped for plotting
    return Evap

def get_sim_flowrate_SH(DT_SH_target,m_dot_exp,verbouseness=1,type='wCircuitry',plotit=False,maldistrib=1,append=True,equalflow=False,mode=0):
    #find mass flowrate to obtain same exit superheat
    #m_dot_exp is used as guess value for the simulation
    #returns solved evaporator instance
    
    from scipy.optimize import fsolve
    def objective(x,type,maldistrib,equalflow,mode,save_results=False):
        x0=x[0]
        if x0<0.001:
            print "DOE_HP_evaporator det_sim_flowrate_SH: x0 is",x0
            x0=0.001
        print 'x0',x0
        
        Evap=CEC_LRCS_recircuit(type=type,plotit=False,maldistrib=maldistrib,equalflow=equalflow,mode=mode)
        Tin_r=C2K(42.9)
        condpsat_r=1987
        Evap.hin_r=Props('H','T',Tin_r,'P',condpsat_r,Evap.Ref)*1000.0     #from 35C non-maldistributed hybrid test, \\tools.ecn.purdue.edu\bachc\cec-share\00 Large Room Cooling System\02Hybrid_Control_Testing\be_2C_35C_cont_Hybrid_reduced.xlsx
        Evap.psat_r=497.0 #evaporator out, since inlet unknown
        
        kwargs={'mdot_r': x0}
        Evap.Update(**kwargs)
        Evap.x_start=LRCS_start_value_guesstimator(Evap)
        Evap.Calculate()
        T_sat_out_r=Props('T','Q',1.0,'P',Evap.Pout_r,Evap.Ref)
        if save_results:
            Write2CSV(Evap,open('Debug_LRCS_interleaved.csv','w'),append=append)
        return [Evap.hout_r/1000.0-Props('H','T',DT_SH_target+T_sat_out_r,'P',Evap.Pout_r,Evap.Ref)]
    
    x0=m_dot_exp #some of the points resulted in 0 superheat, therefore reduce flowrate as a starting point
    if plotit:
        xi = np.linspace(1.1*m_dot_exp,0.6*m_dot_exp,5)
        yi=np.zeros(len(xi))
        import matplotlib.pyplot as plt
        for i in range(0,len(xi)):
            print "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx  preparing plot run",i,"of",len(xi),"using m_dot",xi[i],"which is",xi[i]/m_dot_exp,"times the experimental mass flowrate"
            yi[i]= objective([xi[i]],type,maldistrib,equalflow,mode,save_results=True)[0]
        plt.plot(xi,yi,marker='o')
        plt.show()
    x0=0.105
    x0=fsolve(objective, [x0],args=(type,maldistrib,equalflow,mode))
    resids=objective(x0,type,maldistrib,equalflow,mode,save_results=True)

def LRCS_start_value_guesstimator(Evap):
    #returns guess for starting value of h_r in to first row
    #completely empirical
    x_start=[]
    for n in range(0,len(Evap.air_pre)/2): 
        if Evap.mdot_r<0.08:
            x_start.append(Props('H','Q',0.5,'P',Evap.psat_r,Evap.Ref)*1000.0)
        else:
            x_start.append(Props('H','Q',1.0,'P',Evap.psat_r,Evap.Ref)*1000.0)
    return x_start

def run_95F_case(type='wCircuitry'):
    #prepare evaporator
    Evap=CEC_LRCS_recircuit(type=type,plotit=False,maldistrib=0,mode=0)
    #inlet enthalpy
    Tin_r=C2K(42.9)
    Evap.mdot_r=91.484/1000.0*(2.0) #divide by 8 for debug-evaporator  #\\tools.ecn.purdue.edu\bachc\cec-share\00 Large Room Cooling System\02Hybrid_Control_Testing\be_2C_35C_cont_Hybrid_reduced.xlsx
    #Evap.mdot_r=0.108333333333
    condpsat_r=1987
    Evap.hin_r=Props('H','T',Tin_r,'P',condpsat_r,Evap.Ref)*1000.0     #from 35C non-maldistributed hybrid test, \\tools.ecn.purdue.edu\bachc\cec-share\00 Large Room Cooling System\02Hybrid_Control_Testing\be_2C_35C_cont_Hybrid_reduced.xlsx
    #check if conditions are subcooled
    print "Evap.hin_r",Evap.hin_r,"saturated enthalpy at inlet pressure",Props('H','Q',0.0,'P',condpsat_r,Evap.Ref),"resulting subcooling",Props('T','Q',0.0,'P',condpsat_r,Evap.Ref)-Tin_r
    Evap.psat_r=497.0 #evaporator out, since inlet unknown
    print 
    #set low humidity to avoid instabilities
    if True: #calculate single point
    #set guess value
        #Evap.x_start=LRCS_start_value_guesstimator(Evap)
        Evap.Calculate()
        Write2CSV(Evap,open('Debug_LRCS_interleaved.csv','w'),append=False)
        print Evap.OutputList()
        print "calculated mass flowrate",Evap.mdot_r,"calculated outlet superheat",Evap.Tout_r-Props('T','Q',1.0,'P',Evap.psat_r+Evap.DP_r/1000.0,Evap.Ref)
        print "since this was to easy, now lets find the outlet superheat"
    #from DOE_HP_evaporator import get_sim_flowrate_SH
    
    #print get_sim_flowrate_SH(5.0,Evap.mdot_r,verbouseness=0,type='wCircuitry',plotit=True,maldistrib=0,append=False,mode=1)
    #print get_sim_flowrate_SH(5.0,Evap.mdot_r,verbouseness=0,type='wCircuitry',plotit=False,maldistrib=0,append=False)
    #print get_sim_flowrate_SH(5.0,Evap.mdot_r,verbouseness=0,type='wCircuitry',plotit=False,maldistrib=1,append=True)
    #print get_sim_flowrate_SH(5.0,Evap.mdot_r,verbouseness=0,type='wCircuitry',plotit=False,maldistrib=1,append=True)

    
if __name__=='__main__':
   run_95F_case(type='wCircuitry')