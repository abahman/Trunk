"""
This is the definition file for the evaporator of the DOE-vapor-injected heat pump

created: 2013-05-12 by bachc, based on ACHP evaporator
"""

from Evaporator import EvaporatorClass
"""Option to swap against muly circuit evaporator by changing next line"""
from MultiCircuitEvaporator_w_circuitry import MultiCircuitEvaporator_w_circuitryClass
from FinCorrelations import FinInputs
from convert_units import *
import numpy as np
from CoolProp.CoolProp import Props
import scipy.optimize
from CoolProp.HumidAirProp import HAProps
import matplotlib.pyplot as plt
from ACHPTools import simple_write_to_file,Write2CSV   #tools

def Evaporator_DOE_HP(type='wCircuitry'):
    #define parameters for evaporator as used in DOE HP
    Evaporator=EvaporatorClass()
    
    
    Fins=FinInputs()
     
    #--------------------------------------
    #--------------------------------------
    #           Evaporator
    #           -> see Condenser and GUI for explanations
    #--------------------------------------
    #--------------------------------------
    Fins.Tubes.NTubes_per_bank=48
    Fins.Tubes.Nbank=2
    Fins.Tubes.Ncircuits=10
    Fins.Tubes.Ltube=in2m(105.) #measured fin pack length
    #tube diameters from D:\Purdue\CEC BERG PROJECT\bachc\components\heat pump\ACHP\25HCC560-FV4C006-Data.xlsx
    Fins.Tubes.OD=mm2m(7.3) #measured 25HCC560 (same for HNB560); data for HCC unit indicated 7mm, actually larger
    Fins.Tubes.ID=mm2m(6.3) #measured 25HCC560 (assuming same for HNB560); data for HCC unit didn't make sense
    Fins.Tubes.Pl=cm2m(2.)       #depth of each fin-sheet (measured)
    Fins.Tubes.Pt=in2m(0.83)   #48tubes, 40-1/8 total distance between first and last tube
    Fins.Fins.FPI=20 #from datasheet
    Fins.Fins.Pd=0.001  #since fins are lanced, this value is meaningless - tune to fit to data
    Fins.Fins.xf=0.001 
    Fins.Fins.t=mm2m(0.11)   #measurement with callipper, 
    Fins.Fins.k_fin=237 #Pure aluminum (Incropera, Sixth edition)
    Fins.Air.Vdot_ha=cfm2cms(4668) #high flow value of HNB960  D:\Purdue\DOE Project\HP-design\datasheets\HNB960A300.pdf
    #Fins.Air.Vdot_ha=cfm2cms(4209.0) #high flow value of HNB960
    Fins.Air.Tmean=C2K(8.33)  #update according to your requirement
    Fins.Air.Tdb=Fins.Air.Tmean
    Fins.Air.p=101.325      #Air pressure
    Fins.Air.RH=0.48
    Fins.Air.RHmean=0.48
    Fins.Air.FanPower=HP2W(1.0/5.0)  #rated HP according to HNB960 datasheet

     #now swap the evaporator against a multi circuited one
    if type=='MCE':
        from MultiCircuitEvaporator import MultiCircuitEvaporatorClass
        Evaporator=MultiCircuitEvaporatorClass() #use multi circuit evaporator
    elif type=='wCircuitry':
        Evaporator=MultiCircuitEvaporator_w_circuitryClass() #use multi circuit evaporator
    elif type=='standard':
        Evaporator=EvaporatorClass() #use "normal" evaporator
    else:
        print "unsupported evaporator type"
        raise
    Evaporator.Fins=Fins
    params={
         'Verbosity':0,
         'DT_sh':5.0,  #initial setting, update according to requirement
         'Ref':'R410A'
    }
    Evaporator.Update(**params)
    return Evaporator
def DOE_HP_circuitry2():
    #this type uses a more flexible circuitry definition
    from plot_circuitry import plot_circuitry2
    n_banks=2
    height=48
    row0=[(1,0),(2,0),(3,0),(-1,0),(-1,0),(4,0),(5,0),(6,0),(7,0),(8,0),(11,0),(12,0),(13,0),(-1,0),(-1,0),(14,0),(15,0),(16,0),(17,0),(18,0),(21,0),(22,0),(23,0),(-1,0),(25,0),(26,0),(27,0),(-1,0),(-1,0),
          (28,0),(29,0),(30,0),(31,0),(32,0),(35,0),(36,0),(37,0),(-1,0),(-1,0),(38,0),(39,0),(40,0),(41,0),(42,0),(-1,0),(44,0),(45,0),(46,0)]
    row1=[(0,0),(0,1),(1,1),(2,1),(3,1),(4,1),(7,1),(8,1),(9,1),(9,0),(10,0),(10,1),(11,1),(12,1),(13,1),(14,1),(17,1),(18,1),(19,1),(19,0),(20,0),(20,1),(21,1),(22,1),(24,0),(24,1),(25,1),(26,1),
          (27,1),(28,1),(31,1),(32,1),(33,1),(33,0),(34,0),(34,1),(35,1),(36,1),(37,1),(38,1),(41,1),(42,1),(43,1),(43,0),(45,1),(46,1),(47,1),(47,0)]
    for i in range(len(row1)):
        print i+1,"previous",row1[i][0]+1,row1[i][1]
    print len(row0),len(row1)
    ref_pre=row0+row1
    plot_circuitry2(ref_pre,n_banks,layoutname='undefined.png')
def DOE_HP_recircuit(mode=0,plotit=False,maldistrib=False):
    #simple definitian that  only allows for 1 group in first row and 1 group in second row
    from plot_circuitry import plot_circuitry
    
    #get evaporator
    Evap=Evaporator_DOE_HP()
    #modify air distribution factors and assignment of tubes per evaporator to fit Layout
    Evap.Circs_tubes_per_bank=np.array([4,6,4,6,4,4,6,4,6,4,6,4,6,4,4,6,4,6,4,4])
    Evap.Vdot_ha_coeffs= Evap.Circs_tubes_per_bank*1.0/(np.sum( Evap.Circs_tubes_per_bank))*1.0
    #determine maldistribution
    if maldistrib==1:
        print "applying type 1 rectangular airflow maldistribution",
        typeA=np.array(([0.9]*5+[1.1]*5)*2)/20.0
        print typeA,typeA.shape,typeA.sum()-1.0
    if maldistrib==2:
        print "applying type 2 rectangular airflow maldistribution",
        typeA=np.array(([0.8]*5+[1.2]*5)*2)/20.0
        print typeA,typeA.shape,typeA.sum()-1.0
    if maldistrib==3:
        print "applying type 3  rectangular airflow maldistribution",
        typeA=np.array(([0.7]*5+[1.3]*5)*2)/20.0
        print typeA,typeA.shape,typeA.sum()-1.0
        
    if maldistrib!=0:  #normalize airflow distribution factors
        typeA=Evap.Vdot_ha_coeffs* typeA                 #consider nominal per circuit flowrate
        Evap.Vdot_ha_coeffs=typeA/np.sum(typeA)  #normalize
    Evap.Fins.Air.Vdot_ha=cfm2cms(4668)*2 #need to double flow since we have same flow for first and second tube sheet; high flow value of HNB960  D:\Purdue\DOE Project\HP-design\datasheets\HNB960A300.pdf
    Evap.Fins.Tubes.Nbank=1  #tuning to obtain same suface area as for 2 bank-case
    #The next line doesn't make sense, since the number of tubes should remain the same for both case.
    #Evap.Fins.Tubes.Ltube=Evap.Fins.Tubes.Ltube/2.0  #since we have double the circuits, we need to make the tubes shorter
                                                                                    #alternative would be to change the numbers of tubes per sheet, which would be more confusing
    n_banks=2
    height=48
    
    #air inlet is same for all circuits
    air_pre=[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9]  #define air inlet to circuits
    #define ref inlet to circuits
    if mode==0:
         ref_pre=[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9] #non-interleaved circuits, model 1
         layoutname='non-interleaved'
    if mode==1:
        ref_pre=[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 6,7,8,9,0,1,2,3,4,5]  #interleaved circuits, model 1
        layoutname='interleaved'
    kwargs={'air_pre':air_pre,  #define air inlet to circuits
            'ref_pre':ref_pre,  #define ref inlet to circuits
            'Custom_circuitry':True   #this attribute has to be set for the calculate function
            }
    Evap.Update(**kwargs)
    Evap.TestDescription=layoutname+" maldistribution-type "+str(maldistrib)
    Evap.Fins.Tubes.Ncircuits=20
    
    if plotit: plot_circuitry(air_pre,ref_pre,n_banks,layoutname=layoutname)
    return Evap
def DOE_HP_recircuit_single_slab(mode=0,plotit=False):
    #single slab for comparison (cut off second slab)
    #simple definition that  only allows for 1 group in first row and 1 group in second row
    from plot_circuitry import plot_circuitry
    
    #get evaporator
    Evap=Evaporator_DOE_HP()
    #modify air distribution factors and assignment of tubes per evaporator to fit Layout
    Evap.Circs_tubes_per_bank=np.array([4,6,4,6,4,4,6,4,6,4])
    Evap.Vdot_ha_coeffs= Evap.Circs_tubes_per_bank*1.0/(np.sum( Evap.Circs_tubes_per_bank))*1.0
    Evap.Fins.Air.Vdot_ha=cfm2cms(4668) #need to double flow since we have same flow for first and second tube sheet; high flow value of HNB960  D:\Purdue\DOE Project\HP-design\datasheets\HNB960A300.pdf
    Evap.Fins.Tubes.Nbank=1  #tuning to obtain same suface area as for 2 bank-case
    #The next line doesn't make sense, since the number of tubes should remain the same for both case.
    #Evap.Fins.Tubes.Ltube=Evap.Fins.Tubes.Ltube/2.0  #since we have double the circuits, we need to make the tubes shorter
                                                                                    #alternative would be to change the numbers of tubes per sheet, which would be more confusing
    n_banks=1
    height=48
    
    #air inlet is same for all circuits
    air_pre=[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]  #define air inlet to circuits
    #define ref inlet to circuits
    if mode==0:
         ref_pre=[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1] #non-interleaved circuits, model 1
         layoutname='non-interleaved'
    if mode==1:
        ref_pre=[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]  #interleaved circuits, model 1
        layoutname='interleaved'
    kwargs={'air_pre':air_pre,  #define air inlet to circuits
            'ref_pre':ref_pre,  #define ref inlet to circuits
            'Custom_circuitry':True   #this attribute has to be set for the calculate function
            }
    Evap.Update(**kwargs)
    Evap.Fins.Tubes.Ncircuits=10
    
    if plotit: plot_circuitry(air_pre,ref_pre,n_banks,layoutname=layoutname)
    return Evap
def get_sim_flowrate_SH(Evap,DT_SH_target,m_dot_exp,verbouseness=1,plotit=False):
    #find mass flowrate to obtain same exit superheat
    #m_dot_exp is used as guess value for the simulation
    #returns solved evaporator instance
    from scipy.optimize import fsolve
    def objective(x):
        x0=x[0]
        if x0<0.001:
            print "DOE_HP_evaporator det_sim_flowrate_SH: x0 is",x0
            x0=0.001
        #print 'x0',x0
        kwargs={'mdot_r': x0}
        Evap.Update(**kwargs)
        Evap.Calculate()
        T_sat_out_r=Props('T','Q',1.0,'P',Evap.Pout_r,Evap.Ref)
        return [Evap.hout_r/1000.0-Props('H','T',DT_SH_target+T_sat_out_r,'P',Evap.Pout_r,Evap.Ref)]
    
    x0=m_dot_exp*0.7 #some of the points resulted in 0 superheat, therefore reduce flowrate as a starting point
    if plotit:
        xi = np.linspace(1.1*m_dot_exp,0.8*m_dot_exp,10)
        yi=np.zeros(len(xi))
        import matplotlib.pyplot as plt
        for i in range(0,len(xi)):
            print "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx  preparing plot run",i,"of",len(xi),"using m_dot",xi[i],"which is",xi[i]/m_dot_exp,"times the experimental mass flowrate"
            yi[i]= objective([xi[i]])[0]
        plt.plot(xi,yi,marker='o')
        plt.show()
    x0=0.105
    x0=fsolve(objective, [x0])
    resids=objective(x0)
    
    if verbouseness>0: print "residuals for get_sim_flowrate_SH objective fct are: ",objective(x0)
    
    #exception handeling
    if resids[0]>0.5:
         x0=m_dot_exp*1.1
         x0=fsolve(objective, x0)
         resids=objective(x0)
         if verbouseness>0:print "residuals for get_sim_flowrate_SH objective fct are: ",objective(x0)
    if resids[0]>0.5:
        x0=m_dot_exp*0.9
        x0=fsolve(objective, x0)
        resids=objective(x0)
        if verbouseness>0: print "residuals for get_sim_flowrate_SH objective fct are: ",objective(x0)
    if resids[0]>0.5:
        x0=m_dot_exp*1.3
        x0=fsolve(objective, x0)
        resids=objective(x0)
        if verbouseness>0: print "residuals for get_sim_flowrate_SH objective fct are: ",objective(x0)
    return Evap
def verify_single_slab():
    #verify that calculation of first slab with new method is correct
    #see also D:\Purdue\DOE Project\Simulation\Verify_single_slab.xlsx
    
    #first calculate "normal" MCE
    Evap=Evaporator_DOE_HP(type='wCircuitry')  #multi circuit evaporator new version
    Evap.Circs_tubes_per_bank=np.array([4,6,4,6,4,4,6,4,6,4])
    Evap.Vdot_ha_coeffs= Evap.Circs_tubes_per_bank*1.0/(np.sum( Evap.Circs_tubes_per_bank))*1.0
    Evap.Fins.Air.FanPower=0.0  #only interested in refrigerant side
    kwargs={'Ref': 'R410A',
                'mdot_r':  0.106889403,
                'psat_r':  Props('P','T',C2K(0),'Q',1.0,'R410A'),
                'hin_r':Props('H','P',Props('P','T',C2K(0),'Q',1.0,'R410A'),'Q',0.15,'R410A')*1000
                }
    Evap.Update(**kwargs)
    Evap.Fins.Tubes.Nbank=1
    Evap.Calculate()
    DT_SH_target=5
    m_dot_target=0.09147390281390178 #DOE_HP_recircuited
    get_sim_flowrate_SH(Evap,DT_SH_target,m_dot_target,verbouseness=0)
    print Evap.OutputList()
    print "calculated mass flowrate",Evap.mdot_r,"calculated outlet superheat",Evap.Tout_r-Evap.Tdew_r
    Write2CSV(Evap,open('Verify_single_slab.csv','a'),append=False)
    
    #calculate recircuited slab
    Evap=DOE_HP_recircuit_single_slab(mode=0,plotit=False)  #first slab only
    Evap.Fins.Air.FanPower=0.0
    kwargs={'Ref': 'R410A',
                'mdot_r':  0.106889403,
                'psat_r':  Props('P','T',C2K(0),'Q',1.0,'R410A'),
                'hin_r':Props('H','P',Props('P','T',C2K(0),'Q',1.0,'R410A'),'Q',0.15,'R410A')*1000
                }
    Evap.Update(**kwargs)
    Evap.Calculate()
    DT_SH_target=5
    m_dot_target=0.09147390281390178 #DOE_HP_recircuited
    get_sim_flowrate_SH(Evap,DT_SH_target,m_dot_target,verbouseness=0)
    print Evap.OutputList()
    #need to multiply overall mass flowrate by 2, since assuming 2 slabs for recircuited coil
    #, therefore divided by 2 in MultiCircuitedEvaporator_w_circuitry
    print "calculated mass flowrate",Evap.mdot_r[0]*2,"calculated outlet superheat",Evap.Tout_r-Evap.Tdew_r
    Write2CSV(Evap,open('Verify_single_slab.csv','a'),append=False)
    
    
if __name__=='__main__':
    #if main is run, the parametric study for different airside maldistribution levels is run
    #to consider pressure drop, set sel.considerPD to true in MCE_W_Circuitry  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<!!
    
    if False: #plot circuitry
        DOE_HP_circuitry2()
    
    if False:  #set to True to run single slab verification
        print "calculating capacity for first slab only for verification"
        verify_single_slab()
        print"done with 'verify_single_slab():'"
    
    mode_maldistrib=[(0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2),(1,3)]  #different modes for maldistribution
    for i in range(0,len(mode_maldistrib)):
        Evap=DOE_HP_recircuit(mode=mode_maldistrib[i][0],plotit=False,maldistrib=mode_maldistrib[i][1])
        Evap.Fins.Air.FanPower=0.0
        kwargs={'Ref': 'R410A',
                    'mdot_r':  0.106889403,
                    'psat_r':  Props('P','T',C2K(0),'Q',1.0,'R410A'),
                    'hin_r':Props('H','P',Props('P','T',C2K(0),'Q',1.0,'R410A'),'Q',0.15,'R410A')*1000
                    }
        Evap.Update(**kwargs)
        Evap.Calculate()
        DT_SH_target=5
        m_dot_target=0.1 #DOE_HP_recircuited, equal flow, no pressure drop
        get_sim_flowrate_SH(Evap,DT_SH_target,m_dot_target,verbouseness=0,plotit=False)
        print Evap.OutputList()
        print "calculated mass flowrate",Evap.mdot_r,"calculated outlet superheat",Evap.Tout_r-Props('T','Q',1.0,'P',Evap.Pout_r,Evap.Ref)
        Write2CSV(Evap,open('maldistrib_study_noPD.csv','a'),append=False)
    
    #calculate the corresponding subcooling for nominal 43.3C condensing temperature
    h=Evap.hin_r
    T_sat_cond=C2K(43.3)
    P_sat_cond=Props('P','T',T_sat_cond,'Q',0.0,'R410A')
    T_subc=Props('T','H',h,'P',P_sat_cond,'R410A')
    DT_subc=T_sat_cond-T_subc
    print "used the following inlet conditions for the simulation h,DT_subc,K2C(T_subc) >>",h,DT_subc,K2C(T_subc)
