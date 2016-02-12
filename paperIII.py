import matplotlib
matplotlib.use('GTKAgg')
from mpl_toolkits.mplot3d import Axes3D

import sys,os

if sys.platform =='win32' or sys.platform =='win64':
    try:
        sys.path.append(os.environ['MYPYTHONHOME'])
    except:
        print 'MYPYTHONHOME does not exist'
    
sys.path.append('PythonModules')

import shutil,glob

from PyScroll.Plot import plotScrolls as ps
from  FileIO.prep_csv2rec import prep_csv2rec as prep
import numpy as np
from matplotlib.mlab import csv2rec
from scipy.optimize import fmin,fmin_l_bfgs_b
from scipy.interpolate import interp1d
import subprocess
from FloodedCycleModeling.LFEC import solveCycle
import CoolProp.FloodProp as FP
from math import pi

import pylab
params = {'axes.labelsize': 10,
          'axes.linewidth':0.5,
          'font.size':10,
          'text.fontsize': 10,
          'legend.fontsize': 8,
          'legend.labelspacing':0.2,
          'xtick.labelsize': 10,
          'ytick.labelsize': 10,
          'lines.linewidth': 0.5,
          'text.usetex': False,
          'font.family':'Times New Roman',
          'ps.fonttype':42}
pylab.rcParams.update(params)


def FreshenPythonFiles():
    if sys.platform == "win32" or sys.platform == "win64":
        # Make a local copy of the PythonModules folder
        try:
            #check that the 
            oldFolder=os.environ['MYPYTHONHOME']
            shutil.rmtree('PythonModules')
        except:
            pass
            
        try:
            shutil.copytree(os.environ['MYPYTHONHOME'],'PythonModules',ignore=shutil.ignore_patterns('Binaries'))
        except:
            print "Couldn't rebuild PythonModules"
    

def EricssonLosses():
    r=prep('Validation\\Results.csv')    
##     print r.dtype.names
    
    f=pylab.figure(figsize=(3.5,3.5))
    ax=f.add_axes((0.13,0.11,0.82,0.84))
    y=r['leakageflank']+r['leakageradial']+r['suction']+r['discharge']+r['mechanical']+r['w_adiabatic']+r['q_scroll_amb']+r['heattransfer']
    x=r['p_shaft']
    ax.plot(x,y,'bo',ms=4)
    ax.plot(np.r_[0,5],np.r_[0,5],'k')
    ax.plot(np.r_[0,5],np.r_[0,5*0.97],'k--',lw=0.5)
    ax.plot(np.r_[0,5],np.r_[0,5*1.03],'k--',lw=0.5)
    ax.text(3.8+0.05,0.97*3.8,'-3%',ha='left',va='top')
    ax.text(3.6-0.05,1.03*3.6,'+3%',ha='right',va='bottom')
    ax.set_xlim(2,5)
    ax.set_ylim(2,5)
    ax.set_xlabel('Experimental Shaft Power [kW]')
    ax.set_ylabel('Losses+Adiabatic Power [kW]')
    f.savefig('EricssonLosses.pdf')
    f.savefig('EricssonLosses.eps')
    f.savefig('EricssonLosses.png',dpi=600)
    pylab.close()
    
    ind = np.arange(27)    # the x locations for the groups
    width = 0.8      # the width of the bars: can also be len(x) sequence

    xL=r['xl']
    pratio=r['p_out']/r['p_in']
    flank=r['leakageflank']
    rad=r['leakageradial']
    suct=r['suction']
    disc=r['discharge']
    mech=r['mechanical']
    adia=r['w_adiabatic']
    
    f=pylab.figure(figsize=(3.5,3.5))
    ax=f.add_axes((0.1,0.12,0.85,0.83))
    
    tot=adia+flank+rad+suct+disc
    l6=pylab.bar(ind, mech,   width, color='k', bottom=tot,lw=0.4,label='Mechanical')
    tot=tot-disc
    l5=pylab.bar(ind, disc,   width, color='c', bottom=tot,lw=0.4,label='Discharge')
    tot=tot-suct
    l4=pylab.bar(ind, suct,   width, color='g', bottom=tot,lw=0.4,label='Suction')
    tot=tot-rad
    l3=pylab.bar(ind, rad,   width, color='yellow',bottom=tot,lw=0.4,label='Radial')
    tot=tot-flank
    l2=pylab.bar(ind, flank,   width, color='r', bottom=tot,lw=0.4,label='Flank')
    tot=tot-adia
    l1=pylab.bar(ind, adia,   width, color='b',bottom=tot,lw=0.4,label='Adiabatic')
    
    ax.set_xlabel('Run Number [-]')
    ax.set_ylabel('Power terms [kW]')
    ax.set_ylim(0,5)
    leg=ax.legend(loc='lower right')
    frame  = leg.get_frame()  
    frame.set_linewidth(0.5)
    f.savefig('ExperimentLosses.pdf')
    f.savefig('ExperimentLosses.eps')
    f.savefig('ExperimentLosses.png',dpi=600)
    pylab.close()
    
    f=open('EricssonTable.tex','w')
    head='%Made from paperIII.py function EricssonData()\n'
    head=head+r'\begin{table}[ht]'+'\n'
    head=head+r'\setlength{\tabcolsep}{3pt}'+'\n'
    head=head+r'\renewcommand*\arraystretch{1.0}'+'\n'
    head=head+r'\centering'+'\n'
    head=head+r'\caption{Irreversibility generation components for each experimental testing point from LFEC experimental testing}'+'\n'
    head=head+r'\label{tab:EricssonLosses}'+'\n'
    head=head+r'\begin{tabular}{*{%d}{c}}' % 9 +'\n'
    head=head+r'\hline\hline' +'\n'
    head=head+r'Run& $p_r$ & $x_l$ & $\dot W_{ad}$  & Rad. & Flank  & Suct. & Disc.& Mech.'+ r'\\'+'\n'
    head=head+r'\#& - & - & kW  & kW & kW  & kW & kW& kW'+ r'\\'+'\hline\n'
    f.write(head)
        
    
    for i in range(len(flank)):        
        flank=r['leakageflank'][i]
        rad=r['leakageradial'][i]
        suct=r['suction'][i]
        disc=r['discharge'][i]
        mech=r['mechanical'][i]
        adia=r['w_adiabatic'][i]
        xL=r['xl'][i]
        pratio=r['p_out'][i]/r['p_in'][i]
        data='%d & %0.2f & \multicolumn{1}{c|}{%0.2f} & %0.2f & %0.2f & %0.2f & %0.2f & %0.2f & %0.2f' %(i+1,pratio,xL,adia,rad,flank,suct,disc,mech)
        f.write(data+r'\\'+'\n')

    f.write(r'\hline\hline'+'\n')
    f.write(r'\end{tabular}'+'\n')
    f.write(r'\end{table}'+'\n')
    f.close()
    
    def ThreeD():
        N1=0
        N2=9
        xL=r['xl'][N1:N2]
        pratio=r['p_out'][N1:N2]/r['p_in'][N1:N2]
        flank=r['leakageflank'][N1:N2]
        rad=r['leakageradial'][N1:N2]
        suct=r['suction'][N1:N2]
        disc=r['discharge'][N1:N2]
        mech=r['mechanical'][N1:N2]
        adia=r['w_adiabatic'][N1:N2]
        f=pylab.figure(figsize=(3.5,3.5))
        ax=f.add_subplot(111, projection='3d')
        
        dx = 0.01 * np.ones_like(pratio)
        dy = 0.03 * np.ones_like(pratio)
        
        
        
        tot3=adia+flank+rad+suct+disc
    ##     print xL,pratio,tot3,dx,dy,mech
        l6=ax.bar3d(xL,pratio,tot3,dx,dy,mech, color='k',lw=0.4,label='Mechanical',alpha=1.0)
        tot3=tot3-disc
        l5=ax.bar3d(xL,pratio,tot3,dx,dy,disc, color='c', lw=0.4,label='Discharge',alpha=1.0)
        tot3=tot3-suct
        l5=ax.bar3d(xL,pratio,tot3,dx,dy,suct, color='yellow', lw=0.4,label='Suction',alpha=1.0)
        tot3=tot3-rad
        l5=ax.bar3d(xL,pratio,tot3,dx,dy,rad, color='g', lw=0.4,label='Radial',alpha=1.0)
        tot3=tot3-flank
        l5=ax.bar3d(xL,pratio,tot3,dx,dy,flank, color='r', lw=0.4,label='Flank',alpha=1.0)
        tot3=tot3-adia
        l5=ax.bar3d(xL,pratio,tot3,dx,dy,adia, color='b', lw=0.4,label='Adiabatic',alpha=1.0)
        
        ax.set_xlabel('Oil Mass Fraction [-]')
        ax.set_zlabel('Pressure Ratio [-]')
        ax.set_zlabel('Power terms [kW]')
    ##     ax.set_ylim(0,5)
        ax.legend(loc='lower right')
        f.savefig('ExperimentLosses3D.pdf')
        f.savefig('ExperimentLosses3D.eps')
        f.savefig('ExperimentLosses3D.png',dpi=600)
        pylab.close()

    
def ConfigA():
    r=prep('ConfigA\\Results.csv')
    
##     print r.dtype.names
    w=3.5
    h=3.5
    d=0.15
    f=pylab.figure(figsize=(3.5,3.5))
    ax=f.add_axes((0.1,0.1,0.8,0.8))
    data=np.r_[r['leakageflank'],r['leakageradial'],r['suction'],r['discharge'],r['mechanical']]
    labels=['Leakage\n(flank)','Leakage\n(radial)','Suction','Discharge','Mechanical']
    (path,labelText,pcts)=ax.pie(data,labels=labels,autopct='%0.1f%%', colors=['yellow','g','r','c','m'])
    [t.set_size(8) for t in labelText]
    [p.set_size(8) for p in pcts]
    [p.set_linewidth(0.5) for p in path]
    f.savefig('ConfigALossPie.pdf')
    f.savefig('ConfigALossPie.eps',set_bbox='tight')
    f.savefig('ConfigALossPie.png',dpi=600)
    pylab.close()
    
    path='ConfigA\\Run 0001'
    pi=np.pi
    axes_size=((0.15,0.18,0.8,0.77))
    theta=np.loadtxt(path+'/'+"theta.csv",delimiter=',')
    p=np.loadtxt(path+'/'+"p.csv",delimiter=',')
    p=np.where(p<1e-10,np.nan,p)
    # Plot data
    fig=pylab.figure(num=1,figsize=(3.5,2.5))
    pylab.axes(axes_size)
    ##pylab.subplot(131)
    pylab.plot(theta,p[1,:],label='$p_s$')
    pylab.plot(theta,p[9,:],label='$p_{c1-1}$')
    pylab.plot(theta,p[2,:],label='$p_{d1}$')
    pylab.plot(theta,p[6,:],label='$p_{ddd}$')
    pylab.plot(theta,p[5,:],label='$p_{dd}$')
    pylab.plot(theta,p[7,:],'k--',label='$p_{dd}$')
    pylab.plot(theta,p[8,:],'k--')
    pylab.text(3*pi/2.,500+10,'Suction',ha='center',va='bottom',size=8)
    pylab.text(pi/2.,1850-10,'Discharge',ha='center',va='top',size=8)
    pylab.text(7*pi/4.,440,'$s_1$',ha='center',va='top',size=10,color='b')
    pylab.text(pi,830,'$c_1$',ha='left',va='top',size=10,color='g')
    pylab.text(3*pi/2,1150,'$d_1$',ha='left',va='top',size=10,color='r')
    pylab.text(3*pi/2,1700,'$dd$',ha='center',va='top',size=10,color='m')
    pylab.plot(np.r_[1.36*pi,1.36*pi],np.r_[900,2200],'k--')
    pylab.text(1.36*pi-0.05,1450,'Discharge Angle',rotation=90,size=7,ha='right',va='center')
    pylab.plot(np.r_[1.72*pi,1.72*pi],np.r_[900,2200],'k--')
    pylab.text(1.72*pi+0.05,1000,'Merge',rotation=90,size=7,ha='left',va='center')
    pylab.text(3*pi/4,2050,'ddd',color='c',ha='center',va='center')
    
    pylab.xticks([0,pi/2,pi,3*pi/2,2*pi],('0','$\pi$/2','$\pi$','3$\pi$/2','2$\pi$'))
    pylab.xlim([0,2*pi])
    pylab.ylim([300,2200])
    pylab.xlabel(r"$\theta $ [radians]")
    pylab.ylabel(r'Pressure [kPa]')
    pylab.savefig('ConfigAPressure.pdf')
    pylab.savefig('ConfigAPressure.eps')
    pylab.savefig('ConfigAPressure.png',dpi=600)
    pylab.close()
    
    geo=ps.geoVals()
    ps.plotScrollSet(theta=3*pi/2.0,discOn=True,lw=1)
    pylab.savefig('ConfigAScrollSet.pdf')
    pylab.savefig('ConfigAScrollSet.eps')
    pylab.savefig('ConfigAScrollSet.png',dpi=600)
    pylab.close()
    
    
def ConfigB():
    
    f=pylab.figure(figsize=(4,2))
    ax=f.add_axes((0,0,0.5,1))
    geo=ps.geoVals()
    ps.plotScrollSet(7*3.141592654/4,discOn=True,lw=1,geo=geo,axis=ax)
    pylab.gca().set_xlim((-0.017,0.010))
    pylab.gca().set_ylim((-0.021,0.006))
    
    ax=f.add_axes((0.5,0,0.5,1))
    geo=ps.geoVals()
    ps.setDiscGeo(geo,'2Arc',0.001)
    geo.disc_x0=-0.00269898
    geo.disc_y0=-0.00348883
    geo.disc_R=0.01187952
    ps.plotScrollSet(7*3.141592654/4,discOn=True,lw=1,geo=geo,axis=ax)
    pylab.gca().set_xlim((-0.017,0.010))
    a=-0.004
    pylab.gca().set_ylim((-0.021-a,0.006-a))    
    
    pylab.savefig('ConfigBDischargeBlockage.pdf')
    pylab.savefig('ConfigBDischargeBlockage.eps')
    pylab.savefig('ConfigBDischargeBlockage.png',dpi=400)
    pylab.close()
        
    path='ConfigA\\Run 0001'
    A0=np.loadtxt(path+'/'+"Adisc.csv",delimiter=',')
    path='ConfigB\\Run 0001'
    A1=np.loadtxt(path+'/'+"Adisc.csv",delimiter=',')
    path='ConfigB\\Run 0005'
    A2=np.loadtxt(path+'/'+"Adisc.csv",delimiter=',')
    path='ConfigB\\Run 0009'
    A3=np.loadtxt(path+'/'+"Adisc.csv",delimiter=',')
    
    pi=np.pi
    axes_size=((0.18,0.20,0.77,0.74))
    
    fig=pylab.figure(figsize=(3.5,2.5))
    pylab.axes(axes_size)
    
    pylab.plot(A0[:,0],A0[:,1]*1000**2,label='$p_s$')
##     pylab.plot(A1[:,0],A1[:,1]*1000**2,label='$p_s$')
    pylab.plot(A2[:,0],A2[:,1]*1000**2,label='$p_s$')
##     pylab.plot(A3[:,0],A3[:,1]*1000**2,label='$p_s$')
    
    pylab.text(0.9*3.14,120,'Baseline',ha='center',va='bottom')
    pylab.text(0.8*3.14,372,'Larger port',ha='center',va='bottom',rotation=7)
    
    pylab.xticks([0,pi/2,pi,3*pi/2,2*pi],('0','$\pi$/2','$\pi$','3$\pi$/2','2$\pi$'))
    pylab.xlim([0,2*pi])
    pylab.ylim([0,500])
    pylab.xlabel(r"$\theta $ [radians]")
    pylab.ylabel(r'Discharge Port Area [mm$^2$]')
    pylab.savefig('ConfigBAdisc.pdf')
    pylab.savefig('ConfigBAdisc.eps')
    pylab.savefig('ConfigBAdisc.png',dpi=600)
    pylab.close()
    
    path='ConfigB\\Run 0005'
    pi=np.pi
    axes_size=((0.15,0.18,0.8,0.77))
    theta=np.loadtxt(path+'/'+"theta.csv",delimiter=',')
    p=np.loadtxt(path+'/'+"p.csv",delimiter=',')
    p=np.where(p<1e-10,np.nan,p)
    # Plot data
    fig=pylab.figure(num=1,figsize=(3.5,2.5))
    pylab.axes(axes_size)
    ##pylab.subplot(131)
    pylab.plot(theta,p[1,:],label='$p_s$')
    pylab.plot(theta,p[9,:],label='$p_{c1-1}$')
    pylab.plot(theta,p[2,:],label='$p_{d1}$')
    pylab.plot(theta,p[6,:],label='$p_{ddd}$')
    pylab.plot(theta,p[5,:],label='$p_{dd}$')
    pylab.plot(theta,p[7,:],'k--',label='$p_{dd}$')
    pylab.plot(theta,p[8,:],'k--')
    pylab.text(3*pi/2.,500+10,'Suction',ha='center',va='bottom',size=8)
    pylab.text(pi/2.,1850-10,'Discharge',ha='center',va='top',size=8)
    pylab.text(7*pi/4.,440,'$s_1$',ha='center',va='top',size=10,color='b')
    pylab.text(pi,830,'$c_1$',ha='left',va='top',size=10,color='g')
    pylab.text(3*pi/2,1150,'$d_1$',ha='left',va='top',size=10,color='r')
    pylab.text(3*pi/2+0.2,1800,'$dd$',ha='center',va='top',size=10,color='m')
    pylab.plot(np.r_[1.36*pi,1.36*pi],np.r_[900,2200],'k--')
    pylab.text(1.36*pi-0.05,1450,'Discharge Angle',rotation=90,size=7,ha='right',va='center')
    th=1.75*pi
    pylab.plot(np.r_[th,th],np.r_[900,2200],'k--')
    pylab.text(th+0.05,1000,'Merge',rotation=90,size=7,ha='left',va='center')
    pylab.text(3*pi/4,1950,'ddd',color='c',ha='center',va='center')
    pylab.xticks([0,pi/2,pi,3*pi/2,2*pi],('0','$\pi$/2','$\pi$','3$\pi$/2','2$\pi$'))
    pylab.ylim([300,2200])
    pylab.xlim([0,2*pi])
    pylab.xlabel(r"$\theta $ [radians]")
    pylab.ylabel(r'Pressure [kPa]')
    pylab.savefig('ConfigBPressure.pdf')
    pylab.savefig('ConfigBPressure.eps')
    pylab.savefig('ConfigBPressure.png',dpi=600)
    pylab.close()
def InlineLabel(xv,yv,x,**kwargs):
    def ToPixelCoords(xv,yv,axis,fig):
        [Axmin,Axmax]=axis.get_xlim()
        [Aymin,Aymax]=axis.get_ylim()
        DELTAX_axis=Axmax-Axmin
        DELTAY_axis=Aymax-Aymin
        
        width=fig.get_figwidth()
        height=fig.get_figheight()
        pos=axis.get_position().get_points()
        [[Fxmin,Fymin],[Fxmax,Fymax]]=pos
        DELTAX_fig=width*(Fxmax-Fxmin)
        DELTAY_fig=height*(Fymax-Fymin)
        
        #Convert coords to pixels
        x=(xv-Axmin)/DELTAX_axis*DELTAX_fig+Fxmin
        y=(yv-Aymin)/DELTAY_axis*DELTAY_fig+Fymin
        
        return x,y
    
    def ToDataCoords(xv,yv,axis,fig):
        [Axmin,Axmax]=axis.get_xlim()
        [Aymin,Aymax]=axis.get_ylim()
        DELTAX_axis=Axmax-Axmin
        DELTAY_axis=Aymax-Aymin
        
        width=fig.get_figwidth()
        height=fig.get_figheight()
        pos=axis.get_position().get_points()
        [[Fxmin,Fymin],[Fxmax,Fymax]]=pos
        DELTAX_fig=(Fxmax-Fxmin)*width
        DELTAY_fig=(Fymax-Fymin)*height
        
        #Convert back to measurements
        x=(xv-Fxmin)/DELTAX_fig*DELTAX_axis+Axmin
        y=(yv-Fymin)/DELTAY_fig*DELTAY_axis+Aymin
        
        return x,y
    
    axis=kwargs.get('axis',pylab.gca())
    fig=kwargs.get('figure',pylab.gcf())
    
    trash=0*x
    (xv,yv)=ToPixelCoords(xv,yv,axis,fig)
    (x,trash)=ToPixelCoords(x,trash,axis,fig)
    
    #Get the rotation angle
    f = interp1d(xv, yv)
    y = f(x)
    h = 0.001*x
    dy_dx = (f(x+h)-f(x-h))/(2*h)
    rot = np.arctan(dy_dx)/np.pi*180.
    
    (x,y)=ToDataCoords(x,y,axis,fig)
    return (x,y,rot)
    
dx=0.17
ConfigCAxes=(dx,0.15,0.95-dx,0.8)

def OptimalRb(Vratio,Vdisp,t,phi_os=0.3,F=3):
    def OBJECTIVE(rb):
        phi_i0=0.0
        phi_is=np.pi     
        delta=12e-6
        phi_o0=phi_i0-t/rb
        hs=(Vdisp)/(rb**2*2*pi*Vratio*(pi+phi_o0)*(2*phi_os+3*pi-phi_o0))
        phi_ie=Vdisp/(4*pi*hs*rb**2*(pi+phi_o0))+(3*pi+phi_o0)/2.0
        A_radial=2*delta*rb*((phi_ie-np.pi)**2/2-(phi_is+np.pi)**2/2-phi_i0*(phi_ie-phi_is-2*np.pi))*1e6
        Nfl=(phi_ie-phi_is)/(2*pi)
        A_flank=2*Nfl*delta*hs*1e6*F
        return A_flank+A_radial
    return fmin(OBJECTIVE,0.003,ftol=1e-12)
        
def FindRb():
    
    
    for Vratio in [1.8,2.1,2.4,2.7,3.0,3.38]:
        Vdisp=104.8e-6
        t=0.00466
        phi_i0=0.0
        phi_is=np.pi
        phi_os=0.3        
        delta=12e-6
        rb=np.linspace(2e-3,8e-3,100000)
        phi_o0=phi_i0-t/rb
        hs=(Vdisp)/(rb**2*2*pi*Vratio*(pi+phi_o0)*(2*phi_os+3*pi-phi_o0))
        phi_ie=Vdisp/(4*pi*hs*rb**2*(pi+phi_o0))+(3*pi+phi_o0)/2.0
        
        A_radial=2*delta*rb*((phi_ie-np.pi)**2/2-(phi_is+np.pi)**2/2-phi_i0*(phi_ie-phi_is-2*np.pi))*1e6
        Nfl=(phi_ie-phi_is)/(2*pi)
        A_flank=2*Nfl*delta*hs*1e6*3
        
        minI=np.argmin(A_flank+A_radial)
        print 'Vratio: %0.1f rb: %g' %(Vratio,rb[minI])
        
        if Vratio==2.7:
            f=pylab.figure(figsize=(3.5,2.5))
            ax=f.add_axes(ConfigCAxes)
            pylab.plot(rb*1000,A_radial)
            pylab.plot(rb*1000,A_flank)
            pylab.plot(rb*1000,A_flank+A_radial)
            pylab.xlabel(r"Base circle radius [mm]")
            pylab.ylabel(r'$A^*$ [mm$^2$]')
            (x,y,rot)=InlineLabel(rb*1000,A_radial,3.5)
            ax.text(x,y,'Radial',ha='center',va='center',rotation=rot,rotation_mode='anchor',bbox=dict(boxstyle='round',color='white',ec='none',alpha=0.9))
            (x,y,rot)=InlineLabel(rb*1000,A_flank,4)
            ax.text(x,y,'Flank',ha='center',va='center',rotation=rot,rotation_mode='anchor',bbox=dict(boxstyle='round',color='white',ec='none',alpha=0.9))
            (x,y,rot)=InlineLabel(rb*1000,A_flank+A_radial,4)
            ax.text(x,y+2.5,'Total',ha='center',va='center',rotation=rot,rotation_mode='anchor',bbox=dict(boxstyle='round',color='white',ec='none',alpha=0.9))
            
            pylab.savefig('EffLeakageArea.pdf')
            pylab.savefig('EffLeakageArea.eps')
            pylab.savefig('EffLeakageArea.png',dpi=600)
            pylab.close()
            
    fig=pylab.figure(figsize=(3.5,2.5))
    ax=fig.add_axes(ConfigCAxes)
    ax.set_xlim((1.5,3.3))
    ax.set_ylim((2.0,6))
                
    DispList=[10e-6,20e-6,40e-6,70e-6,104.8e-6]
    for Vdisp in DispList:  
        Vr=np.linspace(1.5,3.6)
        RB=np.zeros_like(Vr)
        for i in range(len(Vr)):
            Vratio=Vr[i]
            RB[i]=OptimalRb(Vratio,Vdisp,t)
        pylab.plot(Vr,RB*1000,lw=1.3)
        
        axis=pylab.gca()
        xlims=axis.get_xlim()
        ylims=axis.get_ylim()
        DELTAX=xlims[1]-xlims[0]
        DELTAY=ylims[1]-ylims[0]
        (x,y,rot)=InlineLabel(Vr,RB*1000,2.1)
#        pylab.plot(np.r_[x,x+0.4],np.r_[y,y+0.4*np.tan(rot*DELTAY/DELTAX/180.*np.pi)],'k')
#        if Vdisp==104.8e-6:
#            dy=1
#        else:
#            dy=0
        dy=-0.22
        pylab.text(x,y+dy,r'$V_{disp}$ = '+'%0.1f cm$^3$' %(Vdisp*1e6),ha='center',va='center',rotation=rot,size=8)#,bbox=dict(color='white',boxstyle='round',ec='none',alpha=0.9))
    pylab.plot(np.r_[1.8,2.1,2.4,2.7,3.0],np.r_[5.03,4.66,4.31,3.94,3.58],'o',mfc='none',mec='m')
##     pylab.plot(np.r_[3.3],np.r_[2.55],'^',mfc='none',mec='g')
    
    pylab.plot(2.4,5.5,'o',mec='m',mfc='none')
    pylab.text(2.5,5.5,'LFEC Compressor',ha='left',va='center')
    
    ax.set_xticks((1.5,1.8,2.1,2.4,2.7,3.0,3.3,3.6))
    ax.set_xlabel('$V_{ratio}$ [-]')
    ax.set_ylabel('$r_b$ [mm]')
    
    pylab.savefig('Optimalrb.pdf')
    pylab.savefig('Optimalrb.eps')
    pylab.savefig('Optimalrb.png',dpi=600)
    pylab.close()

def ConfigC():
    
    
##     T1=278.0
##     p1=500.0
##     p2=1850.0
##     pratio=p2/p1
##     xlv=np.linspace(0,1,5000)
##     kstar=np.zeros_like(xlv)
##     Vratio=np.zeros_like(xlv)
##     Vratiostar=np.zeros_like(xlv)
##     a=np.zeros_like(xlv)
##     for i in range(len(xlv)):
##         xl=xlv[i]
##         a[i]=1-FP.VoidFrac('N2','Zerol',T1,p1,xl)
##         kstar[i]=FP.kstar_m('N2','Zerol',T1,p1,xl)
##         Vratiostar[i]=pratio**(1.0/kstar[i])
##         Vratio[i]=Vratiostar[i]/(1-a[i]+a[i]*Vratiostar[i])
##     
##     def OBJECTIVE_xl(xl):
##         xl=xl[0]
##         a=1-FP.VoidFrac('N2','Zerol',T1,p1,xl)
##         kstar=FP.kstar_m('N2','Zerol',T1,p1,xl)
##         Vratiostar=pratio**(1.0/kstar)
##         Vratio=Vratiostar/(1-a+a*Vratiostar)
##         return 1/Vratio
##         
##     xl=fmin(OBJECTIVE_xl,0.7)
##     print xl
##     f=pylab.figure(figsize=(3.5,2.5))
##     ax=f.add_axes((0.15,0.15,0.70,0.80))
##     ax.plot(xlv,Vratiostar,'k')
##     ax.plot(xlv,Vratio,'b')
##     ax.plot(xl,1/OBJECTIVE_xl(xl),'bo',ms=3,mew=1,mec='b',mfc='none',lw=0.3)
##     ax.set_xlabel('$x_l$ [-]',size=10)
##     ax.set_ylabel('Volume Ratios [-]',size=10)
##     ax.set_ylim(2.4,4.2)
##     ax.set_yticks(np.linspace(2.4,4.2,10))
##     ax2=ax.twinx()
##     ax2.plot(xlv,kstar,'r')
##     ax2.set_ylabel('$k^{*}$ [-]',size=10,color='r') 
##     ax.text(0.7,2.7,'$k^{*}$',color='r')
##     ax.text(0.7,3.6,'$V_{ratio}^{*}$',color='k',rotation=21)
##     ax.text(0.8,3.25,'$V_{ratio}$',color='b',ha='center')
##     f.savefig('ConfigCVratioIdeal.pdf')
##     f.savefig('ConfigCVratioIdeal.eps')
##     f.savefig('ConfigCVratioIdeal.png',dpi=600)
##     pylab.close()
    
    def Vratio(Ref,Liq,T1,p1,xl,p2,eta=1.0):
        h1=FP.h_m(Ref,Liq,T1,p1,xl)
        s1=FP.s_m(Ref,Liq,T1,p1,xl)
        h2s=FP.h_sp(Ref,Liq,s1,p2,xl,T1)
        h2=(h2s-h1)/eta+h1
        T2=FP.T_hp(Ref,Liq,h2,p2,xl,T1)
        rho1=FP.rho_m(Ref,Liq,T1,p1,xl)
        rho2s=FP.rho_m(Ref,Liq,T2,p2,xl)
        Vratio=rho2s/rho1
        return Vratio
    
    T1=278
    p1=500
    p2=1850
    pratio=p2/p1
    Ref='Nitrogen'
    fig=pylab.figure(figsize=(3.5,2.5))
    ax=fig.add_axes(ConfigCAxes)
    ax.set_ylim((1.0,3.5))
    #First the ideal curve for N2 and Zerol
    XL=[]
    VRATIO=[]
    Liq='Zerol'
    for xl in np.linspace(0,1.0,100):
        XL.append(xl)
        VRATIO.append(Vratio(Ref,Liq,T1,p1,xl,p2,eta=1.0))
    pylab.plot(XL,VRATIO,'b',lw=1,label='Ideal')
    (x,y,rot)=InlineLabel(XL,VRATIO,0.2,axis=ax)
    pylab.text(x,y,'Ideal',ha='center',va='center',rotation=rot,rotation_mode='anchor',bbox=dict(boxstyle='round',color='white',ec='none',alpha=0.9))
    #Next the 60% curve for N2 and Zerol
    XL=[]
    VRATIO=[]
    Liq='Zerol'
    for xl in np.linspace(0,1.0,100):
        XL.append(xl)
        VRATIO.append(Vratio(Ref,Liq,T1,p1,xl,p2,eta=0.6))
    pylab.plot(XL,VRATIO,'b--',label='$\eta$=0.6')
    (x,y,rot)=InlineLabel(XL,VRATIO,0.2,axis=ax)
    pylab.text(x,y,'$\eta_a$ = 0.6',ha='center',va='center',rotation=rot,rotation_mode='anchor',bbox=dict(boxstyle='round',color='white',ec='none',alpha=0.9))
    print interp1d(XL,VRATIO)(0.8796)
    
    pylab.xlabel('$x_l$ [-]')
    pylab.ylabel('$V_{ratio}$ [-]')
    pylab.savefig('VratioN2Zerol.pdf')
    pylab.savefig('VratioN2Zerol.eps')
    pylab.savefig('VratioN2Zerol.png',dpi=600)
    
    pylab.close()
    
    r=prep('ConfigC\\Results.csv')
    
    f=pylab.figure(figsize=(3.5,2.5))
    ax=f.add_axes(ConfigCAxes)
    N=19
    VratioList=['1.8','2.1','2.4','2.7','3.0']
    mt=['-o','-s','-*','-^','-<']
    for i in range(4,-1,-1):
        rb=r['rb'][i*N:(i+1)*N]*1000
        eta_c=r['eta_c'][i*N:(i+1)*N]
        ax.plot(rb,eta_c,mt[i],ms=4,label='$V_{ratio}$ = '+VratioList[i],mec='none',mew=0.1)
    pylab.xlabel(r"Base circle radius [mm]")
    pylab.ylabel(r'Overall isentropic efficiency [-]')
    leg=pylab.legend(loc='lower center',numpoints=1)
    frame  = leg.get_frame()  
    frame.set_linewidth(0.5)
    pylab.savefig('ConfigCeta_c.pdf')
    pylab.savefig('ConfigCeta_c.eps')
    pylab.savefig('ConfigCeta_c.png',dpi=600)
    pylab.close()
    
    
##     rbv=np.linspace(0.0025,0.006,20)
##     Vratiov=np.linspace(1.8,3.0,20)
##     (RBV,VRATIOV)=np.meshgrid(rbv,Vratiov)
##     ETA=0.0*RBV
##     for i in range(len(rbv)):
##         for j in range(len(Vratiov)):
##             rb=rbv[i]
##             vratio=Vratiov[j]
##             eta_c=-4.584697022+1084.03786637*rb-321163.091505*rb**(2)-38337008.4761*rb**(3)+11489934645.8*rb**(4)+2.63651770137e+12*rb**(5)-3.85103217203e+14*rb**(6)-1.63350067385e+17*rb**(7)+3.68493663615e+19*rb**(8)-2.10073219716e+21*rb**(9)-0.211549615641*vratio+0.253758321903*vratio**(2)+0.132426017479*vratio**(3)+0.0342253390286*vratio**(4)-0.000798476328877*vratio**(5)-0.00613563590463*vratio**(6)-0.00348600045232*vratio**(7)-0.000729077772629*vratio**(8)+0.000600849416383*vratio**(9)+29.15885081*rb**(+0.2927852684)*vratio**(-1.115372635)+213.1143723*rb*vratio-32.78419709*rb*vratio**2-604.6535839*rb**2*vratio
##             ETA[i,j]=eta_c
##     fig=pylab.figure(figsize=(3.25,3.25))
##     CS=pylab.contour(RBV*1000,VRATIOV,ETA,[0.65,0.675,0.7,0.71,0.72],colors='k',lw=1.5)
##     pylab.clabel(CS,fontsize=10,inline=1,fmt='%1.3f %%')
##     pylab.close()
    
    f=pylab.figure(figsize=(3.5,2.5))
    ax=f.add_axes(ConfigCAxes)
    i=3
    rb=r['rb'][i*N:(i+1)*N]*1000
    rad=r['leakageradial'][i*N:(i+1)*N]
    flank=r['leakageflank'][i*N:(i+1)*N]
    ax.plot(rb,rad,'-o',ms=4,label='Radial',mec='none',mew=0.1)
    ax.plot(rb,flank,'-s',ms=4,label='Flank',mec='none',mew=0.1)
    ax.plot(rb,rad+flank,'k-',ms=4,label='Total',mec='none',mew=0.1,lw=1)
    pylab.text(4.8,0.22,'Flank',rotation=-15,ha='center',va='center')
    pylab.text(4.8,0.6,'Radial',rotation=27,ha='center',va='center')
    pylab.text(4.8,0.94,'Total',rotation=15,ha='center',va='center')
    pylab.xlabel(r"Base circle radius [mm]")
    pylab.ylabel(r'Leakage terms [kW]')
##     pylab.legend(loc='center right',numpoints=1)
    pylab.savefig('ConfigCleakage.pdf')
    pylab.savefig('ConfigCleakage.eps')
    pylab.savefig('ConfigCleakage.png',dpi=600)
    pylab.close()
    
    phi_i0=r['phi_i0'][i*N:(i+1)*N]
    phi_is=r['phi_is'][i*N:(i+1)*N]
    phi_ie=r['phi_ie'][i*N:(i+1)*N]
    phi_o0=r['phi_o0'][i*N:(i+1)*N]
    phi_os=r['phi_os'][i*N:(i+1)*N]
    phi_oe=r['phi_oe'][i*N:(i+1)*N]
    rb=r['rb'][i*N:(i+1)*N]
    h=r['hs'][i*N:(i+1)*N]
    disc_x0=r['x_port'][i*N:(i+1)*N]
    disc_y0=r['y_port'][i*N:(i+1)*N]
    disc_R=r['r_port'][i*N:(i+1)*N]
    radialleak=r['leakageradial'][i*N:(i+1)*N]
    
    delta=12e-6
    f=pylab.figure(figsize=(3.5,2.5))
    ax=f.add_axes(ConfigCAxes)
    A_radial=2*delta*rb*(phi_ie*phi_ie/2-phi_is*phi_is/2-phi_i0*(phi_ie-phi_is))*1e6
    Nfl=(phi_ie-phi_os-2*pi)/(2*pi)
    A_flank=2*Nfl*delta*h*1e6
    
    pylab.plot(rb*1000,A_radial)
    pylab.plot(rb*1000,A_flank)
    pylab.plot(rb*1000,A_flank+A_radial)
    pylab.xlabel(r"Base circle radius [mm]")
    pylab.ylabel(r'Leakage area [mm$^2$]')
    pylab.savefig('ConfigCleakArea.pdf')
    pylab.savefig('ConfigCleakArea.eps')
    pylab.savefig('ConfigCleakArea.png',dpi=600)
    pylab.close()
    
##     i=4
##     geo=ps.geoVals()
##     geo.rb=rb[i]
##     geo.h=h[i]
##     geo.phi_i0=phi_i0[i]
##     geo.phi_is=phi_is[i]
##     geo.phi_ie=phi_ie[i]
##     geo.phi_o0=phi_o0[i]
##     geo.phi_os=phi_os[i]
##     geo.phi_oe=phi_oe[i]
##     geo.disc_x0=disc_x0[i]
##     geo.disc_y0=disc_y0[i]
##     geo.disc_R=disc_R[i]
##     geo.ro=geo.rb*(pi-geo.phi_i0+geo.phi_o0)
##     ps.setDiscGeo(geo, '2Arc', r2=0.0003)
##     ps.plotScrollSet(pi,geo=geo,discOn=True,lw=1)
##     pylab.savefig('ConfigCScrollSet.pdf')
##     pylab.savefig('ConfigCScrollSet.eps')
##     pylab.savefig('ConfigCScrollSet.png',dpi=600)
##     pylab.close()
    
def ConfigCAngles():        
    pylab.figure(figsize=(3.5,3.5))
    pylab.axes((0.0,0.0,1.0,1.0))
    geo=ps.LoadGeo()

    phi_i0=geo.phi_i0;
    phi_is=geo.phi_is;
    phi_ie=geo.phi_ie;
    
    phi_o0=geo.phi_o0;
    phi_os=geo.phi_os;
    phi_oe=geo.phi_oe;
    
    phi1=np.linspace(phi_is,phi_ie,1000)
    phi2=np.linspace(phi_i0,phi_is,1000)
    phi3=np.linspace(phi_os,phi_oe,1000)
    phi4=np.linspace(phi_o0,phi_os,1000)
    (xi1,yi1)=ps.coords_inv(phi1,geo,0,'fi')
    (xi2,yi2)=ps.coords_inv(phi2,geo,0,'fi')
    (xo1,yo1)=ps.coords_inv(phi3,geo,0,'fo')
    (xo2,yo2)=ps.coords_inv(phi4,geo,0,'fo')
    
    #### Inner and outer involutes
    pylab.plot(xi1,yi1,'k',lw=1)
    pylab.plot(xi2,yi2,'k:',lw=1)
    pylab.plot(xo1,yo1,'k',lw=1)
    pylab.plot(xo2,yo2,'k:',lw=1)
    
    ### Innver involute labels
    pylab.plot(xi2[0],yi2[0],'k.',markersize=5,mew=2)
    pylab.text(xi2[0],yi2[0]+0.0025,'$\phi_{i0}$',size=8,ha='right',va='bottom')
    pylab.plot(xi1[0],yi1[0],'k.',markersize=5,mew=2)
    pylab.text(xi1[0]+0.002,yi1[0],'$\phi_{is}$',size=8)
    pylab.plot(xi1[-1],yi1[-1],'k.',markersize=5,mew=2)
    pylab.text(xi1[-1]-0.002,yi1[-1],'   $\phi_{ie}$',size=8,ha='right',va='center')
    
    ### Outer involute labels
    pylab.plot(xo2[0],yo2[0],'k.',markersize=5,mew=2)
    pylab.text(xo2[0]+0.002,yo2[0],'$\phi_{o0}$',size=8,ha='left',va='top')
    pylab.plot(xo1[0],yo1[0],'k.',markersize=5,mew=2)
    pylab.text(xo1[0]+0.002,yo1[0],'$\phi_{os}$',size=8)
    pylab.plot(xo1[-1],yo1[-1],'k.',markersize=5,mew=2)
    pylab.text(xo1[-1]-0.002,yo1[-1],'   $\phi_{oe}$',size=8,ha='right',va='center')
    
    ### Base circle
    t=np.linspace(0,2*pi,100)
    pylab.plot(geo.rb*np.cos(t),geo.rb*np.sin(t),'b-')
    pylab.plot(np.r_[0,geo.rb*np.cos(9*pi/8)],np.r_[0,geo.rb*np.sin(9*pi/8)],'k-')
    pylab.text(geo.rb*np.cos(9*pi/8)+0.0005,geo.rb*np.sin(9*pi/8)+0.001,'$r_b$',size=8,ha='right',va='top')
    
    pylab.axis('equal')
    pylab.setp(pylab.gca(),'ylim',(min(yo1)-0.005,max(yo1)+0.005) )
    pylab.axis('off')

    pylab.savefig('FixedScrollAngles.png',dpi=600)
    pylab.savefig('FixedScrollAngles.eps')
    pylab.savefig('FixedScrollAngles.pdf')
    pylab.close()
    
## def ConfigCdry():
##     r=prep('ConfigCdry Joined3\\Results.csv')
##     
##     f=pylab.figure(figsize=(3.5,2.5))
##     ax=f.add_axes((0.18,0.15,0.77,0.8))
##     
##     N=9
##     VratioList=['1.6','1.85','2.1','2.35','2.6']
##     mt=['-o','-s','-*','-^','-<']
##     print len(r),len(r['rb'])
##     for i in range(4,-1,-1):
##         rb=r['rb'][i*N:(i+1)*N]*1000
##         print rb
##         eta_c=r['eta_c'][i*N:(i+1)*N]
##         eta_c[eta_c<0.1]=np.nan
##         ax.plot(rb,eta_c,mt[i],ms=4,label='$V_{ratio}$ = '+VratioList[i],mec='none',mew=0.1)
##     
##     pylab.xlabel(r"Base circle radius [mm]")
##     pylab.ylabel(r'Overall isentropic efficiency [-]')
##     leg=pylab.legend(loc='lower center',numpoints=1)
##     frame  = leg.get_frame()  
##     frame.set_linewidth(0.5)
##     pylab.savefig('ConfigCdryeta_c.pdf')
##     pylab.savefig('ConfigCdryeta_c.eps')
##     pylab.savefig('ConfigCdryeta_c.png',dpi=600)
##     os.startfile('ConfigCdryeta_c.pdf')
##     pylab.close()
##     
    
def ConfigD():
    r=prep('ConfigD\\Results.csv')
    
    params = {'ps.fonttype':3}
    pylab.rcParams.update(params)

    f=pylab.figure(figsize=(3.5,2.5))
    ax=f.add_axes((0.18,0.15,0.75,0.80))
    for i in [4]:
        ax.plot(r['delta_radial'][i*7:(i+1)*7],r['eta_c'][i*7:(i+1)*7])
    ax.set_xlabel(r'$\delta_{radial}$, $\delta_{flank}$ [$\mu$m]',size=10)
    ax.set_ylabel('Overall Isentropic Efficiency [-]',size=10)
    ax.set_ylim(0.7,0.9)
    f.savefig('ConfigDetac.pdf')
    f.savefig('ConfigDetac.eps')
    f.savefig('ConfigDetac.png',dpi=600)
    
    pylab.close()
def ConfigE():
    Tc=233
    Th=298
    eta=0.8
        
    reCalc=False
    
    N=50
    prv=np.linspace(1.5,4.5,N)
    psv=np.linspace(300,700,N)
    (PS,PR)=np.meshgrid(psv,prv)
    ETAII=np.zeros_like(PS)
    CAP=np.zeros_like(PS)
    ETAC=np.zeros_like(PS)
    XL=np.zeros_like(PS)
    for TL in [223.0,275.0]:
        if reCalc==True:
            for i in range(len(prv)):
                pr=prv[i]
                for j in range(len(psv)):
                    ps=psv[j]
                    eta=0.70
                    def OBJECTIVE_Cratio(x):
                        Run=solveCycle('N2','Zerol','Zerol',ps,x[0],x[1],TL,298,pr,0.9,eta,eta,eta,eta,0.0,1.0,1.0,use_etac_fit=True)
                        return 1/Run.Cycle.etaII 
                        
                    x0=[10.0,6.0]
                    b=[(1.0,15.0),(1.0,10.0)]
                    (x,f,d)=fmin_l_bfgs_b(OBJECTIVE_Cratio,np.array(x0),bounds=b,approx_grad=True)
                    Run=solveCycle('N2','Zerol','Zerol',ps,x[0],x[1],TL,298,pr,0.9,eta,eta,eta,eta,0.0,1.0,1.0,use_etac_fit=True)
                    print x,Run.Comp.xL,Run.Exp.xL
                    print ps,pr,Run.Cycle.etaII*100,Run.ColdHX.Q,Run.Comp.eta_c,Run.Comp.T_in
                    PS[i,j]=ps
                    PR[i,j]=pr
                    ETAII[i,j]=Run.Cycle.etaII
                    CAP[i,j]=Run.ColdHX.Q
                    ETAC[i,j]=Run.Comp.eta_c
                    XL[i,j]=Run.Comp.xL
            foldPath=os.path.join('ConfigE Joined','%0.0f'%(TL))
            np.savetxt(os.path.join(foldPath,'PS.csv'),PS)
            np.savetxt(os.path.join(foldPath,'PR.csv'),PR)
            np.savetxt(os.path.join(foldPath,'ETAII.csv'),ETAII)
            np.savetxt(os.path.join(foldPath,'CAP.csv'),CAP)
            np.savetxt(os.path.join(foldPath,'ETAC.csv'),ETAC)
            np.savetxt(os.path.join(foldPath,'XL.csv'),XL)
        else:
            foldPath=os.path.join('ConfigE Joined','%0.0f'%(TL))
            PS=np.loadtxt(os.path.join(foldPath,'PS.csv'))
            PR=np.loadtxt(os.path.join(foldPath,'PR.csv'))
            ETAII=np.loadtxt(os.path.join(foldPath,'ETAII.csv'))
            CAP=np.loadtxt(os.path.join(foldPath,'CAP.csv'))
            ETAC=np.loadtxt(os.path.join(foldPath,'ETAC.csv'))
            XL=np.loadtxt(os.path.join(foldPath,'XL.csv'))
        
        levelsCoarse=[]
        levelsFine=[]
        if TL==275:
            levelsCoarse=[5,6,7]
            levelsFine=[5.5,6.5,7.5]
        elif TL==223:
            levelsCoarse=[12,15,18]
            levelsFine=[13.5,16.5]
        fig=pylab.figure(figsize=(3.5,2.5))
        ax=fig.add_axes((0.15,0.15,0.8,0.8))
        CS=ax.contourf(PS,PR,CAP,cmap=pylab.get_cmap('Blues'))
        cb=fig.colorbar(CS)
        cb.set_label('Capacity [kW]')
        CS=ax.contour(PS,PR,ETAII*100,colors='k',levels=levelsCoarse,linewidths=0.8)
        ax.clabel(CS,fontsize=8,inline=1,fmt='%1.1f %%',inline_spacing=1)
        CS=ax.contour(PS,PR,ETAII*100,colors='k',levels=levelsFine,linewidths=0.2,linestyles='dashed')
        ax.clabel(CS,fontsize=8,inline=1,fmt='%1.1f %%',inline_spacing=1,manual=True)
        ax.set_xlabel('$p_s$ [kPa]')
        ax.set_ylabel('$p_{ratio}$ [-]')
        fig.savefig('ConfigE'+'%0.0f'%(TL)+'.pdf')
        fig.savefig('ConfigE'+'%0.0f'%(TL)+'.eps')
        fig.savefig('ConfigE'+'%0.0f'%(TL)+'.png',dpi=400)
        pylab.close()
        
##         fig=pylab.figure(figsize=(3.5,2.5))
##         ax=fig.add_axes((0.15,0.15,0.8,0.8))
##         CS=ax.contour(PS,PR,ETAC*100,colors='k')
##         ax.clabel(CS,fontsize=10,inline=1,fmt='%1.1f %%')
##         ax.set_xlabel('$p_s$ [kPa]')
##         ax.set_ylabel('$p_{ratio}$ [-]')
##         fig.savefig('ConfigEetac'+'%0.0f'%(TL)+'.pdf')
##         fig.savefig('ConfigEetac'+'%0.0f'%(TL)+'.eps')
##         fig.savefig('ConfigEetac'+'%0.0f'%(TL)+'.png',dpi=400)
##         pylab.close()
        
        fig=pylab.figure(figsize=(3.5,2.5))
        ax=fig.add_axes((0.15,0.15,0.8,0.8))
        CS=ax.contour(PS,PR,XL*100,colors='k',levels=[65,70,75,80],linewidths=1)
        ax.clabel(CS,fontsize=8,inline=1,fmt='%1.1f %%')
        CS=ax.contour(PS,PR,XL*100,colors='k',levels=[67.5,72.5,77.5],linewidths=0.2,linestyles='dashed')
        ax.clabel(CS,fontsize=8,inline=1,fmt='%1.1f %%')
        ax.set_xlabel('$p_s$ [kPa]')
        ax.set_ylabel('$p_{ratio}$ [-]')
        fig.savefig('ConfigExLopt'+'%0.0f'%(TL)+'.pdf')
        fig.savefig('ConfigExLopt'+'%0.0f'%(TL)+'.eps')
        fig.savefig('ConfigExLopt'+'%0.0f'%(TL)+'.png',dpi=400)
        pylab.close()
        
    
    
def CopyFiles():
    if sys.platform.startswith('linux'):
        subprocess.call(['sh','cropConfigALossPie.sh'])
    else:
        subprocess.call(['cropConfigALossPie.bat'])
    shutil.copy('EricssonLosses.eps',os.path.join('..','TeX','fig01.eps'))
    shutil.copy('ConfigAPressure.eps',os.path.join('..','TeX','fig02.eps'))
    shutil.copy('ConfigALossPie.eps',os.path.join('..','TeX','fig03.eps'))
    shutil.copy('ConfigBDischargeBlockage.eps',os.path.join('..','TeX','fig04.eps'))
    shutil.copy('ConfigBAdisc.eps',os.path.join('..','TeX','fig05.eps'))
    shutil.copy('ConfigBPressure.eps',os.path.join('..','TeX','fig06.eps'))
    shutil.copy('VratioN2Zerol.eps',os.path.join('..','TeX','fig07.eps'))
    shutil.copy('FixedScrollAngles.eps',os.path.join('..','TeX','fig08.eps'))
    shutil.copy('ScrollFamily.eps',os.path.join('..','TeX','fig09.eps'))
    shutil.copy('EffLeakageArea.eps',os.path.join('..','TeX','fig10.eps'))
    shutil.copy('Optimalrb.eps',os.path.join('..','TeX','fig11.eps'))
    shutil.copy('ConfigCeta_c.eps',os.path.join('..','TeX','fig12.eps'))
    shutil.copy('ConfigCleakage.eps',os.path.join('..','TeX','fig13.eps'))
    shutil.copy('ConfigDetac.eps',os.path.join('..','TeX','fig14.eps'))
    shutil.copy('EricssonConfig.eps',os.path.join('..','TeX','fig15.eps'))
    shutil.copy('ConfigE275.eps',os.path.join('..','TeX','fig16.eps'))
    shutil.copy('ConfigExLopt275.eps',os.path.join('..','TeX','fig17.eps'))
    shutil.copy('ConfigE223.eps',os.path.join('..','TeX','fig18.eps'))
    
    shutil.copy('EricssonTable.tex',os.path.join('..','TeX','EricssonLossesTable.tex'))

def BuildPDF():
    subprocess.call(['python','makeVersions.py','2Col'],cwd=os.path.join('..','TeX'))

def DisplayPDF():
    os.startfile(os.path.join('..','TeX','PaperIII_2Col.pdf'))
    
if __name__=='__main__':
##     FreshenPythonFiles()
##     EricssonLosses()
##     ConfigA()
##     ConfigB()
##     ConfigC()
##     ConfigCAngles()
##     FindRb()
    ###########################ConfigCdry()
##     ConfigD()
    ####ConfigE()
    CopyFiles()
    BuildPDF()
    ######################DisplayPDF()