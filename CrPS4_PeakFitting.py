# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 16:18:20 2022

@author: brian
"""
import os
import numpy as np
import pandas as pd
import fnmatch
import matplotlib.pyplot as plt
import lmfit
from lmfit.models import GaussianModel, VoigtModel, LinearModel, ConstantModel



def _1Voigt(x, ampG1, cenG1, sigmaG1, ampL1, cenL1, widL1):
    return (ampG1*(1/(sigmaG1*(np.sqrt(2*np.pi))))*(np.exp(-((x-cenG1)**2)/((2*sigmaG1)**2)))) +\
              ((ampL1*widL1**2/((x-cenL1)**2+widL1**2)) )


def my_baseline(m1x,m1y,m2x,m2y,x):
    #y = mx + b
    m = (m2y-m1y)/(m2x-m1x)
    b = -m*m1x + m1y
    y = m*x + b
    
    return y


file = r'C:\Users\Student\OneDrive - University of Tennessee\Desktop\Research\MPS (M = Cr, Mn)\CrPS4\2.3.2022\python'
os.chdir(file)
cwd = os.getcwd()
print(cwd)
list = os.listdir(file)
print(list)

Temperature = np.array([11,15,20,25,30,32,35,38,40,45,50,60,80,100,150,200,250,300])
Temperature_params = np.zeros((18,18))

i = 0



for k in range(len(list)):
    if fnmatch.fnmatch(list[k],'*fit_result*'):
        pass
    elif fnmatch.fnmatch(list[k],'*baseline_corrected*'):
        pass
    elif fnmatch.fnmatch(list[k],'*PeakFit - *'):
        pd_data = pd.read_csv(list[k])
        np_data = pd_data.to_numpy()        
        b1y = min(np_data[10:26,1])
        for kk in range(16):
            if np_data[kk+10,1] == b1y:
                b1x = np_data[kk+10,0]
                print(k)
                kkk1 = kk
        
        
        b2y = min(np_data[73:94,1])
        for kk in range(21):
            if np_data[kk+73,1] == b2y:
                b2x = np_data[kk+73,0]
                print(k)
                kkk2 = kk
        
        
        baseline_data = np_data[kkk1+10:kkk2+73,0]
        baseline_data = np.column_stack((baseline_data,my_baseline(b1x,b1y,b2x,b2y,np_data[kkk1+10:kkk2+73,0])))
        pd_baseline_data = pd.DataFrame(baseline_data,columns=['Absorption','Energy (eV)'])
        
        corrected = np_data[kkk1+10:kkk2+73,1] - baseline_data[:,1]
        
        fig = plt.figure()
        fig.suptitle(list[k])
        
        ax1 = fig.add_subplot(2,2,1)
        ax1.plot(np_data[:,0],np_data[:,1])
        ax1.plot(baseline_data[:,0],baseline_data[:,1])
        ax1.set_title('Before baseline correction')
        ax1.set_xlabel('Energy (eV)')
        ax1.set_ylabel('Absorption')
        
        ax2 = fig.add_subplot(2,2,2)
        ax2.plot(np_data[kkk1+10:kkk2+73,0],corrected)
        ax2.set_title('Baseline corrected')
        ax2.set_xlabel('Energy (eV)')
        ax2.set_ylabel('Absorption')
        
        model = VoigtModel() + ConstantModel()
        
        x_array = np_data[kkk1+10:kkk2+73,0]  
        
        pd_corrected = pd.DataFrame(corrected,columns = ['Absorption'])
        pd_x_array = pd.DataFrame(x_array,columns = ['Energy (eV)'])
        
        
        corrected_data_x_array = np.column_stack((x_array,corrected))
        
        #pd_corrected.to_csv('Baseline_corrected' + list[k],index=False)
        #pd_x_array.to_csv('Energies for baseline_corrected' + list[k],index=False)
        
        #pars = model.guess(pd_corrected, x=pd_x_array)
        
        
        params = model.make_params(amplitude=200, center=1.7, \
                           sigma=1, gamma=.2, c= 0)
        
        # do the fit, print out report with results 
        result = model.fit(corrected, params,x=x_array)
        print(result.fit_report(min_correl=0.25))
        
        #name_result = "PeakFitting Results - " + list[k]
        
        #with open('fit_result.txt' + list[k], 'w') as fh:
        #    fh.write(result.fit_report())
        
        

        """
        ampG1 = 20
        cenG1 = 17
        sigmaG1 = 1
        ampL1 = 10
        cenL1 = 5
        widL1 = 10
        
        x_array = np_data[kkk1+10:kkk2+73,0]    
        
        p0 = [np_data[kkk1+10:kkk2+73,0],ampG1,cenG1,sigmaG1,ampL1,cenL1,widL1]

        
        popt_1voigt, pcov_1voigt = scipy.optimize.curve_fit(_1Voigt, x_array, corrected,\
                    p0=[ampG1, cenG1, sigmaG1,ampL1, cenL1, widL1],maxfev=2000)
        
        voigt_peak_1 = _1Voigt(x_array, *popt_1voigt)
        perr_1voigt = np.sqrt(np.diag(pcov_1voigt))
        pars_1 = popt_1voigt
        """
        print(list[k])
        Temperature_params[i][0] = result.params['amplitude']
        Temperature_params[i][1] = result.params['sigma']
        Temperature_params[i][2] = result.params['c']
        Temperature_params[i][3] = result.params['gamma']
        Temperature_params[i][4] = result.params['fwhm']
        Temperature_params[i][5] = result.params['height']
        Temperature_params[i][6] = result.params['center']
        
        
        
        #Temperature = np.append(Temperature,Temperature_amplitude)
        #np.savetxt('PeakFit_data' + list[k],result.data)
        corrected_data_x_array = np.column_stack((corrected_data_x_array,result.best_fit))
        pd_corrected_data_x_array = pd.DataFrame(corrected_data_x_array,columns = ['Energy (eV)','Corrected_Absorption','Fit_data'])
        #pd_corrected_data_x_array.to_csv('Fit_data and Data' + list[k],index=False)


        
        i += 1
        
        ax2.plot(x_array,result.best_fit,'r--')
        ax2.fill_between(x_array, result.best_fit.min(), result.best_fit, facecolor="green", alpha=0.5)

        
        
        """
        WithBase = baseline_data[:,1] + 


        ax3 - fig.add_subplot(2,2,3)
        ax3.plot()

        """
        
        
        plt.show()

#np.savetxt('params.txt',Temperature_params)

