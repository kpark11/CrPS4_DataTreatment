# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 16:00:18 2022

@author: Student
"""

import os
import numpy as np
import pandas as pd
import fnmatch
import matplotlib.pyplot as plt
import lmfit
from lmfit.models import GaussianModel, VoigtModel, LinearModel, ConstantModel
import re




def _1Voigt(x, ampG1, cenG1, sigmaG1, ampL1, cenL1, widL1):
    return (ampG1*(1/(sigmaG1*(np.sqrt(2*np.pi))))*(np.exp(-((x-cenG1)**2)/((2*sigmaG1)**2)))) +\
              ((ampL1*widL1**2/((x-cenL1)**2+widL1**2)) )


def my_baseline(m1x,m1y,m2x,m2y,x):
    #y = mx + b
    m = (m2y-m1y)/(m2x-m1x)
    b = -m*m1x + m1y
    y = m*x + b
    
    return y


file = r'C:\Users\brian\OneDrive - University of Tennessee\Desktop\Research\MPS (M = Cr, Mn)\MnPS3\Amal\1_28_2016_Amal_variable_temp_MnPS3\python'
os.chdir(file)
cwd = os.getcwd()
print(cwd)
list = os.listdir(file)
print(list)

Temperature = np.array([5,10,20,40,50,60,65,70,74,80,85,90,100,110,120,135,140,160,170,180,200,210,220,230,250,370,300])
Temperature_params = np.zeros((len(Temperature),len(Temperature)))

i = 0



for k in range(len(list)):
    if fnmatch.fnmatch(list[k],'*fit_result*'):
        pass
    if fnmatch.fnmatch(list[k],'python*'):
        filename = list[k]        
        pd_data = pd.read_csv(filename)
        np_data = pd_data.to_numpy()        
        b1y = np_data[68,5]
        b1x = np_data[68,2]
        
        b2y = np_data[164,5]
        b2x = np_data[164,2]

        
        
        corrected_data = np_data[68:164,2]
        corrected_data = np.column_stack((corrected_data,np_data[68:164,5]))
        corrected_data = np.column_stack((corrected_data,my_baseline(b1x,b1y,b2x,b2y,np_data[68:164,2])))
        corrected_data = np.column_stack((corrected_data,corrected_data[:,1] - corrected_data[:,2]))
        
        fig = plt.figure()
        fig.suptitle(list[k])
        
        ax1 = fig.add_subplot(2,2,1)
        ax1.plot(corrected_data[:,0],corrected_data[:,1])
        ax1.set_title('Before baseline correction')
        ax1.set_xlabel('Energy (eV)')
        ax1.set_ylabel('Absorption')
        
        ax2 = fig.add_subplot(2,2,2)
        ax2.plot(corrected_data[:,0],corrected_data[:,3])
        ax2.set_title('Baseline corrected')
        ax2.set_xlabel('Energy (eV)')
        ax2.set_ylabel('Absorption')
        
        model = VoigtModel() + ConstantModel()
        
        
        
        params = model.make_params(amplitude=200, center=1.9, \
                           sigma=1, gamma=.2, c= 0)
        
        # do the fit, print out report with results 
        result = model.fit(corrected_data[:,3], params,x=corrected_data[:,0])
        print(result.fit_report(min_correl=0.25))
        
        #name_result = "PeakFitting Results - " + list[k]
        
        #with open('fit_result.txt' + list[k], 'w') as fh:
        #    fh.write(result.fit_report(sort_pars=True))
        

        print(list[k])
        
        
        Temperature_params[i][1] = result.params['amplitude']
        Temperature_params[i][2] = result.params['sigma']
        Temperature_params[i][3] = result.params['c']
        Temperature_params[i][4] = result.params['gamma']
        Temperature_params[i][5] = result.params['fwhm']
        Temperature_params[i][6] = result.params['height']
        Temperature_params[i][7] = result.params['center']
        
        regex = re.compile(r'\d+')
        temp_name = regex.findall(filename)
        for item in temp_name:
            float(item)
        
        Temperature_params[i][0] = item
        
        integration = np.sum(corrected_data[:,3])
        Temperature_params[i][8] = integration
        
        #Temperature = np.append(Temperature,Temperature_amplitude)
        #np.savetxt('PeakFit_data' + list[k],result.data)
        corrected_data = np.column_stack((corrected_data,result.best_fit))
        pd_corrected_data = pd.DataFrame(corrected_data,columns = ['Energy (eV)','Absorption (cm-1)','Linear Baseline','Corrected_Absorption (cm-1)','best_fit data'])
        #pd_corrected_data.to_csv('Fit_data and Data' + list[k],index=False)
        

        
        i += 1
        
        ax2.plot(corrected_data[:,0],result.best_fit,'r--')
        ax2.fill_between(corrected_data[:,0], result.best_fit.min(), result.best_fit, facecolor="green", alpha=0.5)

        
        plt.show()
        
fig_temp = plt.figure()
ax3 = fig_temp.add_subplot(1,1,1)
ax3.scatter(Temperature_params[:,0],Temperature_params[:,1],s=200)
ax3.set_title('MnPS3')
ax3.set_xlabel('Temperature (K)')
ax3.set_ylabel('Amplitude')

fig_temp1 = plt.figure()
ax4 = fig_temp1.add_subplot(1,1,1)
ax4.scatter(Temperature_params[:,0],Temperature_params[:,2],s=200)
ax4.set_title('MnPS3')
ax4.set_xlabel('Temperature (K)')
ax4.set_ylabel('Center (cm-1)')


fig_temp1 = plt.figure()
ax5 = fig_temp1.add_subplot(1,1,1)
ax5.scatter(Temperature_params[:,0],Temperature_params[:,8],s=200)
ax5.set_title('MnPS3')
ax5.set_xlabel('Temperature (K)')
ax5.set_ylabel('Oscillator strength')
#np.savetxt('params.txt',Temperature_params)

