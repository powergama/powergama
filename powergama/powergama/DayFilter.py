# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 13:51:57 2020

@author: tilv
"""


from __future__ import division
import time
import matplotlib.pyplot as plt
import numpy as np
import powergama.PrintFunctions as prnt
import pandas as pd
from matplotlib.ticker import AutoMinorLocator




###############################################################################



prnt.Print('Day Filtering Module Loaded',1)








def DayFilter(HourlyTimeSeries):

    
    NumberOfSeries = HourlyTimeSeries.shape[1]
    NumberOfHours = HourlyTimeSeries.shape[0]
    HoursPerDay = 24
    NumberOfDays = NumberOfHours/HoursPerDay
    

    if float(NumberOfDays).is_integer():
        NumberOfDays = int(NumberOfDays)
    else:
        print('Problem')
        raise

    Matrix = np.zeros(shape=(NumberOfDays,HoursPerDay,NumberOfSeries))

    
    for counter_hours in range(NumberOfHours):
        counter_hoursinday = int(counter_hours % HoursPerDay)
        counter_days = int((counter_hours - counter_hoursinday) / HoursPerDay)
        
        Matrix[counter_days,counter_hoursinday,:]   = HourlyTimeSeries.iloc[counter_hours,:]    
    
    
    ADHTS = pd.DataFrame(Matrix.mean(axis=0))

    ADHTS.columns = HourlyTimeSeries.columns

    
    return ADHTS





def DayPlot(ADHTS,fname=None,ylabel='',title=None):
 
    plt.rcParams.update({'font.size': 22})

    HoursPerDay = 24

    list_legend = list(ADHTS.columns)
    list_legend.insert(0,'_')

    ADHTS = ADHTS.append(pd.DataFrame(ADHTS.iloc[-1,:]).T, ignore_index=True)

    flatline = pd.DataFrame([1 for i in range(HoursPerDay+1)])

    fig4 = plt.figure(figsize=(16 , 9))
    ax4 = fig4.add_subplot(111)
    flatline.plot(ax=ax4, linewidth=2, color=['grey'])
    
    ADHTS.plot(ax=ax4, drawstyle='steps-post', linewidth=2, color=['r','y','b','c','black'],style=['-','-','-','-','--'])
    
    ax4.legend(list_legend) # ,loc="upper left"
    
    if title is not None:
        plt.title(title)
    plt.xlabel("Hour")
    plt.ylabel(ylabel)
    
    plt.xlim(0,HoursPerDay)
    
    labelpositions = np.arange(0.5,HoursPerDay+0.5,1)
    plt.xticks(labelpositions, range(1,HoursPerDay+1))
    
    minor_locator = AutoMinorLocator(2)
    plt.gca().xaxis.set_minor_locator(minor_locator)
    plt.grid(which='minor')
    
    if fname is not None:
        plt.savefig(fname,bbox_inches = 'tight',dpi=300)









'''


fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
flatline.plot(ax=ax2, figsize=(8 , 4.5), linewidth=2, color=['grey'])

y2 = ADHTS_LCoE.div(Total_LCoE.iloc[0,4])
y2 = y2.append(pd.DataFrame(y2.iloc[-1,:]).T, ignore_index=True)
y2.plot(ax=ax2, drawstyle='steps-post', linewidth=4, color=['r','y','b','c','black'])


ax2.legend(list_legend_extended)


plt.title('Average-day Value of Electricity for Case {}'.format(caseID))
plt.xlabel("Hour")
plt.ylabel("LVoE [normalised]")
#plt.grid()

plt.xlim(0,konst_hoursperday)

labelpositions = np.arange(0.5,konst_hoursperday+0.5,1)
plt.xticks(labelpositions, range(1,konst_hoursperday+1))

minor_locator = AutoMinorLocator(2)
plt.gca().xaxis.set_minor_locator(minor_locator)
plt.grid(which='minor')

plt.savefig(resultpath + "Plot_ADHTS_LCoE_{}.png".format(caseID), bbox_inches = 'tight')




'''









