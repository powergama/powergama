'''
Mehtods to create seasonal and daily filters of hourly timeseries typically
used in PowerGAMA.
'''
from __future__ import division
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator
from math import pi, sqrt, exp, sin


def gauss(n=11,sigma=1):
    r = range(-int(n/2),int(n/2)+1)
    return [1 / (sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r]


def sinc(n=11,T=1):
    r = range(-int(n/2),int(n/2)+1)
    return [ ( sin(pi * ((float(x) + 0.000000000001 ) / T) )
              / (pi * (float(x) + 0.000000000001 ) ) ) for x in r]


def SeasonFilter(HourlyTimeSeries,FilterPeriodInMonths=4,
    PlotFilter=False, Quick=False):
    '''Create seasonal filters

    Parameters
    ----------
    HourlyTimeSeries : pandas.dataframe
        orignial time series, should be one year
    FilterPeriodInMonths : float
        filter period
    PlotFilter : boolean
        wheter to make a plot
    Quick : boolean
        whether to make a quick
    '''

    time_filter_start = time.time()

    HoursPerYear = HourlyTimeSeries.shape[0]
    HoursPerMonth = HoursPerYear/12
    FilterLengthInYears = 4*FilterPeriodInMonths

    if Quick:
        FilterLengthInYears = FilterLengthInYears/4

    FilterLengthInHours = FilterLengthInYears*HoursPerYear


    if float(FilterLengthInYears).is_integer():
        FilterLengthInYears = int(FilterLengthInYears)
    else:
        print('Problem')
        raise

    filter_T = HoursPerMonth*FilterPeriodInMonths/2
    filter_n = FilterLengthInHours-1
    filter_offset = (filter_n/2)-0.5

    IdealFilter = pd.Series(sinc(n=filter_n, T=filter_T))
    IdealFilter.index = IdealFilter.index.values - filter_offset

    sigma = (FilterLengthInHours)/10

    if Quick:
        sigma = sigma*1.5

    window = pd.Series(gauss(n=filter_n, sigma=sigma))
    window.index = window.index.values - filter_offset

    window_normalised = pd.Series(window.values / window.loc[0])
    window_normalised.index = window_normalised.index.values - filter_offset

    window_plotting = pd.Series(window_normalised.values * IdealFilter.loc[0])
    window_plotting.index = window_plotting.index.values - filter_offset

    CroppedlFilter = pd.Series(IdealFilter.values * window_normalised.values)
    CroppedlFilter.index = CroppedlFilter.index.values - filter_offset

    if PlotFilter:
        plt.figure()
        IdealFilter.plot.line(figsize=(8 , 4.5), label='Ideal filter')
        window_plotting.plot.line(label='Window')
        CroppedlFilter.plot.line(label='Cropped Filter')
        plt.legend()

    #Sum Numbers for Quality Check

    Ideal_filter_integral_error = 1 - IdealFilter.sum()
    Ideal_filter_Cut_off = IdealFilter.iloc[0]/IdealFilter.loc[0]

    Window_integral_error = 1 - window.sum()
    Window_Cut_off = window_normalised.iloc[0]/window_normalised.loc[0]

    Combined_filter_integral_error = 1 - CroppedlFilter.sum()
    Combined_filter_Cut_off = CroppedlFilter.iloc[0]/CroppedlFilter.loc[0]

    HourlyTimeSeriesExtended = HourlyTimeSeries

    for counterfilterlength in range(FilterLengthInYears):
        HourlyTimeSeriesExtended = HourlyTimeSeriesExtended.append(
            HourlyTimeSeries, ignore_index=True)

    HourlyTimeSeriesExtended.index = (
        HourlyTimeSeriesExtended.index.values - (FilterLengthInHours/2))
    HourlyTimeSeriesFiltered = HourlyTimeSeries*0

    for counter_hours in range(HoursPerYear):
        temp1 = HourlyTimeSeriesExtended.iloc[:,:].mul(CroppedlFilter, axis=0)
        temp2 = pd.DataFrame(temp1.sum(axis=0)).T

        HourlyTimeSeriesFiltered.iloc[counter_hours, :] = temp2.iloc[0,:]
        HourlyTimeSeriesExtended.index = HourlyTimeSeriesExtended.index.values - 1

    time_filter_end = time.time()
    return HourlyTimeSeriesFiltered


def SeasonPlot(HourlyTimeSeries,fname=None,ylabel='',title=None):

    plt.rcParams.update({'font.size': 22})
    list_monate_kurz = ['Jan.','Feb.','Mar.','Apr.','Mai','Jun.',
                        'Jul.','Aug.','Sep.','Okt.','Nov.','Dec.']
    HoursPerYear = HourlyTimeSeries.shape[0]
    HoursPerMonth = HoursPerYear/12

    list_legend = list(HourlyTimeSeries.columns)
    list_legend.insert(0,'_')

    flatline = pd.DataFrame([1 for i in range(HoursPerYear)])

    fig4 = plt.figure(figsize=(16 , 9))
    ax4 = fig4.add_subplot(111)
    flatline.plot(ax=ax4, linewidth=2, color=['grey'])

    HourlyTimeSeries.plot(ax=ax4, linewidth=2,
        color=['r','y','b','c','black'],style=['-','-','-','-','--'])

    ax4.legend(list_legend,loc="lower left")

    if title is not None:
        plt.title(title)
    plt.xlabel("Month")
    plt.ylabel(ylabel)

    plt.xlim(0,HoursPerYear)

    labelpositions = np.arange(0.5*HoursPerMonth,(12.5)*HoursPerMonth,1*HoursPerMonth)
    plt.xticks(labelpositions, list_monate_kurz)

    minor_locator = AutoMinorLocator(2)
    plt.gca().xaxis.set_minor_locator(minor_locator)
    plt.grid(which='minor')

    if fname is not None:
        plt.savefig(fname,bbox_inches = 'tight',dpi=300)



def DayFilter(HourlyTimeSeries):
    '''Filter timeseries

    Parameters
    ----------
    HourlyTimeSeries : pandas.DataFrame
        time series to be filtered. Should be hourly time steps
    '''

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
    '''Plot the output from DayFilter'''

    plt.rcParams.update({'font.size': 22})

    HoursPerDay = 24
    list_legend = list(ADHTS.columns)
    list_legend.insert(0,'_')

    ADHTS = ADHTS.append(pd.DataFrame(ADHTS.iloc[-1,:]).T, ignore_index=True)

    flatline = pd.DataFrame([1 for i in range(HoursPerDay+1)])

    fig4 = plt.figure(figsize=(16 , 9))
    ax4 = fig4.add_subplot(111)
    flatline.plot(ax=ax4, linewidth=2, color=['grey'])

    ADHTS.plot(ax=ax4, drawstyle='steps-post', linewidth=2,
        color=['r','y','b','c','black'],style=['-','-','-','-','--'])

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
