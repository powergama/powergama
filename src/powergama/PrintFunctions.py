# -*- coding: utf-8 -*-
'''
Module containing print functions
'''

#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
#import matplotlib as mpl
#import numpy as np

import math

#import powergama.database as db
#import csv
    
def PrintTime(text,time,indent=1):
    
    HoursUsed = math.floor(time/60/60)
    MinutesUsed = math.floor(((time/60/60)-HoursUsed)*60)
    SecondsUsed = int(round(((((time/60/60)-HoursUsed)*60)-MinutesUsed)*60))
    
    if indent==0:
        indentstring=""
    elif indent==1:
        indentstring="    "
    elif indent==2:
        indentstring="        "
    elif indent==3:
        indentstring="            "
    elif indent==4:
        indentstring="                "
    else:
        print("WRONG INDENT SPECIFICATION")
        
    if HoursUsed==0:
        if MinutesUsed==0:
            print("\n{}{}{} seconds\n"
                  .format(indentstring,text, SecondsUsed))
        else:
            print("\n{}{}{} minutes and {} seconds\n"
                  .format(indentstring,text, MinutesUsed, SecondsUsed))
    else:
        print("\n{}{}{} hours, {} minutes and {} seconds\n"
                  .format(indentstring,text, HoursUsed, MinutesUsed, SecondsUsed))
  
    
 

def Print(text,indent=1):
      
    if indent==0:
        indentstring=""
    elif indent==1:
        indentstring="    "
    elif indent==2:
        indentstring="        "
    elif indent==3:
        indentstring="            "
    elif indent==4:
        indentstring="                "
    else:
        print("WRONG INDENT SPECIFICATION")
        

    print("\n{}{}\n".format(indentstring,text))























