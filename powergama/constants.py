# -*- coding: utf-8 -*-
"""
Created on Fri Nov 01 12:49:28 2013

@author: hsven
"""

baseS = 100.0e6
baseV = 400.0e3
loadshedcost = 1000.0
hoursperyear = 8760.0
MWh_per_GWh = 1000.0

# Derived quantities
baseZ = baseV**2/baseS
baseMVA = baseS*1e-6
