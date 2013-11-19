# -*- coding: utf-8 -*-
'''
Module for PowerGAMA constants
'''

## Per unit base value for power in W (100 MW)
baseS = 100.0e6

## Per unit base value for voltage in V (400 kV)
baseV = 400.0e3

## Penalty (€/MWh) for load shedding
loadshedcost = 1000.0

## Hours per year (365*24 = 8760)
hoursperyear = 8760.0

## Conversion factor from GWh to MWh
MWh_per_GWh = 1000.0

# Derived quantities
baseZ = baseV**2/baseS
baseMVA = baseS*1e-6
