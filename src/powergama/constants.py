# -*- coding: utf-8 -*-
"""
Module for PowerGAMA constants
"""

baseS = 100.0e6
"""Per unit base value for power in W (100 MW)"""

baseV = 400.0e3
"""Per unit base value for voltage in V (400 kV)"""

loadshedcost = 1000.0
"""Penalty (/MWh) for load shedding and dump load"""

hoursperyear = 8760.0
"""Hours per year (365*24 = 8760)"""

MWh_per_GWh = 1000.0
"""Conversion factor from GWh to MWh"""

baseAngle = 1
"""Base value for voltage angle"""

# Derived quantities
# baseZ = baseV**2/baseS
baseMVA = baseS * 1e-6

flexload_outside_cost = 1000.0
"""(Very high) storage value for flexible demand outside flexibility range"""

spillcost = 1.0e-3
"""Small penalty (/MWh) on spilled inflow. Breaks the tie when the storage value
is zero, so that inflow is spilled only when the reservoir is full."""
