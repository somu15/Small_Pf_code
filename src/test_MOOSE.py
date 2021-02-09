#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 21:03:41 2021

@author: dhulls
"""

############## Navier-Stokes ##################

import os
import csv
import numpy as np

visc = 0.006
dens = 0.89
utop = 1.0
ubot = 0.5
uright = 1.5
uleft = 0.5

file1 = open('/Users/dhulls/projects/moose/modules/navier_stokes/test/tests/ins/2d_steady/NS.i', 'r') 
Lines = file1.readlines()
Lines[113] = "    "+"prop_values = '"+str(dens)+" "+str(visc)+"'\n"
Lines[80] = "    "+"values = '"+str(utop)+" 0.0 0.0'\n"
Lines[86] = "    "+"values = '"+str(-ubot)+" 0.0 0.0'\n"
Lines[92] = "    "+"values = '0.0 "+str(-uleft)+" 0.0'\n"
Lines[98] = "    "+"values = '0.0 "+str(uright)+" 0.0'\n"

file1 = open('/Users/dhulls/projects/moose/modules/navier_stokes/test/tests/ins/2d_steady/NS.i', 'w') 
file1.writelines(Lines) 
file1.close() 

os.chdir('/Users/dhulls/projects/moose/modules/navier_stokes/test/tests/ins/2d_steady')
os.system('/Users/dhulls/projects/moose/modules/navier_stokes/navier_stokes-opt -i NS.i')

path1 = '/Users/dhulls/projects/moose/modules/navier_stokes/test/tests/ins/2d_steady/NS.csv'
with open(path1) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    Samp0 = []
    Samp1 = []
    count = 0
    for row in readCSV:
        if count > 0:
            Samp0.append(float(row[1]))
            Samp1.append(float(row[2]))
        count = count + 1

resultant_NS = np.sqrt(Samp0[1]**2+Samp1[1]**2)

############## Stokes ##################

import os
import csv
import numpy as np

visc = 0.006
dens = 0.89
utop = 1.0
ubot = 0.5
uright = 1.5
uleft = 0.5

file1 = open('/Users/dhulls/projects/moose/modules/navier_stokes/test/tests/ins/2d_steady/S.i', 'r') 
Lines = file1.readlines()
Lines[109] = "    "+"prop_values = '"+str(dens)+" "+str(visc)+"'\n"
Lines[76] = "    "+"values = '"+str(utop)+" 0.0 0.0'\n"
Lines[82] = "    "+"values = '"+str(-ubot)+" 0.0 0.0'\n"
Lines[88] = "    "+"values = '0.0 "+str(-uleft)+" 0.0'\n"
Lines[94] = "    "+"values = '0.0 "+str(uright)+" 0.0'\n"

file1 = open('/Users/dhulls/projects/moose/modules/navier_stokes/test/tests/ins/2d_steady/S.i', 'w') 
file1.writelines(Lines) 
file1.close() 

os.chdir('/Users/dhulls/projects/moose/modules/navier_stokes/test/tests/ins/2d_steady')
os.system('/Users/dhulls/projects/moose/modules/navier_stokes/navier_stokes-opt -i S.i')

path1 = '/Users/dhulls/projects/moose/modules/navier_stokes/test/tests/ins/2d_steady/S.csv'
with open(path1) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    Samp0 = []
    Samp1 = []
    count = 0
    for row in readCSV:
        if count > 0:
            Samp0.append(float(row[1]))
            Samp1.append(float(row[2]))
        count = count + 1

resultant_S = np.sqrt(Samp0[1]**2+Samp1[1]**2)


