import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math

#Retrieving data and making a raw data list that will not be changed
def get_records(file_name):
    general_table = []
    with open(file_name+".csv", 'r') as f:
    #with open('./buffer/homogenous.csv', 'r') as f:
        reader = csv.reader(f)
        general_table = list(reader)
    f.close()
    return general_table
raw_data=get_records('original')

#Define lists to fill
l = len(raw_data) # length of the csv file
t_real = []
t_rounded = [] #time
c = [] #conc
r_real = [] 
r_bins = [] # distance to the origin
x = [] # x coordinate
y = [] # y coordinate

#Fill lists from raw data
for i in range(1, l):
    t_real.append(float(raw_data[i][1]))
    t_rounded.append(round(float(raw_data[i][1]),2))
    #Set any -1 concentrations to zero
    if float(raw_data[i][0])<=0:
        c.append(0)
    if float(raw_data[i][0])>0:
        c.append((raw_data[i][0]))
    if c[0]==-1:
        c[0]
    r_real.append((np.sqrt(float(raw_data[i][3])**2 + float(raw_data[i][4])**2)))
    x.append(float(raw_data[i][3]))
    y.append(float(raw_data[i][4]))

#Place r in bins for plotting
r_width=5
for i in r_real:
    bin_no=(math.floor(i/r_width))
    r_bins.append(bin_no*r_width+r_width/2)

plot_radius=227.5
temp_c = []
temp_t = []
temp_r = []

for i in range(1, len(r_bins)):
    if r_bins[i] == plot_radius:
        temp_t.append(t_real[i])
        temp_c.append(c[i])
        temp_r.append(r_bins[i])

plt.scatter(temp_t,temp_c,s=0.1,label='radius='+str(plot_radius))
    


plot_radius=527.5
temp_c = []
temp_t = []
temp_r = []

for i in range(1, len(r_bins)):
    if r_bins[i] == plot_radius:
        temp_t.append(t_real[i])
        temp_c.append(c[i])
        temp_r.append(r_bins[i])

plt.scatter(temp_t,temp_c,s=0.1)
    


