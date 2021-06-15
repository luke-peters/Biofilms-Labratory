import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def func1(x, a, b, c):
    return a * (x ** -b) +c

def powerlaw(c, v):
    xdata = np.array(c)
    ydata = np.array(v)
    #plt.scatter(xdata, ydata, marker='.', s=5, label='original data')
    plt.xlabel('curvature (micrometer^-1)')
    plt.ylabel('average velocity (micrometer/min)')

    popt, pcov = curve_fit(func1, xdata, ydata, maxfev = 200000)

    y2 = [func1(i, popt[0], popt[1], popt[2]) for i in np.linspace(0.002, 0.03, 1000)]
    plt.plot(np.linspace(0.002, 0.03, 1000), y2, 'r--', label = 'power law fit')
    #plt.legend(loc=1)
    #plt.show()

def func2(x, a, b, c, d):
    return a / (np.exp(-b * x) + c) ** d

#def func(x, a, b, c):
    #return a*x**b+c


def sigmoid(t, r):
    xdata = np.array(t)
    ydata = np.array(r)

    plt.scatter(xdata, ydata, marker='.', s=5, label='original data')
    popt, pcov = curve_fit(func2, xdata, ydata, maxfev = 200000)

    y2 = [func2(i, popt[0], popt[1], popt[2], popt[3]) for i in np.linspace(0, 82, 1000)]
    #y2 = [func(i, popt[0], popt[1], popt[2]) for i in np.linspace(0, 70, 1000)]

    plt.plot(np.linspace(0, 82, 1000), y2, 'r--', label = 'sigmoid fit')
    #plt.plot(np.linspace(0, 70, 1000), y2, 'r--', label = 'r =' + str(popt[0]) + ' * t ** ' + str(popt[1]) + ' + ' + str(popt[2]))


# take in simulation data
def get_records(file_name):
    general_table = []
    with open(file_name+".csv", 'r') as f:
    #with open('./buffer/homogenous.csv', 'r') as f:
        reader = csv.reader(f)
        general_table = list(reader)
    f.close()
    return general_table


# grid define
def grid_preparation():
    for i in range(0, r_max+1):
        grid.append([])
        r.append((i+0.5)*r_grid_length)
        for j in range(0, t_max+1):
            grid[i].append([0, 0])
            if i==0:
                t.append(j*t_grid_length)


# put data in grid and calculate average concentration
def average_concentration():
    l = len(raw_data)
    for i in range(1, l):
        R = int(np.sqrt(float(raw_data[i][3])**2 + float(raw_data[i][4])**2)/r_grid_length)
        T = int(float(raw_data[i][1])/t_grid_length)
        if raw_data[i][0] == '-1':
            raw_data[i][0] = '0'
        grid[R][T][0] *= grid[R][T][1]
        grid[R][T][0] += float(raw_data[i][0])
        grid[R][T][1] += 1
        grid[R][T][0] /= grid[R][T][1]

# search for peaks, return indice
def peak(v):
    v_max = 0
    ptr = 0
    for i in range(0, len(v)):
        if v[i] > v_max:
            ptr = i
            v_max = v[i]
    return ptr

def profile_ct():
    for j in range(0, len(r)):
        rr = int(r[j]/r_grid_length)
        c = []
        temp = []
        for i in range(0, t_max+1):
            if grid[rr][i][0]>0 and t[i]<t_cut and r[j]<r_cut: 
            #if grid[rr][i][0]>0 and t[i]<300:   
                c.append(grid[rr][i][0])
                temp.append(t[i])
        #plt.plot(temp, c, label = str(r[j])+'micrometers')
        plt.plot(temp, c)
        k = peak(c)
        if len(temp)>0:    
            pk_ct[0].append(r[j])
            pk_ct[1].append(temp[k])
    plt.xlabel('Time (min)')
    plt.ylabel('Potassium Concentration (Aribitrary Units)')
    #plt.legend(loc=1, title='radius +/- '+ str(r_grid_length/2))
    plt.title('Potassium Profile At Different Radii (grid_length='+r_grid_length+')')
    plt.xticks(np.arange(180, 330, 10))
    plt.yticks(np.arange(0, 55, 5))
    plt.show()

def all_peaks():
    plt.scatter(pk_ct[1], pk_ct[0], marker='.', s=5)
    plt.xlabel('time (min)')
    plt.ylabel('peak position (micrometer)')
    plt.title('all peak points obtained')
    plt.show()

# sort peak points according to time
def take_t(item):
    return item[1]

def average_peak_position():
    l = len(pk_ct[0])
    temp = []
    for i in range(0,l):
        temp.append([pk_ct[0][i], pk_ct[1][i]])

    temp.sort(key=take_t)

    for i in range(0,l):
        pk_ct[0][i] = temp[i][0]
        pk_ct[1][i] = temp[i][1]

    # calculate and plot average peak position against time
    l = len(pk_ct[0])
    count = 0
    t_up = -1
    temp = []
    for i in range(0, l):
        if pk_ct[1][i] > t_up:
            if t_up > -1:

                step_error.append(r_grid_length/(2*np.sqrt(count)))
                m = 0
                for j in range(0, len(temp)):
                    m += ((temp[j] - pkr[-1])**2)/count
                stat_error.append(np.sqrt(m)/2)
                

            temp.clear()
            temp.append(pk_ct[0][i])
            
            
            t_up = pk_ct[1][i]
            pkt.append(pk_ct[1][i]-190)
            pkr.append(pk_ct[0][i])
            count = 1
            
        else:
            temp.append(pk_ct[0][i])



            pkr[-1] *= count
            pkr[-1] += pk_ct[0][i]
            count += 1
            pkr[-1] /= count

    step_error.append(r_grid_length/(2*np.sqrt(count)))
    m = 0
    for j in range(0, len(temp)):
        m += ((temp[j] - pkr[-1])**2)/count
    stat_error.append(np.sqrt(m)/2)

    
    #plt.errorbar(pkt, pkr, fmt='none', yerr=stat_error, ecolor='dodgerblue', elinewidth=3, ms=5, mfc='wheat', mec='salmon', capsize=3, label='statistic')
    #plt.errorbar(pkt, pkr, fmt='none', yerr=step_error, ecolor='red', elinewidth=3, ms=5, mfc='wheat', mec='salmon', capsize=3, label='step')
    plt.scatter(pkt, pkr, marker='.', s=10)
    plt.xlabel('time (min)')
    plt.ylabel('average peak position (micrometer)')
    plt.title('movement of peak')
    #plt.legend(loc=2)
    #sigmoid(pkt, pkr)
    plt.show()

# instantaneous/average velocity
def cal_velocity():
    l = len(pkr)
    for i in range(1, l):
        #velocity.append((pkr[i]-pkr[i-1])/(pkt[i]-pkt[i-1])) # instantaneous velocity
        curvature.append(1/(pkr[i]))
        velocity.append((pkr[i])/(pkt[i])) # average velocity
        stat_errorr.append(np.sqrt((stat_error[i]/pkt[i])**2))
        step_errorr.append(np.sqrt((r_grid_length/pkt[i])**2 + ((pkr[i]*t_grid_length)/(pkt[i]**2))**2))
        #curvature.append(1/(pkr[i]))

    plt.errorbar(curvature, velocity, fmt='none', yerr=stat_errorr, ecolor='dodgerblue', elinewidth=3, ms=5, mfc='wheat', mec='salmon', capsize=3, label='statistical')
    plt.errorbar(curvature, velocity, fmt='none', yerr=step_errorr, ecolor='red', elinewidth=3, ms=5, mfc='wheat', mec='salmon', capsize=3, label='step')
    plt.scatter(curvature, velocity, marker='.', s=5)
    plt.xlabel('curvature (micrometer^-1)')
    plt.ylabel('peak velocity (micrometer/min)')
    plt.title('velocity at different curvatures')
    plt.legend(loc=1)
    powerlaw(curvature, velocity)
    plt.show()




raw_data = get_records('original_mini')

# grid parameter
r_grid_length = 0.2
r_limit = 900
t_grid_length = 0.05
t_limit = 380

#data_selection
r_cut = 800
t_cut = 260

r_max = int(r_limit/r_grid_length)
t_max = int(t_limit/t_grid_length)

grid = []
r = []
t = []

grid_preparation()

average_concentration()

# c versus t - fix r
pk_ct = [[], []] # 0-r, 1-t

profile_ct()

all_peaks()

# peak trajectory
pkr = []
pkt = []
stat_error = []
step_error = []

average_peak_position()

f1 = open("./output/trajectory.txt",'w')
print('pkr = ', pkr, file=f1)
print('pkt = ', pkt, file=f1)
print('stat_error = ', stat_error, file=f1)
print('step_error = ', step_error, file=f1)

# velocity
velocity = []
curvature = []
stat_errorr = []
step_errorr = []

cal_velocity()

f2 = open("./output/velocity.txt",'w')
print('velocity = ', velocity, file=f2)
print('curvature = ', curvature, file=f2)
print('stat_errorr = ', stat_errorr, file=f2)
print('step_errorr = ', step_errorr, file=f2)








