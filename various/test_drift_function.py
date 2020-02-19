import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.interpolate
import random
from scipy import optimize
from matplotlib.widgets import Button

# this script is a part of the Indentation assistant project
# it's goal is to test an approach for the determination of the drift function (line + sin) coefficients

def interpol(x,y,N):
	if len(x) != len(y):
		raise NameError('X and Y have different length')
	f = scipy.interpolate.interp1d(x, y)
	x_interpolated = np.linspace(min(x),max(x),N)
	y_interpolated = f(x_interpolated)
	return x_interpolated, y_interpolated

def Squeeze(array):
	span = float(max(array) - min(array))
	array_new = [x / span  for x in array]
	return array_new

def Convert(x,y,k):
	xnew = []
	ynew = []
	for i in range(0,len(x)):
		xnew.append(x[i]*k[0] + k[1])
		ynew.append(y[i]*k[2] + k[3] + xnew[i]*k[4] + k[5]*math.sin(k[6]*xnew[i]+k[7]))
	return xnew, ynew

def ErrorFunc(k, x1, y1, x2, y2, func):
	if len(x1) != len(x2) or len(x1) != len(y1) or len(x1) != len(y2):
		raise NameError('X and Y have different length')
	sum = 0
	xnew, ynew = func(x2,y2,k)
	for array in [x1,y1,xnew,ynew]:
		array = Squeeze(array)
	for i in range(0,len(x1)):
		sum += (x1[i] - xnew[i])**2 + (y1[i] - ynew[i])**2
	return sum/len(x1)

class Fit(object):
        parameters = [1, 1, 1, 1, 1, 1, 1, 1] # initial parameters
        parameters_old = [1, 1, 1, 1, 1, 1, 1, 1]
        def optimize(self, event):
                self.parameters_old = self. parameters
                solution = optimize.minimize(ErrorFunc, self.parameters,
                                                    args = (x1_interpolated, y1_interpolated,
                                                            x2_interpolated, y2_interpolated, Convert))
                self.parameters = solution.x
                xdata, ydata = Convert(x2,y2,self.parameters)
                l.set_xdata(xdata)
                l.set_ydata(ydata)
                plt.draw()
                print(solution)
        def reset(self, event):
                self.parameters = self.parameters_old
                xdata, ydata = Convert(x2,y2,self.parameters)
                l.set_xdata(xdata)
                l.set_ydata(ydata)
                plt.draw()

# get initial data points
f = lambda x: 2*math.sin(8*x)+2*x
g = lambda x: 0.7*random.gauss(0, 1)+ 0.1*x*random.gauss(0, 1)
x1 = np.arange(0,10,0.05)
y1 = [f(x)+g(x) for x in x1]
x2 = [2*x +3 for x in x1]
y2 = [3*f(x) + 3*g(x) - 50 + 10*x - 60*math.sin(0.4*x-2) for x in x1]

# optimize parameters
N = 100 # number of points taken from each array for error function evaluation
x1_interpolated, y1_interpolated = interpol(x1, y1, N)                                                 
x2_interpolated, y2_interpolated = interpol(x2, y2, N)

callback = Fit()
xnew, ynew = Convert(x2,y2,callback.parameters)

fig = plt.figure()
fig.add_subplot(211).plot(x1,y1,'o',x2,y2,'o',markersize=4)
_, l = fig.add_subplot(212).plot(x1,y1,'o',xnew,ynew,'o',markersize=4)

axoptimize = plt.axes([0.7, 0.05, 0.1, 0.075])
axreset = plt.axes([0.81, 0.05, 0.1, 0.075])
boptimize = Button(axoptimize, 'optimize')
boptimize.on_clicked(callback.optimize)
breset = Button(axreset, 'reset')
breset.on_clicked(callback.reset)

plt.show()
