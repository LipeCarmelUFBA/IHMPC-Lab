# System found at:
# A. Alvarez, E. M. Francischinelli, B. F. Santoro, D. Odloak, Stable Model Predictive Control for Integrating Systems with Optimizing Targets, Industrial & Engineering Chemistry Research 48 (20) (2009) 9141–9150. URL http://dx.doi.org/10.1021/ie900400j

# State-Space Model:
# D. Odloak, Extended robust model predictive control, AIChE Journal 50 (8) (2004) 1824–1836. URL https://doi.org/10.1002/aic.10175

# Control formulation:
# D. D. Santana, Economic and distributed model predictive control for non-stable processes : stability , feasibility , and integration, Ph.D. thesis, Universidade Federal da Bahia (2020).

import sys
sys.path.append('/code')

from ihmpclab.components import Model, IHMPC, ClosedLoop
from control import tf
from numpy import array


# numerators and denominators of G:
num = [[[4.7], [1.4e-3]],
       [[1.9], [61e-3]]]

den = [[[9.3, 1], [6.8, 1]],
       [[10.1, 1], [6.6, 1]]]

# defining the transfer function matrix
Ts = 1  # sampling time in min
g = tf(num, den)

# setting the model objects
plant = Model(g, Ts)

controllermodel = Model(g, Ts)

controllermodel.labels["inputs"] = ['F /(ton/h)', 'Q /($m^3$/d)']
controllermodel.labels["outputs"] = ['$T /(C)$', 'Flood %']
controllermodel.labels["time"] = 'Time /(min)'

# Creating the controller
m = 4                # control horizon
qy = [1, 1]          # output weights
r = [1, 1 / 60]      # input movement weights
sy = 100             # output slack weights
controller = IHMPC(controllermodel, m, qy, r, sy=sy, zone=1)

# Economic objective
fss = 0                            # Economic objective at steady state
dfdy = array([0, 1], ndmin=2)      # ysp for the second output
dfdu = array([0, 0], ndmin=2)
P = .1
controller.set_eco_tracking(dfdy, dfdu, fss, P)


# Kalman Filter
W = .5  # model
V = .5  # plant
controller.Kalman(W, V)

# Initial conditions
u0 = [0, 0]
x0_controller = [0] * controllermodel.nx
x0_plant = [0] * plant.nx  # dimensions are resolved by validation

# Closedloop
closedloop = ClosedLoop(plant, controller)
closedloop.initialConditions(x0_plant, x0_controller)

# Constraints
umin = [-.5, -30]    # lower bounds of inputs
umax = [.5, 30]      # upper bounds of inputs
dumax = [.2, 10]     # maximum variation of input moves

# Zones and targets
ymin = [[0, 0], [.1, -3], [.1, 0], [0, 0]]
ymax = [[0, 0], [.3, -1], [.3, 2], [0, 0]]
esp = [0, -1.5, .5, 0]
ysp_change = [10, 600, 800]

tf = 850
closedloop.configPlot.folder = '../../../../../results'
closedloop.configPlot.subfolder = 'Stable - Distillation Column - Linear economic performance function'
closedloop.simulation(tf, u0=u0, umin=umin, umax=umax, dumax=dumax, spec_change=ysp_change, ymin=ymin, ymax=ymax, esp=esp)
