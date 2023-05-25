# System found at:
# A. Alvarez, E. M. Francischinelli, B. F. Santoro, D. Odloak, Stable Model Predictive Control for Integrating Systems with Optimizing Targets, Industrial & Engineering Chemistry Research 48 (20) (2009) 9141–9150. URL http://dx.doi.org/10.1021/ie900400j

# State-Space Model:
# M. A. Rodrigues, D. Odloak, An infinite horizon model predictive control for stable and integrating processes, Computers and Chemical Engineering 27 (8-9) (2003) 1113–1128. URL http://dx.doi.org/10.1016/S0098-1354(03)00040-1

# Control formulation:
# D. D. Santana, Economic and distributed model predictive control for non-stable processes : stability , feasibility , and integration, Ph.D. thesis, Universidade Federal da Bahia (2020).

import sys
sys.path.append('/code')

from ihmpclab.components import Model, IHMPC, ClosedLoop
from control import tf
from numpy import array

# numerators and denominators of G:
num = [[[2.3], [-0.7e-3], [0.2]],
       [[4.7], [1.4e-3], [0.4]],
       [[1.9], [61e-3], [0.2]]]

den = [[[1, 0], [1, 0], [1, 0]],
       [[9.3, 1], [6.8, 1], [11.6, 1]],
       [[10.1, 1], [6.6, 1], [12.3, 1]]]
# defining the transfer function matrix
Ts = 1  # sampling time in min
g = tf(num, den)

# setting the model objects
plant = Model(g, Ts)
plant.odloakModel()


controllermodel = Model(g, Ts, uss=[4.7, 2650, 88.5], yss=[47, 52.5, 91])
controllermodel.labels["inputs"] = ['F /(ton/h)', 'R /($m^3$/d)', 'T_{f}/(°C)']
controllermodel.labels["outputs"] = ['Level %', 'Flood %', 'T/(°C)']
controllermodel.labels["time"] = 'Time /(min)'

# Creating the controller
m = 4  # control horizon
qy = [1, 1, 1]       # output weights
r = [10, 10, 10]     # input movement weights
sy = 1e3             # output slack weights
si = 1e5             # integratings slack weights
controller = IHMPC(controllermodel, m, qy, r, sy=sy, si=si, zone=1)

# Economic objective
fss = 5     # maximum level
dfdy = array([-1, 0, 0], ndmin=2) # Tracking the level of liquid in the top drum
dfdu = array([0, 0, 0], ndmin=2)
P = .5
controller.set_eco_tracking(dfdy, dfdu, fss, P)

# Kalman Filter
W = .5  # model
V = .5  # plant
controller.Kalman(W, V)

# Initial conditions
u0 = [0, 0, 0]
x0_controller = [0] * controllermodel.nx
x0_plant = [0] * plant.nx  # dimensions are resolved by validation

# Closedloop
closedloop = ClosedLoop(plant, controller)
closedloop.initialConditions(x0_plant, x0_controller)

# Constraints
umin = [-.7, -250, -3.5]    # lower bounds of inputs
umax = [.8, 150, 1.5]       # upper bounds of inputs
dumax = [.2, 25, 0.5]       # maximum variation of input moves

# Zones and targets
ymin = [[-2, -0.5, -21], [-2, -0.5, -21]]
ymax = [[5, 0.5, -1], [2, 0.5, -1]]
esp = [0, 0]    # The economic target is made infeasible by the contraction of the zones
ysp_change = [80]

tf = 150
closedloop.configPlot.folder = '../../../../../results'
closedloop.configPlot.subfolder = 'Integrating - Distillation Column - Linear economic performance function'
closedloop.simulation(tf, u0=u0, umin=umin, umax=umax, dumax=dumax, spec_change=ysp_change, ymin=ymin, ymax=ymax, esp=esp)
