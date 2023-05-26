# System found at:
# A. Alvarez, E. M. Francischinelli, B. F. Santoro, D. Odloak, Stable Model Predictive Control for Integrating Systems with Optimizing Targets, Industrial & Engineering Chemistry Research 48 (20) (2009) 9141–9150. URL http://dx.doi.org/10.1021/ie900400j

# Model and controller based on:
# D. Odloak, Extended robust model predictive control, AIChE Journal 50 (8) (2004) 1824–1836. URL https://doi.org/10.1002/aic.10175
import os
from src.ihmpclab.components import Model, IHMPC, ClosedLoop
from control import tf

# numerators and denominators of G:
num = [[[4.7], [1.4e-3]],
       [[1.9], [61e-3]]]

den = [[[9.3, 1], [6.8, 1]],
       [[10.1, 1], [6.6, 1]]]

# defining the transfer function matrix
Ts = 1.5  # sampling time in min
g = tf(num, den)

# setting the model objects
plant = Model(g, Ts, "Positional")
controllermodel = Model(g, Ts)

controllermodel.labels["inputs"] = ['F /(ton/h)', 'Q /($m^3$/d)']
controllermodel.labels["outputs"] = ['$T /(C)$', 'Flood %']
controllermodel.labels["time"] = 'Time /(min)'

# creating the controller
m = 4                # control horizon
qy = [1, 1]          # output weights
r = [1, 1 / 60]      # input movement weights
sy = 100             # output slack weights
controller = IHMPC(controllermodel, m, qy, r, sy=sy, zone=0)

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

ysp = [[0, 0], [.1, -1], [.1, 1], [0, 0]]  # Set-point values
ysp_change = [10, 50, 100]

tf = 150
closedloop.configPlot.folder = os.path.abspath('../results')
closedloop.configPlot.subfolder = 'Stable - Distillation Column - Set-point Tracking'
closedloop.simulation(tf, u0=u0, umin=umin, umax=umax, dumax=dumax, spec_change=ysp_change, ysp=ysp)
