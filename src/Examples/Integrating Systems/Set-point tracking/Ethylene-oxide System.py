# System, state-space and controller found at:
# M. A. Rodrigues, D. Odloak, An infinite horizon model predictive control for stable and integrating processes, Computers and Chemical Engineering 27 (8-9) (2003) 1113–1128. URL http://dx.doi.org/10.1016/S0098-1354(03)00040-1
import os
from src.ihmpclab.components import Model, IHMPC, ClosedLoop
from control import tf

# numerators and denominators of G:
num = [[[6.43e-2], [-3.67e-2], [-2.50e-2], [-9.80e-3], [2.61e-4], [-8.34e-5]],
       [[4.3], [-2.1], [-7.3], [1.2], [-2.37e-2], [2.97e-2]],
       [[-1.7], [-0.767], [-2.5], [-6.0], [4.45e-2], [2.40e-3]],
       [[-4.33e-2], [5.66e-2], [6.32e-2], [6.33e-2], [5.36e-2], [-1.59e-2]],
       [[-0.190], [0.235], [-5.96], [-0.250], [0.140], [0]]]

den = [[[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0]],
       [[15.1, 1], [19.5, 1], [199.5, 1], [7.8, 1], [49.5, 1], [55.1, 1]],
       [[19.5, 1], [31.8, 1], [17.7, 1], [199.5, 1], [49.5, 1], [10.9, 1]],
       [[1, 0], [1, 0], [1, 0], [11.4, 1], [142.4, 1], [36.5, 1]],
       [[1, 0], [1, 0], [98.5, 1], [12.5, 1], [55.1, 1], [1]]]

# defining the transfer function matrix
Ts = 1  # sampling time in min
g = tf(num, den)

# setting the model objects
plant = Model(g, Ts)
plant.incrementalModel()
plant.odloakModel()
controllermodel = Model(g, Ts)

controllermodel.labels["inputs"] = ['$F_{O2} /(ton/h)$', '$F_{C2H4} /(ton/h)$', '$F_{N2} /(ton/h)$',
                                    '$T /(C)$', '$F_{EDC} /(ton/h)$', '$F_{KOH} /(ton/h)$']
controllermodel.labels["outputs"] = ['$X_{O2}$', '$T /(C)$', 'S %$',
                                     'P', '$X_{C2H4}$']
controllermodel.labels["time"] = 'Time /(min)'

# Creating the controller
m = 2                              # control horizon
qy = [1, 10, 1, 1, 1]              # output weights
r = 0.01                           # input movement weights
sy = [400, 400, 400, 400, 400*.2]  # output slack weights
si = 5e7                           # integrating states slack weights
controller = IHMPC(controllermodel, m, qy, r, sy=sy, si=si)

# Kalman Filter
W = .5  # model
V = .5  # plant
controller.Kalman(W, V)

# Initial conditions
u0 = [0] * controllermodel.nu
x0_controller = [0] * controllermodel.nx
x0_plant = [0] * plant.nx  # dimensions are resolved by validation

# Closedloop
closedloop = ClosedLoop(plant, controller)
closedloop.initialConditions(x0_plant, x0_controller)

# Constraints
umin = [-.1, -.2, -.05, -2, -5, -10]
umax = [.1, .15, .05, 1, 15, 15]
dumax = [0.2, 0.2, 0.2, 0.5, 1, 1]  # maximum variation of input moves

ysp = [[0, 0, 0, 0, 0], [0, 0.4, 0, 0, 0]]
ysp_change = [30]
tf = 600

closedloop.configPlot.folder = os.path.abspath('../results')
closedloop.configPlot.subfolder = 'Integrating - Ethylene-oxide - Set-point Tracking'
closedloop.simulation(tf, u0=u0, umin=umin, umax=umax, dumax=dumax, spec_change=ysp_change, ysp=ysp)
