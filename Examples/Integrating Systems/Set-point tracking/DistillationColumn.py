# System  and controller found at:
# A. Alvarez, E. M. Francischinelli, B. F. Santoro, D. Odloak, Stable Model Predictive Control for Integrating Systems with Optimizing Targets, Industrial & Engineering Chemistry Research 48 (20) (2009) 9141–9150. URL http://dx.doi.org/10.1021/ie900400j

# State-Space Model:
# M. A. Rodrigues, D. Odloak, An infinite horizon model predictive control for stable and integrating processes, Computers and Chemical Engineering 27 (8-9) (2003) 1113–1128. URL http://dx.doi.org/10.1016/S0098-1354(03)00040-1

import sys
sys.path.append('/code')

from ihmpclab.components import Model, IHMPC, ClosedLoop
from control import tf

# system linearized at
yss = [47, 52.5, 91]
uss = [4.7, 2650, 88.5]

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
plant = Model(g, Ts, yss=yss, uss=uss)
controllermodel = Model(g, Ts, yss=yss, uss=uss)

controllermodel.labels["inputs"] = ['F /(ton/h)', 'Q /($m^3$/d)', 'T /(°C)']
controllermodel.labels["outputs"] = ['Level %', '$T /(C)$', 'Flood %']
controllermodel.labels["time"] = 'Time /(min)'

# Creating the controller
m = 3                # control horizon
qy = [1, 1, 1]       # output weights
r = [1, 1, 1]        # input movement weights
sy = 1e3             # output slack weights
si = [1e5, 0, 0]     # integrating states slack weights
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
umin = [4.0, 2400, 85]
umin = [umin[i] - uss[i] for i in range(len(uss))]
umax = [5.5, 2800, 90]
umax = [umax[i] - uss[i] for i in range(len(uss))]
dumax = [0.2, 25, 0.5]  # maximum variation of input moves

ysp = [[0, 0, 0], [49-yss[0], 52.2-yss[1], 85-yss[2]], [49-yss[0], 53-yss[1], 90-yss[2]]]
utarget = [[0, 0, 0], [-0.2, -100, -1.5], [0, 0, 0]]
ysp_change = [5, 100]
tf = 200

closedloop.configPlot.folder = '../../../results'
closedloop.configPlot.subfolder = 'Integrating - Distillation Column - Set-point Tracking'
closedloop.simulation(tf, u0=u0, umin=umin, umax=umax, dumax=dumax, spec_change=ysp_change, ysp=ysp, input_target=utarget)

