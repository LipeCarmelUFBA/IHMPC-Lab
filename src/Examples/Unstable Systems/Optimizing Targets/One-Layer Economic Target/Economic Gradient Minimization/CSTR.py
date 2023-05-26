# System found at:
# L. P. Russo, W. B. Bequette, State-Space versus Input/Output Representations for Cascade Control of Industrial & Engineering Chemistry Research 36 (6) (1997) 2 URL https://doi.org/10.1021/ie960677o

# State-Space Model:
#  M. A. F. Martins, D. Odloak, A robustly stabilizing model predictive control strategy of stable and unstable processes, Automatica 67 (2016) 132–143. URL http://dx.doi.org/10.1016/j.automatica.2016.01.046

# Control formulation:
# L. A. Alvarez, D. Odloak, Reduction of the qp-mpc cascade structure to a single layer mpc, Journal of Process Control 24 (10) (2014) 1627–1638. URL http://dx.doi.org/10.1016/j.jprocont.2014.08.008

import os
from src.ihmpclab.components import Model, IHMPC, ClosedLoop
from control import tf
from numpy import zeros, array, hstack, vstack, loadtxt

# numerators and denominators of G:
num = [[[0.22262, 1.345, 0.95922], [0.86234]],
       [[-1.483, -8.8126, -6.9883], [-4.4694, -5.7493]],
       [[-4.449, -4.193], [-14.898, -15.536, -1.9177]]]

# denominators:
den = [[[1, 6.0428, 4.4428, -0.51411], [1, 6.0428, 4.4428, -0.51411]],
       [[1, 6.0428, 4.4428, -0.51411], [1, 6.0428, 4.4428, -0.51411]],
       [[1, 6.0428, 4.4428, -0.51411], [1, 6.0428, 4.4428, -0.51411]]]

g = tf(num, den)
Ts = 0.05  # tau

plant = Model(g, Ts)
plant.odloakModel()

controllermodel = Model(g, Ts)
controllermodel.labels["inputs"] = ['$Q_{c}$', 'Q']
controllermodel.labels["outputs"] = ['$X_{1}$', '$X_{2}$', '$X_{3}$']
controllermodel.labels["time"] = r'$\tau$'

# creating the controller
m = 3  # control horizon
qy = [1, 1, 0.1]     # output weights
r = [5, 5]           # input movement weights
sy = [5e6] * 3       # output slack weights
sun = [2e3] * 3      # unstable states slack weights
controller = IHMPC(controllermodel, m, qy, r, sy=sy, sun=sun, zone=1)

wy = array([[2.06603853, 1.09511942, 0.]])
wu = array([[0., -0.47331686]])
P = [1 for ele in range(0, controller.nu)]
controller.set_eco_gradmin(wy, wu, P)

# Kalman Filter
W = .5  # plant
V = .5  # model
controller.Kalman(W, V)

# Initial conditions
y0 = [0.35, -1.3, -0.01]
u0 = [0.0, 0.0]
xmk = vstack((array(y0, ndmin=2).transpose(), zeros((controllermodel.Fst.shape[0] + controllermodel.Fun.shape[0], 1))))

# Closedloop
closedloop = ClosedLoop(plant, controller)
closedloop.initialConditions(xmk, xmk)

# Constraints
umin = [-0.5, -1]           # lower bounds of inputs
umax = [4, 6]               # upper bounds of inputs
dumax = [1.00, 2.00]        # maximum variation of input moves

# Zones
ymin = [[-0.2, -1., -1], [-0.2, -1., -1], [-0.2, -1.0, -1.], [-0.2, -1., -1.]]
ymax = [[0.2, 2., 1.], [0.2, 2.0, -0.4], [0.2, 2.0, -0.4], [0.2, 2., 1.]]
esp = [2., 2., 3., 3.]

ysp_change = [4, 6, 10]
tf = 12

closedloop.configPlot.folder = os.path.abspath('../results')
closedloop.configPlot.subfolder = 'Unstable - CSTR - Economic Gradient Minimization'
closedloop.simulation(tf, u0=u0, umin=umin, umax=umax, dumax=dumax, spec_change=ysp_change, ymin=ymin, ymax=ymax, esp=esp)
