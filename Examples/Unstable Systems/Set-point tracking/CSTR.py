# System found at:
# L. P. Russo, W. B. Bequette, State-Space versus Input/Output Representations for Cascade Control of Industrial & Engineering Chemistry Research 36 (6) (1997) 2 URL https://doi.org/10.1021/ie960677o

# State-Space Model:
# M. A. F. Martins, D. Odloak, A robustly stabilizing model predictive control strategy of stable and unstable processes, Automatica 67 (2016) 132–143. URL http://dx.doi.org/10.1016/j.automatica.2016.01.046

# Control formulation:
# D. Odloak, Extended robust model predictive control, AIChE Journal 50 (8) (2004) 1824–1836. URL https://doi.org/10.1002/aic.10175

# A more detailed example can be found at: https://colab.research.google.com/drive/18qOZcHWGGaJy6fijo16Z7dmx12o7R0UI
import sys
sys.path.append('/code')

from control import tf
from ihmpclab.components import Model, IHMPC, ClosedLoop


# In order to define G(s), we will define the numerator and denominator of each transfer function as:
# numerators:
num = [[[0.22262, 1.345, 0.95922], [0.86234]],
       [[-1.483, -8.8126, -6.9883], [-4.4694, -5.7493]],
       [[-4.449, -4.193], [-14.898, -15.536, -1.9177]]]

# denominators:
den = [[[1, 6.0428, 4.4428, -0.51411], [1, 6.0428, 4.4428, -0.51411]],
       [[1, 6.0428, 4.4428, -0.51411], [1, 6.0428, 4.4428, -0.51411]],
       [[1, 6.0428, 4.4428, -0.51411], [1, 6.0428, 4.4428, -0.51411]]]

# This system is obtained by linearizing the model at:
yss = [0.7774, 1.4830, 0.4898]
uss = [1, .2]

# Then, G(s) can be directly defined as
g = tf(num, den)

# Sampling time
Ts = 0.05

# Model to represent the plant - We will address the nominal case, i.e. the plant
# is represented by the same linear model of the control agent
plant = Model(g, Ts)    # Here we have an state-space model using the default option ("positional")
plant.odloakModel()     # We expressly convert the model towards

# Model to be used in the control agent
controllermodel = Model(g, Ts, yss=yss, uss=uss)

# Labels to identify manipulated variables, controlled variables, and the time
# in order to be used in the charts.
controllermodel.labels['inputs'] = [r'Q', r'$Q_{c}$']
controllermodel.labels['outputs'] = [r'$X_{1}$', r'$X_{2}$', r'$X_{3}$']
controllermodel.labels['time'] = r'$\tau$'

# Creating the controller
# Defining the tuning parameters
m = 4  # control horizon
qy = [1e4, 0.1, 0.5]        # weighting matrix elements related to outputs
r = [1.0, 1.0]              # weighting matrix elements related to input increments
sy = [1e5, 1e5, 1e5]        # define the matrix related to output slacks proportional to Qy.
sun = [10.0, 10.0, 10.0]    # weighting matrix elements related to the slacks of unstable states

# Defining the control object
controller = IHMPC(controllermodel, m, qy, r, sy=sy, zone=0, sun=sun)

# Defining the tuning parameters
qu = [0.1, 0.1]  # weighting matrix elements related to inputs
su = 1  # weighting matrix related to the slacks on inputs defined as proportional to qu

W = 5e-5  # variance related to the states of the model. It is assumed that the matrix is composed of equal elements.
V = 2e-3  # variance related to the states of the plant. It is assumed that the matrix is composed of equal elements.
controller.Kalman(W, V)

umin = [-0.5, -1]    # lower bounds of inputs
umax = [4.0, 5.0]    # upper bounds of inputs
dumax = [1.5, 1.5]   # maximum variation of input moves

# Defining the set-points
ysp = [[0.0726, -0.51539, -0.2898], [0.06838, -0.48219, -0.26276], [0.03260, -0.24089, -0.17370]]

# Defining the list containing 2 time instants to transition the zones
ysp_change = [15, 30]

# Defining the targets on inputs
# The targets specify constant flow and less heat removal
utarget = [[0.0, .1], [0.0, 0.0], [0.0, 0.0]]

# Defining the final time for the simulation
end_time = 50

# Initial condition of the plant
y0 = [0.0465, -0.332, -0.1992]
u0 = [0.0, 0.1]

# Initial condition for the model applied in the control formulation
xmk = y0 + [0] * (controllermodel.nx - controllermodel.ny)  # concatenation of lists.
# controller.nx: number of states of the model applied in controller
# controller.ny: number of outputs of the model applied in controller

# Defining the ClosedLoop object
closedloop = ClosedLoop(plant, controller)
# Setting the initial condition
closedloop.initialConditions(xmk, xmk)
# Executing the simulation
closedloop.configPlot.folder = '../../../results'
closedloop.configPlot.subfolder = 'Unstable - CSTR - Set-point tracking'
closedloop.simulation(end_time, u0=u0, umin=umin, umax=umax, dumax=dumax,
                      spec_change=ysp_change, ysp=ysp, input_target=utarget,
                      show=False)
