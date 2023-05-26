# System found at:
# L. P. Russo, W. B. Bequette, State-Space versus Input/Output Representations for Cascade Control of Industrial & Engineering Chemistry Research 36 (6) (1997) 2 URL https://doi.org/10.1021/ie960677o

# State-space and control formulation:
# M. A. F. Martins, D. Odloak, A robustly stabilizing model predictive control strategy of stable and unstable processes, Automatica 67 (2016) 132–143. URL http://dx.doi.org/10.1016/j.automatica.2016.01.046

# A more detailed example can be found at: https://colab.research.google.com/drive/18qOZcHWGGaJy6fijo16Z7dmx12o7R0UI

import os
from control import tf
from src.ihmpclab.components import Model, IHMPC, ClosedLoop

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
plant.odloakModel()     # We expressly convert the model to the form which will be utilized by the controller

# Model to be used in the control agent
controllermodel = Model(g, Ts, yss=yss, uss=uss)

# Labels to identify manipulated variables, controlled variables, and the time
# in order to be used in the charts.
controllermodel.labels['inputs'] = [r'Q', r'$Q_{c}$']
controllermodel.labels['outputs'] = [r'$X_{1}$', r'$X_{2}$', r'$X_{3}$']
controllermodel.labels['time'] = r'$\tau$'

# Creating the controller
# Defining the tuning parameters
m = 4                       # control horizon
qy = [1e4, 0.1, 0.5]        # weighting matrix elements related to outputs
r = [1.0, 1.0]              # weighting matrix elements related to input increments
sy = [1e5, 1e5, 1e5]        # define the matrix related to output slacks proportional to Qy.
sun = [10.0, 10.0, 10.0]    # weighting matrix elements related to the slacks of unstable states

# Defining the control object
controller = IHMPC(controllermodel, m, qy, r, sy=sy, sun=sun, zone=1)

# Defining the tuning parameters
qu = [0.1, 0.1]     # weighting matrix elements related to inputs
su = 1              # weighting matrix related to the slacks on inputs defined as proportional to qu

# Defining the economic term
controller.set_input_targets(qu, su)

W = 5e-5  # variance related to the states of the model. It is assumed that the matrix is composed of equal elements.
V = 2e-3  # variance related to the states of the plant. It is assumed that the matrix is composed of equal elements.
controller.Kalman(W, V)

umin = [-0.5, -1]   # lower bounds of inputs
umax = [4.0, 5.0]   # upper bounds of inputs
dumax = [1.5, 1.5]  # maximum variation of input moves

# Defining the zones
# These zones have been selected to provide a gradual
# reactant concentration reduction
ymin = [[0.85 - yss[0], 0.9 - yss[1], -0.1 - yss[2]],
        [0.80 - yss[0], 1.0 - yss[1], 0.0 - yss[2]],
        [0.75 - yss[0], 1.1 - yss[1], 0.1 - yss[2]]]

ymax = [[0.90 - yss[0], 1.0 - yss[1], 0.2 - yss[2]],
        [0.86 - yss[0], 1.3 - yss[1], 0.3 - yss[2]],
        [0.81 - yss[0], 1.50 - yss[1], 0.4 - yss[2]]]
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
closedloop.initialConditions(system_state=xmk, controller_state=xmk)
# Executing the simulation
folder = os.path.abspath('../results')
closedloop.configPlot.folder = folder
closedloop.configPlot.subfolder = 'Unstable - CSTR (without mismatch)- Input targets'
closedloop.simulation(end_time, u0=u0, umin=umin, umax=umax, dumax=dumax,
                      spec_change=ysp_change, ymin=ymin, ymax=ymax, input_target=utarget,
                      show=False)

# Without mismatch:

from scipy.integrate import solve_ivp
from src.ihmpclab.auxiliary import PlotManager
from numpy import array, hstack, exp, copy

# Model linearized at:
yss = array([yss]).transpose()
uss = array(uss)

# Non-Linear Model - Addressing mismatch
def cstr(y, t, u):
    # System's parameters
    phi = 0.072     # nominal Damkohler number based on the reactor feed
    beta = 8.0      # dimensionless heat of reaction
    delta = 0.3     # dimensionless heat transfer coefficient
    gama = 20       # dimensionless activation energy
    delta1 = 10     # reactor to cooling jacket volume ratio
    delta2 = 1      # reactor to cooling jacket density heat capacity ratio
    X1f = 1
    X2f = 0
    X3f = -1

    dy0dt = u[0] * (X1f - y[0]) - phi * y[0] * exp(y[1] / (1 + (y[1] / gama)))
    dy1dt = u[0] * (X2f - y[1]) - delta * (y[1] - y[2]) + beta * phi * y[0] * exp(y[1] / (1 + (y[1] / gama)))
    dy2dt = delta1 * (u[1] * (X3f - y[2]) + delta * delta2 * (y[1] - y[2]))
    return [dy0dt, dy1dt, dy2dt]


# Constraints
umin_nl = array([umin]).transpose()  # lower bounds of inputs
umax_nl = array([umax]).transpose()  # upper bounds of inputs
dumax_nl = array([dumax]).transpose()  # maximum variation of input moves

# Setting the initial conditions
controller.initialConditions(u0, xmk)

ypk = array([y0]).transpose()  # deviation value
ymk = ypk + yss  # measured nominal value

# Storing values
y_trend = ymk
ymin_trend = array([ymin[0]]).transpose()
ymax_trend = array([ymax[0]]).transpose()


# Defining the final time for the simulation
nsim: int = int(end_time / Ts) + 1


# Executing the simulation
for k in range(1, nsim):
    # Defining conditions
    if k < 0.3 * nsim:
        case = 0
    elif k < 0.6 * nsim:
        case = 1
    else:
        case = 2

    # Picking the current zones
    ymin_nl = array([ymin[case]]).transpose()
    ymax_nl = array([ymax[case]]).transpose()
    utarget_nl = array([utarget[case]]).transpose()

    # Calculating the control action
    # The controller object requires deviation variables
    # Therefore, the nominal zones must be subtracted from the equilibrium point.
    duk1, uk1 = controller.solve(ypk,
                                 umin=umin_nl,
                                 umax=umax_nl,
                                 dumax=dumax_nl,
                                 ymin=ymin_nl,
                                 ymax=ymax_nl,
                                 input_target=utarget_nl)

    # Applying the control action and updating the model
    u = copy(uss)  # The copy prevents pointer misuse
    u = u + uk1

    # Simulating the non-linear system
    ymk = solve_ivp(lambda t, y: cstr(y, t, u.flatten()), (0, Ts), ymk.flatten(), rtol=1e-10, atol=1e-10).y[:, -1:]
    # Calculation deviation
    ypk = ymk - yss
    # Storing trends
    y_trend = hstack((y_trend, ymk))
    ymin_trend = hstack((ymin_trend, ymin_nl))
    ymax_trend = hstack((ymax_trend, ymax_nl))

# Time array
time = Ts * array([i for i in range(nsim)])

plots = PlotManager(folder=folder)
subfolder = 'Unstable - Non-Linear CSTR - Set-point-Tracking'

# Controlled variables
plots.plot(time, y_trend, config_plot={'ls': '-', 'label':'Output'},
           subfolder=subfolder, filename='Outputs')
plots.plot_on_top(time[1:], controller._trendObj.get('ysp') + yss,
                  config_plot={'drawstyle': 'steps-post', 'ls': '--', 'color': 'red', 'label':'Setpoint'})
plots.plot_on_top(time, ymin_trend + yss, {'drawstyle': 'steps-post', 'ls': '--', 'color': 'k', 'label':'Zone'})
plots.plot_on_top(time, ymax_trend + yss, {'drawstyle': 'steps-post', 'ls': '--', 'color': 'k'})
plots.label(controller.model.labels['time'], controller.model.labels['outputs'])

# Manipulated variables
plots.plot(time, controller._trendObj.get('uk') + controllermodel.uss, config_plot={'color': 'k', 'label':'Input'},
           subfolder=subfolder, filename='Inputs')
plots.plot_on_top(time[1:], controller._trendObj.get('input_target') + controllermodel.uss,
                  config_plot={'drawstyle': 'steps-post', 'ls': '--', 'color': 'b', 'label':'Target'})
plots.label(controller.model.labels['time'], controller.model.labels['inputs'])

# Slack variables
plots.plot(time[1:], controller._trendObj.get('skun'), config_plot={'color': 'k'},
           subfolder=subfolder, filename='Slacks for Unstable Modes')
plots.label(controller.model.labels['time'], ['$\delta_{un' + str(i) + '}$' for i in range(1,4)])

plots.show(to_show=False)
