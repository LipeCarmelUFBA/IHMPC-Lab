# -*- coding: utf-8 -*-
import os

from src.ihmpclab.auxiliary import Kalman, PlotManager, Trend

from copy import deepcopy
from math import ceil
from warnings import warn, catch_warnings, simplefilter

# Systems (using control v. 0.9.0)
# Requires slycot (using v. 0.3.5.0)
from control import tf, ss, issys, sample_system
from control.xferfcn import TransferFunction

# Matrices (using numpy v. 1.18.1)
from numpy import diag, array, identity, ones, zeros, tile, dot, hstack, vstack, max, exp, real, imag, delete, all, \
    polymul, diff, ceil, sort, isinf, savetxt
from numpy.linalg import multi_dot, matrix_power, inv, eig, pinv

# Algebra (using scipy v. 1.4.1)
from scipy.io import savemat
from scipy.linalg import solve_discrete_lyapunov
from scipy.signal import residue

# Optimization
# Using cvxopt v. 1.2.3
from cvxopt import matrix, solvers
from cvxopt.solvers import qp

# optimizers by priority:
available_optimizers = ['quadprog', 'cvxopt']
# Using quadprog if available
try:
    from quadprog import solve_qp
except ImportError:
    warn('Could not import quadprog, only cvxopt can be used for optimization')
    available_optimizers.remove('quadprog')

solvers.options['show_progress'] = False


class Model:
    """
    Discrete state-space system.
    ...

    Attributes
    ----------
    A : numpy array
        State matrix
        Dimensions: Number of states x Number of states.

    B : numpy array
        Input matrix
        Dimensions: Number of states  x Number of inputs.

    C : numpy array
        Output matrix.
        Dimensions: Number of outputs  x Number of states.

    trend : dict
        Contains the simulated data.

    ymk : numpy array
        Current output.

    xmk : numpy array
        Current state.

    nx : numpy array
        Number of states.

    nu : numpy array
        Number of inputs.

    ny : numpy array
        Number of outputs.

    Methods
    -------
    method(arg=type)
        <Description>.
    """

    class Library:
        # TODO: this class is underused, consider removing it
        """
        Keeps a dict containing every transfer function or state-space object used to represent the original system.
        ...

        Attributes
        ----------
        versions : dict
            A dictionary containing transfer function and state-space model objects.
        key : str
            Key for the object provided to __init__

        Methods
        -------
        get_as(form=string, dt=None)
            Returns the lti object corresponding to the form string.
            If an appropriate object is not available in 'versions', converts and returns a copy of
            the original object.
        """

        def __init__(self, lti):
            """
            Stores an LTI system as the original
            version of the user's system.

            Parameters
            ----------
            lti : control.StateSpace or control.TransferFunction object
            """

            # Form (transfer function or state-space)
            if isinstance(lti, TransferFunction):
                form = "tf"
            else:
                form = "ss"

            # Backwards compatibility. lti.dt is None for control 0.8.4
            if lti.dt is None or lti.dt == 0:
                form += "c"  # continuous
            else:
                form += "d"  # discrete

            # Storing current version
            self.versions = {form: deepcopy(lti)}
            # Key indicating the original model
            self._key = form

        @property
        def key(self):
            return self._key

        def get_as(self, form, dt=None):
            """
            Returns the lti object corresponding to the form string.
            If an appropriate object is not available in 'versions', converts and returns a copy of
            the original object.

            Parameters
            ----------
            form : str
                Object to return:
                    'ssc' - Continuous state-space model object.
                    'ssd' - Discrete state-space model object.
                    'tfc' - Continuous transfer function model object.
                    'tfd' - Discrete transfer function model object.

            dt : float
                Sampling time for discrete models.
            """

            if form not in {'ssc', 'ssd', 'tfc', 'tfd'}:
                raise ValueError('Incorrect form. Use a string valued ssc, ssd, tfc or tfd.')

            if form[2] == 'd' and dt is None:
                raise ValueError('Discretization without sampling time.')

            # Check available versions
            if form in self.versions:
                if form[1] == 'c':
                    return self.versions[form]
                else:
                    # If discrete, determine if the sampling time is as requested
                    if dt == self.versions[form].dt:
                        return self.versions[form]

            # Handle exception
            # TODO: if control.sample_system is updated to handle MIMO transfer functions, erase this if statement
            if form == 'tfd':
                # as of 0.9.0, MIMO transfer fuctions can not be discretized
                # control.sample_system throws: "NotImplementedError: MIMO implementation not available"
                # To circumvent this limitation, first convert discrete state-space and then to transfer function
                self.versions[form] = tf(self.get_as('ssd', dt))
                return self.versions[form]

            # Converts to the correct representation
            if self.key[0:2] == 'ss' and form[0:2] == 'tf':
                aux = tf(self.versions[self.key])
            elif self.key[0:2] == 'tf' and form[0:2] == 'ss':
                aux = ss(self.versions[self.key])
            else:
                aux = self.versions[self.key]

            # Samples system
            if self.key[2] == 'c' and form[2] == 'd':
                aux = sample_system(aux, dt)
            elif self.key[2] == 'd' and form[2] == 'c':
                print('Unavailable conversion from discrete to continuous, please provide a continuous system.')

            # Stores new object
            self.versions[form] = aux
            return aux

    @property
    def trend(self):
        # Wrapper for _trendObj, returns its dictionary
        return self._trendObj.trends

    @property
    def ymk(self):
        return self._trendObj.trends['ymk'][:, -1:]

    @property
    def xmk(self):
        return self._trendObj.trends['xmk'][:, -1:]

    def __init__(self, system, dt=None, input_type="Positional", yss=None, uss=None):
        """
        Defines discrete state-space matrices, assuming matrix D as null.

        Parameters
        ----------
        system : Control transfer function or state-space object.
            Initial values for states.

        dt : float
            Continuous systems will be linearized using this sampling time.

        input_type : string.
            "Positional" or "Incremental".

        yss : list
            Output column vector.

        uss : list
            Input line vector.
        """

        if not issys(system):
            raise TypeError("The first input must be a transfer function or a state-space object")

        self._input_type = input_type

        # A Trend object will be created by self.initialConditions
        self._trendObj = None

        # Stores current model
        self.library = self.Library(system)

        # Uses the system's sampling time, or the sampling time provided
        self.dt = dt or system.dt or 1

        # Discrete state-space using sampling time dt
        system = self.library.get_as('ssd', self.dt)
        # The state-space matrix D is assumed null
        self.A = array(system.A, ndmin=2, dtype=complex)
        self.B = array(system.B, ndmin=2, dtype=complex)
        self.C = array(system.C, ndmin=2, dtype=complex)

        if 0 == (sum(sum(abs(imag(self.A)))) + sum(sum(abs(imag(self.B)))) + sum(sum(abs(imag(self.C))))):
            self.A = real(self.A)
            self.B = real(self.B)
            self.C = real(self.C)

        self._labels = {"inputs": ['$u_{' + str(i) + '}$' for i in range(1, self.nu + 1)],
                        "outputs": ['$y_{' + str(i) + '}$' for i in range(1, self.ny + 1)],
                        "time": 'Time units'}

        # Steady state values
        if yss is None:
            self.yss = zeros((self.C.shape[0], 1))
            self.uss = zeros((self.B.shape[1], 1))
        else:
            self.yss = array(yss, ndmin=2).transpose()
            self.uss = array(uss, ndmin=2).transpose()

    def __repr__(self):
        return 'State-space model object (ny={}, nu={})'.format(self.ny, self.nu)

    @property
    def poles(self):
        poles = sort(list(eig(self.A)[0][:]))[::-1]
        return array(poles, ndmin=2).transpose()

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, label):
        # TODO: Improve
        label["inputs"] = label["inputs"] or self.labels["inputs"]
        label["outputs"] = label["outputs"] or self.labels["outputs"]
        label["time"] = label["time"] or self.labels["time"]

        if len(label["inputs"]) != self.nu:
            raise ValueError("Incorrect number of inputs")
        elif len(label["outputs"]) != self.ny:
            raise ValueError("Incorrect number of outputs")

        self._labels = {"inputs": label["inputs"],
                        "outputs": label["outputs"],
                        "time": label["time"]}

    @property
    def input_type(self):
        return self._input_type

    @xmk.setter
    def xmk(self, xmk):
        ymk = dot(self.C, xmk)
        self._trendObj.historyData(xmk=xmk, ymk=ymk)

    @property
    def nx(self):
        return self.A.shape[0]

    @property
    def nu(self):
        return self.B.shape[1]

    @property
    def ny(self):
        return self.C.shape[0]

    def pole(self):
        """
        Stores poles at the attribute system_poles
        """
        self.unstable = False
        self.system_poles: list = zeros((self.ny, self.nu)).tolist()
        self.integrating = False
        self.__repeating_integrating_poles = False
        g = self.library.get_as('tfc')
        for i in range(0, self.ny):
            for j in range(0, self.nu):
                p = g[i, j].pole()
                if any(real(p) > 0):
                    self.unstable = True
                elif any(real(p) == 0):
                    self.integrating = True
                    if sum(real(p) == 0) > 1:
                        self.__repeating_integrating_poles = True
                self.system_poles[i][j] = p
        return self.system_poles

    def incrementalModel(self):
        """
        Method to convert the state space model to the incremental form according to ....
        """
        self._input_type = "Incremental"
        if self._trendObj is not None:
            self._trendObj = None
            warn("Changing model after simulation, history data has been erased")

        A = array(self.A)
        B = array(self.B)
        C = array(self.C)
        A0 = vstack((A, zeros((self.nu, self.nx))))

        self.B = vstack((B, diag(ones(self.nu))))
        self.A = hstack((A0, self.B))
        self.C = hstack((C, zeros((self.ny, self.nu))))

    def odloakModel(self):
        """
        Method to convert the state space model to the appropriate Odloak formulation.
        """

        # Updating configuration
        self._input_type = "Incremental"
        self.__is_odloak = True

        # Reseting _trendObj to fit the current number of states
        if self._trendObj is not None:
            self._trendObj = None
            warn("Changing model after simulation, history data has been erased")

        def sortpoles(r, d):
            pos = r.argsort()
            r = r[pos]
            d = d[pos]
            # For each pole
            for ri in range(r.size - 1):
                # If a complex pole is found
                if abs(imag(r[ri])) > 10 ** (-10):
                    # If it is an unsorted pair
                    if abs(imag(r[ri])) == abs(imag(r[ri + 1])) and imag(r[ri]) < imag(r[ri + 1]):
                        # sort it so that imag(p1) > imag(p2)
                        aux_sort = r[ri]
                        r[ri] = r[ri + 1]
                        r[ri + 1] = aux_sort

                        aux_sort = d[ri]
                        d[ri] = d[ri + 1]
                        d[ri + 1] = aux_sort
            return r, d

        self.pole()
        lenpole = []
        for i in range(0, self.ny):
            for j in range(0, self.nu):
                lenpole = lenpole + [len(self.system_poles[i][j])]

        hden = deepcopy(self.library.get_as('tfc').den)

        # Integrator (the convolution operator leads to undesirable approximations)
        for i in range(0, self.ny):
            for j in range(0, self.nu):
                hden[i][j] = polymul(hden[i][j], [1, 0])

        h = tf(self.library.get_as('tfc').num, hden)
        num = h.num
        den = h.den

        # Static Gain Matrix
        self.D0 = self.library.get_as('tfc').dcgain()
        if not isinstance(self.D0, type(array([]))):
            self.D0 = array(self.D0, ndmin=2, dtype=complex)

        self.Nst = array([], ndmin=2, dtype=complex)
        self.Psist = array([], ndmin=2, dtype=complex)

        if self.unstable:
            self.Ddst = []
            self.Fst = []  # Containing every stable pole

            self.Psiun = array([], ndmin=2, dtype=complex)
            self.Nun = array([], ndmin=2, dtype=complex)
            self.Ddun = []
            self.Fun = []

            self.nst = 0  # Number of stable poles
            self.nun = 0  # Number of unstable poles

            ddst = []  # diagonal of Ddst
            fst = []  # diagonal of Fst

            ddun = []  # diagonal of Ddun matrix
            fun = []  # diagonal of Fun matrix

            for i in range(self.ny):
                for j in range(self.nu):
                    if not (num[i][j] == 0).all():
                        terms = residue(num[i][j], den[i][j], tol=0.0001, rtype='avg')  # Error at 1e-15
                        resid = terms[0]
                        pole = terms[1]
                        pole, resid = sortpoles(pole, resid)

                        for k in range(resid.shape[0]):
                            if pole[k] < 0:
                                # Appending to diagonals
                                ddst.append(resid[k])
                                # Diagonal of the Fst matrix
                                fst.append(exp(pole[k] * self.dt))
                                # Updating Psist
                                if self.Psist.size:
                                    self.Psist = hstack((self.Psist, zeros((self.ny, 1))))
                                else:
                                    self.Psist = zeros((self.ny, 1))
                                self.Psist[i][-1] = 1

                            elif pole[k] > 0:
                                # Appending to diagonals
                                ddun.append(resid[k])
                                fun.append(exp(pole[k] * self.dt))
                                # Updating Psiun
                                if self.Psiun.size:
                                    self.Psiun = hstack((self.Psiun, zeros((self.ny, 1))))
                                else:
                                    self.Psiun = zeros((self.ny, 1))
                                self.Psiun[i][-1] = 1

                        # Counting the number of stable poles for the current transfer-function
                        stable = sum(pole < 0)
                        # Updating Nst
                        if stable:
                            auxst = hstack(
                                (zeros((stable, j)), ones((stable, 1)), zeros((stable, self.nu - j - 1))
                                 ))
                            self.Nst = vstack((self.Nst, auxst)) if self.Nst.size else auxst

                        # Counting the number of unstable poles for the current transfer-function
                        unstable = sum(pole > 0)
                        # Updating Nun
                        if unstable:
                            auxun = hstack(
                                (zeros((unstable, j)), ones((unstable, 1)), zeros((unstable, self.nu - j - 1))
                                 ))
                            self.Nun = vstack((self.Nun, auxun)) if self.Nun.size else auxun

            self.Ddst = diag(ddst)
            self.Fst = diag(fst)

            self.Ddun = diag(ddun)
            self.Fun = diag(fun)
            self.Bun = multi_dot((self.Ddun, self.Fun, self.Nun))

            self.nst = self.Fst.shape[0]
            self.nun = self.Fun.shape[0]

            self.A = vstack((
                hstack((identity(self.ny), zeros((self.ny, self.nst)), zeros((self.ny, self.nun))
                        )),
                hstack((zeros((self.nst, self.ny)), self.Fst, zeros((self.nst, self.nun))
                        )),
                hstack((zeros((self.nun, self.ny)), zeros((self.nun, self.nst)), self.Fun
                        )),
            ))
            if self.nst:
                self.Bst = multi_dot((self.Ddst, self.Fst, self.Nst))
                self.B = vstack((self.D0, self.Bst, self.Bun))
                self.C = hstack((identity(self.ny), self.Psist, self.Psiun))
            else:
                # Only unstable poles
                self.B = vstack((self.D0, self.Bun))
                self.C = hstack((identity(self.ny), self.Psiun))

        elif self.integrating:
            if self.__repeating_integrating_poles:
                raise NotImplementedError(
                    'There is no Odloak implementation for repeating integrating poles or sustained oscillation.')

            # Integrating component
            self.Bi = zeros((self.ny, self.nu), dtype=complex)  # aux 1xnu

            self.nun = 0
            self.nast = max(lenpole)
            self.nst = self.nast * self.nu * self.ny
            self.Fst = zeros((self.nst, self.nst), dtype=complex)
            self.Ddst = zeros((self.nst, self.nst), dtype=complex)
            ddaux = []
            ddst = []
            fst = []

            for i in range(self.ny):
                for j in range(self.nu):
                    if not (num[i][j] == 0).all():
                        terms = residue(num[i][j], den[i][j], tol=0.0001, rtype='avg')  # Error at 1e-15
                        resid = terms[0]
                        pole = terms[1]

                        # Handling Integrating poles
                        if isinf(self.D0[i][j]):
                            # COUNT ni
                            # concat auxbi for each ny

                            # Natural integrator (second null pole from residue)
                            self.Bi[i][j] = resid[1]

                            self.D0[i][j] = resid[0]
                            if pole.shape[0] == 2:
                                # Pure integrator: use d from the artificial pole (first null pole from residue)
                                self.D0[i][j] = resid[0]

                        pole, resid = sortpoles(pole, resid)

                        # Collects gains for stable poles, padding with 0
                        ds = [ds for num, ds in enumerate(resid) if pole[num] < 0]
                        ddaux = ddaux + ds + [0] * (self.nast - len(ds))

                        # logical_xor((array([1., 0., 0.])==0),array([ False,  False, True])).any()
                        # if (pole == 0).all():
                        #     # Pure integrator
                        #     self.f = zeros(self.nast)
                        # elif sum(pole == 0) == 1:  # Only artificial and stable
                        #     # Stable
                        #     if zeros(self.nast - pole.shape[0] + 1).size:
                        #         self.f = hstack((exp(pole[0:pole.size - 1] * self.dt),
                        #                          zeros(self.nast - pole.shape[0] + 1)
                        #                          ))
                        #     else:
                        #         self.f = exp(pole[0:pole.size - 1] * self.dt)
                        # else:
                        #     # Stable and integrator
                        #     if self.nast - pole.shape[0] > 0:
                        #         self.f = hstack((exp(pole[0:r.size - 2] * self.dt), 0,
                        #                          ones(self.nast - pole.shape[0])
                        #                          ))
                        #     else:
                        #         self.f = hstack((exp(pole[0:pole.size - 2] * self.dt), 0))
                        for k in range(resid.shape[0]):
                            if pole[k] < 0:
                                # Appending to diagonals
                                ddst.append(resid[k])
                                # Diagonal of the Fst matrix
                                fst.append(exp(pole[k] * self.dt))
                                # Updating Psist
                                if self.Psist.size:
                                    self.Psist = hstack((self.Psist, zeros((self.ny, 1))))
                                else:
                                    self.Psist = zeros((self.ny, 1))
                                self.Psist[i][-1] = 1

                        # Counting the number of stable poles for the current transfer-function
                        stable = sum(pole < 0)
                        # Updating Nst
                        if stable:
                            auxst = hstack(
                                (zeros((stable, j)), ones((stable, 1)), zeros((stable, self.nu - j - 1))
                                 ))
                            self.Nst = vstack((self.Nst, auxst)) if self.Nst.size else auxst

            self.Bs = self.D0 + self.dt * self.Bi
            self.Ddst = diag(ddst)
            self.Fst = diag(fst)
            self.nst = self.Fst.shape[0]

            if self.nst:
                self.Bst = multi_dot((self.Ddst, self.Fst, self.Nst))
                # create Ini
                self.A = vstack((
                    hstack((identity(self.ny), zeros((self.ny, self.nst)), self.dt * identity(self.ny)
                            )),
                    hstack((zeros((self.nst, self.ny)), self.Fst, zeros((self.nst, self.ny))
                            )),
                    hstack((zeros((self.ny, self.ny + self.nst)), identity(self.ny)
                            ))
                ))
                self.B = vstack((self.Bs, self.Bst, self.Bi))
                self.C = hstack((identity(self.ny), self.Psist, zeros((self.ny, self.ny))
                                 ))
            else:
                self.A = vstack((
                    hstack((identity(self.ny), self.dt * identity(self.ny))),
                    hstack((zeros((self.ny, self.ny)), identity(self.ny)))
                ))
                self.B = vstack((self.Bs, self.Bi))
                self.C = hstack((identity(self.ny), zeros((self.ny, self.ny))
                                 ))


        else:
            self.Ddst = []
            self.Fst = []  # Containing every stable pole

            self.nst = 0  # Number of stable poles
            self.nun = 0  # Number of unstable poles

            ddst = []  # diagonal of Ddst
            fst = []  # diagonal of Fst

            for i in range(self.ny):
                for j in range(self.nu):
                    if not (num[i][j] == 0).all():
                        terms = residue(num[i][j], den[i][j], tol=0.0001, rtype='avg')  # Error at 1e-15
                        resid = terms[0]
                        pole = terms[1]
                        pole, resid = sortpoles(pole, resid)

                        for k in range(resid.shape[0]):
                            if pole[k] < 0:
                                # Appending to diagonals
                                ddst.append(resid[k])
                                # Diagonal of the Fst matrix
                                fst.append(exp(pole[k] * self.dt))
                                # Updating Psist
                                if self.Psist.size:
                                    self.Psist = hstack((self.Psist, zeros((self.ny, 1))))
                                else:
                                    self.Psist = zeros((self.ny, 1))
                                self.Psist[i][-1] = 1

                        # Counting the number of stable poles for the current transfer-function
                        stable = sum(pole < 0)
                        # Updating Nst
                        if stable:
                            auxst = hstack(
                                (zeros((stable, j)), ones((stable, 1)), zeros((stable, self.nu - j - 1))
                                 ))
                            self.Nst = vstack((self.Nst, auxst)) if self.Nst.size else auxst

            self.Ddst = diag(ddst)
            self.Fst = diag(fst)

            self.nst = self.Fst.shape[0]

            self.Bst = multi_dot([self.Ddst, self.Fst, self.Nst])

            self.A = vstack((
                hstack((identity(self.ny), zeros((self.ny, self.nst))
                        )),
                hstack((zeros((self.nst, self.ny)), self.Fst))
            ))
            self.B = vstack((self.D0,
                             self.Bst))
            self.C = hstack((identity(self.ny), self.Psist))

        # Adjusting to real matrices
        self.toreal()

        if 0 == (sum(sum(abs(imag(self.A)))) + sum(sum(abs(imag(self.B)))) + sum(sum(abs(imag(self.C))))):
            self.A = real(self.A)
            self.B = real(self.B)
            self.C = real(self.C)
        self.Fst = self.A[self.ny: self.ny + self.nst, self.ny: self.ny + self.nst]
        self.Psist = self.C[:, self.ny: self.ny + self.nst]

    def toreal(self):
        # See: SANTORO, Bruno Faccini. Controle preditivo de horizonte infinito para sistemas
        # integradores e com tempo morto [doi:10.11606/D.3.2011.tde-26042011-150304]. São Paulo :
        # Escola Politécnica, Universidade de São Paulo, 2011. Dissertação de Mestrado em
        # Engenharia Química. [acesso 2021-04-06].
        for i in range(self.nst - 1):
            # If the current and previous poles are complex, adjust
            if abs(imag(self.Fst[i, i])) > 1e-10 and abs(imag(self.Fst[i, i])) > 1e-10:
                real_f = real(self.Fst[i, i])
                imag_f = imag(self.Fst[i, i])

                real_imag = vstack((hstack((real_f, imag_f)),
                                    hstack((-imag_f, real_f))
                                    ))
                self.Fst[i: i + 2, i: i + 2] = real_imag

                real_dd = real(self.Ddst[i, i])
                imag_dd = imag(self.Ddst[i, i])
                self.Ddst[i: i + 2, i: i + 2] = vstack((hstack((real_dd - imag_dd, real_dd + imag_dd)),
                                                        hstack((-real_dd - imag_dd, real_dd - imag_dd))
                                                        ))
                self.Psist[:, i + 1] = 0
        for i in range(self.nun - 1):
            # If the current and previous poles are complex, adjust
            if abs(imag(self.Fun[i, i])) > 1e-10 and abs(imag(self.Fun[i, i])) > 1e-10:
                real_f = real(self.Fun[i, i])
                imag_f = imag(self.Fun[i, i])

                real_imag = vstack((hstack((real_f, imag_f)),
                                    hstack((-imag_f, real_f))
                                    ))
                self.Fun[i: i + 2, i: i + 2] = real_imag

                real_dd = real(self.Ddun[i, i])
                imag_dd = imag(self.Ddun[i, i])
                self.Ddun[i: i + 2, i: i + 2] = vstack((hstack((real_dd - imag_dd, real_dd + imag_dd)),
                                                        hstack((-real_dd - imag_dd, real_dd - imag_dd))
                                                        ))
                self.Psiun[:, i + 1] = 0
        if self.nst:
            self.Ddst = real(self.Ddst)
            self.Fst = real(self.Fst)
            self.Bst = multi_dot([self.Ddst, self.Fst, self.Nst])

        if self.unstable:
            self.Ddun = real(self.Ddun)
            self.Fun = real(self.Fun)
            self.Bun = multi_dot((self.Ddun, self.Fun, self.Nun))

            i = 0
            fun = diag(self.Fun)
            for j in range(self.nun - 1):
                if fun[i] == fun[i + 1] and all(self.Psiun[:, i] == self.Psiun[:, i + 1]):
                    if not (any((self.Bun[i, :] != 0) == (self.Bun[i + 1, :] != 0))):
                        self.Bun[i, :] = self.Bun[i, :] + self.Bun[i + 1, :]
                        self.Bun = delete(self.Bun, (i + 1), axis=0)
                        self.Psiun = delete(self.Psiun, (i + 1), axis=1)
                        fun = delete(fun, (i + 1), axis=0)
                        i = i - 1
                i = i + 1
            self.Fun = diag(fun)
            self.nun = self.Fun.shape[0]

            self.A = vstack((
                hstack((identity(self.ny), zeros((self.ny, self.nst)), zeros((self.ny, self.nun))
                        )),
                hstack((zeros((self.nst, self.ny)), self.Fst, zeros((self.nst, self.nun))
                        )),
                hstack((zeros((self.nun, self.ny)), zeros((self.nun, self.nst)), self.Fun
                        )),
            ))
            if self.nst:
                self.B = vstack((self.D0, self.Bst, self.Bun))
                self.C = hstack((identity(self.ny), self.Psist, self.Psiun))
            else:
                # Only unstable poles
                self.B = vstack((self.D0, self.Bun))
                self.C = hstack((identity(self.ny), self.Psiun))
        elif self.integrating:
            if self.nst:
                self.Bst = multi_dot((self.Ddst, self.Fst, self.Nst))

                self.A = vstack((
                    hstack((identity(self.ny), zeros((self.ny, self.nst)), self.dt * identity(self.ny)
                            )),
                    hstack((zeros((self.nst, self.ny)), self.Fst, zeros((self.nst, self.ny))
                            )),
                    hstack((zeros((self.ny, self.ny + self.nst)), identity(self.ny)
                            ))
                ))
                self.B = vstack((self.Bs, self.Bst, self.Bi))
                self.C = hstack((identity(self.ny), self.Psist, zeros((self.ny, self.ny))
                                 ))
            else:
                self.A = vstack((
                    hstack((identity(self.ny), self.dt * identity(self.ny))),
                    hstack((zeros((self.ny, self.ny)), identity(self.ny)))
                ))
                self.B = vstack((self.Bs, self.Bi))
                self.C = hstack((identity(self.ny), zeros((self.ny, self.ny))
                                 ))
        else:
            self.A = vstack((
                hstack((identity(self.ny), zeros((self.ny, self.nst))
                        )),
                hstack((zeros((self.nst, self.ny)), self.Fst))
            ))
            self.B = vstack((self.D0,
                             self.Bst))
            self.C = hstack((identity(self.ny), self.Psist))

    def initialConditions(self, xmk):
        """
        Sets the initial states for the steady-state.
        If only the outputs are known use the method estimateXk first.

        Parameters
        ----------
        xmk : Numpy array
            Initial values for states.

        """
        xmk = array(xmk, ndmin=2)

        if xmk.shape != (self.nx, 1):
            xmk = xmk.transpose()
            if xmk.shape != (self.nx, 1):
                raise ValueError(f'Inconsistent xmk at InitialCondition shape, expected {self.nx, 1}, got {xmk.shape}.')

        ymk = dot(self.C, xmk)

        self._trendObj = Trend({'xmk': xmk,
                                'ymk': ymk})

    def estimateXk(self, yk, uk):
        """
        Minimum squared error approximation for states.

        Parameters
        ----------
        yk : Numpy array
            Current desired output.

        uk : Numpy array
            Current input applied to the system (provided in accordance to the input_type attribute).
        """
        CA = self.C.dot(self.A)
        CBu = multi_dot([self.C, self.B, uk])
        H = CA.transpose().dot(CA)
        cf = (CBu - yk).transpose().dot(CA)

        # min (1/2)*transpose(x)*H*x + transpose(cf)*x
        return -pinv(H).dot(cf.transpose())

    def update(self, duk):
        """
        Applies an input to the system, storing both states and outputs.

        x(k|k) = A*x(k-1|k) + B*u(k|k)
        y(k|k) = C*x(k|k)

        Parameters
        ----------

        duk : Numpy array
            Input applied to the system (provided in accordance to the input_type attribute).
        """
        xmk = dot(self.A, self.xmk[:, -1:]) + dot(self.B, duk)
        ymk = dot(self.C, xmk)
        # Storing
        self._trendObj.historyData(xmk=xmk, ymk=ymk)
        # Returning outputs
        return ymk

    def simulate(self, du):
        """
        Performs an open-loop simulation.
        The resulting data is stored as trends.

        ----------

        duk : Numpy array
            Inputs applied to the system (provided in accordance to the input_type attribute).
            Dimensions Number of inputs x Number of samples
        """
        for i in range(du.shape[1]):
            self.update(du[:, i].reshape((self.nu, 1)))

    @property
    def matrices(self):
        matrices = {'A': self.A,
                    'B': self.B,
                    'C': self.C,
                    'Fst': self.Fst,
                    'Psi': self.Psist,
                    'Nst': self.Nst,
                    'Ddst': self.Ddst}

        if self.nun:
            matrices.update({"Fun": self.Fun,
                             "Psiun": self.Psiun,
                             "Nun": self.Nun})
        return matrices

    def save(self, file=None, fileformat=None):
        """
        Saves the system's matrices as .txt documents. The txt format was chosen for readability, but values may be trucated.

        ----------

        duk : Numpy array
            Inputs applied to the system (provided in accordance to the input_type attribute).
            Dimensions Number of inputs x Number of samples
        """

        if fileformat is None:
            saveas = 'txt'

        if fileformat == 'txt':
            self.save2txt(file)
        elif fileformat == 'mat':
            self.save2mat(file)
        else:
            warn('Unknown format.')

    def save2txt(self, file):
        for matrix in self.matrices:
            savetxt(file + matrix + '.txt', self.matrices[matrix])

    def save2mat(self, file):
        if file is None:
            file = 'system'
        savemat(file + '.mat', self.matrices)

    def step(self, nsim):
        self.initialConditions(array(zeros((self.nx, 1)), ndmin=2))
        self.update(array([1] * self.nu, ndmin=2).transpose())
        if self._input_type == 'Incremental':
            self.simulate(array([[0] * (nsim - 1)] * self.nu, ndmin=2))
        else:
            self.simulate(array([[1] * (nsim - 1)] * self.nu, ndmin=2))


class IHMPC:
    @property
    def trend(self):
        # Wrapper for _trendObj, returns its dictionary
        return self._trendObj.trends

    class Config:
        def __init__(self):
            # System behaviour
            self._attr = ['unstable', 'integrating']

            # Obligatory for the user (or ClosedLoop)
            self._attr.append('_intialConditions')

            # QP definition
            self._attr = self._attr + ['_static', 'zone', 'input_target', 'eco', 'eco_tracking', 'eco_gradmin']

            # Create attributes as False
            for attr in self._attr:
                setattr(self, attr, False)

        def ready(self):
            isready = self._intialConditions
            if not isready:
                warn("Please run IHMPC.intialConditions.")
            return isready

    def __init__(self, model, m, qy, r, sy=1e5, sun=None, si=None, zone=False, optimizer="default"):
        """
        IHMPC class - defining a stable model predictive controller

        :param model:       class model
        :param m (int):     control horizon
        :param qy (list):   weights of outputs

        [OPTIONAL]
        :param r (list):    suppression move elements
        :param sy (list):   weights of slacks related to outputs
        :param zone (bool): whether the controller uses zones for the outputs

        """

        # Default configurations
        self.config = self.Config()

        # Defining solver:
        self.solvers = {"quadprog": self._solver_quadprog,
                        "cvxopt": self._solver_cvxopt}

        if optimizer == "default":
            optimizer = available_optimizers[0]
        elif optimizer not in available_optimizers:
            errmessage = "\nThe selected optimizer is not available. \nSwitching to {}".format(available_optimizers[0]) \
                         + "\nThe currently available optimizers are: " + ", ".join(available_optimizers)
            warn(errmessage)
            optimizer = available_optimizers[0]

        self.solver = self.solvers[optimizer]

        if zone:
            self.config.zone = True
            self._solve = self._solve_zone
        else:
            self._solve = self._solve_tracking

        # Default optimization options tuple (for cvxopt)
        # self.qp_opt = {'maxiters': 250, 'abstol': 1e-12, 'reltol': 1e-12, 'feastol': 4.5e-12, 'refinement': 0,
        #                'show_progress': False}
        #self.qp_opt = {'maxiters': 250, 'abstol': 1e-16, 'reltol': 1e-16, 'feastol': 4.5e-14, 'refinement': 0,
        #               'show_progress': False}

        self.qp_opt = {'maxiters': 250, 'abstol': 1e-12, 'reltol': 1e-12, 'feastol':1e-13, 'refinement': 0,
                       'show_progress': False}
        #self.qp_opt = {'maxiters': 250,
        #               'show_progress': False}
        # self.qp_opt = {'feastol': 1e-10, 'show_progress': False}

        # Properties from model
        model.odloakModel()
        self.model = model
        self.ny = self.model.ny  # output variables of the system
        self.nu = self.model.nu  # input variables of the system
        self.nx = self.model.nx  # state variables of the system
        self.nd = self.model.nst  # states related to the stable pole of the system
        self.nun = self.model.nun  # states related to the unstable pole of the system
        self.sun = sun
        self.si = si
        self.config.unstable = self.model.unstable
        self.config.integrating = self.model.integrating

        # Tuning
        self.m = m  # control horizon

        # TODO: use def?
        # output weights
        if not isinstance(qy, list):
            self.__qy = [qy] * self.ny
        elif len(qy) == self.ny:
            self.__qy = qy
        else:
            raise ValueError('len(qy) is incompatible with the number of outputs.')

        # move weights
        if not isinstance(r, list):
            self.__r = [r] * self.nu
        elif len(r) == self.nu:
            self.__r = r
        else:
            raise ValueError('len(r) is incompatible with the number of inputs.')

        # stable slacks weigths
        if isinstance(sy, list):
            if len(sy) == self.ny:
                self.__sy = sy
            else:
                raise ValueError('len(sy) is incompatible with the number of outputs.')
        else:
            self.__sy = [sy * q for q in self.__qy]  # weights of output slacks

        # Auxiliary matrices
        self.findxs = hstack((identity(self.ny), zeros((self.ny, self.nd))))
        self.findxst = hstack((zeros((self.nd, self.ny)), identity(self.nd)))

        self.xs = lambda: self.findxs.dot(self.model.xmk)
        self.xst = lambda: self.findxst.dot(self.model.xmk)
        self.xun = lambda: 0
        self.xi = lambda: 0

        if self.config.unstable:
            # unstable slacks weights
            if sun is not None:
                if isinstance(sun, list):
                    if len(sun) == self.nun:
                        self.__sun = sun
                    else:
                        raise ValueError('len(sun) is incompatible with the number of unstable poles.')
                else:
                    self.__sun = [sun] * self.nun
            else:
                self.__sun = None
            self.findxs = hstack((self.findxs, zeros((self.ny, self.nun))))
            self.findxst = hstack((self.findxst, zeros((self.nd, self.nun))))
            self.findxun = hstack((zeros((self.nun, self.ny)), zeros((self.nun, self.nd)), identity(self.nun)))
            self.xun = lambda: self.findxun.dot(self.model.xmk)

            self.xtilde = lambda: dot(self.Nbar, self.model.xmk) - dot(self.Itilde_ny, self.xs()) \
                                  - self.Gammaun.dot(matrix_power(self.model.Fun, self.m)).dot(self.xun())
        elif self.config.integrating:
            # integrating slacks weights
            if si is not None:
                if isinstance(si, list):
                    if len(si) == self.ny:
                        self.__si = si
                    else:
                        raise ValueError('len(si) is incompatible with the number of outputs poles.')
                else:
                    self.__si = [si] * self.ny
            else:
                self.__si = None
            self.findxs = hstack((self.findxs, zeros((self.ny, self.ny))))
            self.findxst = hstack((self.findxst, zeros((self.nd, self.ny))))
            self.findxi = hstack((zeros((self.ny, self.ny)), zeros((self.ny, self.nd)), identity(self.ny)))
            self.xi = lambda: self.findxi.dot(self.model.xmk)
            self.xtilde = lambda: dot(self.Nbar, self.model.xmk) - dot(self.Itilde_ny, self.xs()) \
                                  - self.Itilde_ny.dot(self.m * self.model.dt * self.xi()) - self.Dtilde.dot(
                self.xi())  # TODO: simplify
        else:
            self.xtilde = lambda: dot(self.Nbar, self.model.xmk) - dot(self.Itilde_ny, self.xs())

        # Target at inputs
        self._save_input_target = lambda input_target: 0

        # Economic target (approximated gradient)
        self._save_eco = lambda duk: 0
        self.eco_func = lambda: 0
        self.predicted_eco_func = lambda duk: 0
        self.eco_func_gradient = lambda: 0
        self.update_gradmin = lambda duk: 0

        # Unstable system
        self._save_uns = lambda duk: 0

        # Integrating system
        self._save_i = lambda duk: 0

        # Default filter
        self.Kalman()

        # Creating matrices
        self._set_matrices()

    def _set_matrices(self):
        """
        Matrices that do not depend on states

        """
        Bs = self.model.B[0:self.ny, :]
        self.Bs_tilde = tile(Bs, (1, self.m))
        if self.config.integrating:
            self.Bst = self.model.B[self.ny: self.nx - self.ny, :]
            self.Bi = self.model.B[self.nx - self.ny: self.nx, :]
        else:
            self.Bst = self.model.B[self.ny: self.nx - self.nun, :]

        self.Qy = diag(self.__qy)  # dimension(ny x ny)
        self.Sy = diag(self.__sy)  # dimension (ny x ny)
        self.R = diag(self.__r)  # dimension (nu x nu)

        self.Qybar = diag(tile(self.__qy, [1, self.m])[0])  # dimension(ny * m x  ny * m)
        self.Rbar = diag(tile(self.__r, [1, self.m])[0])  # dimension(nu * m x nu * m)

        self.Itilde_ny = tile(identity(self.ny), (self.m, 1))
        self.Itilde_nu = tile(identity(self.nu), (self.m, 1))

        # Discrete Lyapunov Equation
        Qaux = multi_dot(
            [self.model.Fst.transpose(), self.model.Psist.transpose(), self.Qy, self.model.Psist, self.model.Fst])
        self.Qy_tilde = solve_discrete_lyapunov(self.model.Fst, Qaux)

        #
        self.Nbar = dot(self.model.C, matrix_power(self.model.A, 1))
        self.Thetabar = hstack((dot(self.model.C, self.model.B), zeros((self.ny, self.nu * self.m - self.nu))))
        self.Lambda = self.Bst
        self.Mtilde = hstack((identity(self.nu), zeros((self.nu, self.nu * self.m - self.nu))))

        for j in range(1, self.m):
            self.Nbar = vstack((self.Nbar, dot(self.model.C, matrix_power(self.model.A, j + 1))))
            self.Thetabar = vstack((self.Thetabar,
                                    hstack((multi_dot([self.model.C,
                                                       matrix_power(self.model.A, j),
                                                       self.model.B]),
                                            self.Thetabar[(j - 1) * self.ny: (j - 1) * self.ny + self.ny,
                                            0: self.m * self.nu - self.nu]
                                            ))
                                    ))

        for j in range(1, self.m):
            self.Lambda = hstack((matrix_power(self.model.Fst, j).dot(self.Bst), self.Lambda))
            self.Mtilde = vstack((self.Mtilde,
                                  hstack((identity(self.nu),
                                          self.Mtilde[((j - 1) * self.nu): ((j - 1) * self.nu + self.nu),
                                          0: self.m * self.nu - self.nu]
                                          ))
                                  ))

        # Inequality matrix Aineq*x <= Bineq
        self.Aineq = vstack((self.Mtilde,
                             - self.Mtilde,
                             identity(self.nu * self.m),
                             -identity(self.nu * self.m)
                             ))
        # TODO: move to _set
        if self.config.integrating:
            self.Bi_tilde = tile(self.Bi, (1, self.m))
            self.Dtilde = zeros((self.ny, self.ny))
            # aux = self.Bi
            aux = zeros((self.ny, self.nu))
            for j in range(1, self.m):
                # Dtilde = [(1−m)dt*Iny
                #           (2−m)dt*Iny
                #            ···
                #            0ny]
                self.Dtilde = vstack((identity(self.ny) * (-j * self.model.dt), self.Dtilde))
                aux = hstack((j * self.model.dt * self.Bi, aux))
            self.Lambdas = aux + self.Bs_tilde

            self.Theta_tilde = self.Thetabar - self.Itilde_ny.dot(self.Lambdas) - self.Dtilde.dot(
                self.Bi_tilde)

            # Hessian matrix
            self.H = multi_dot([self.Theta_tilde.transpose(), self.Qybar, self.Theta_tilde]) + \
                     self.Rbar + \
                     multi_dot([self.Lambda.transpose(), self.Qy_tilde, self.Lambda]) + \
                     multi_dot([self.Lambdas.transpose(), self.Sy, self.Lambdas])
        else:
            if self.config.unstable:
                Bun = self.model.B[self.nx - self.nun:, :]

                self.Lambdaun = Bun

                self.Gammaun = self.model.Psiun.dot(matrix_power(self.model.Fun, 1 - self.m))

                for j in range(1, self.m):
                    self.Lambdaun = hstack((dot(matrix_power(self.model.Fun, j), Bun), self.Lambdaun))
                    self.Gammaun = vstack(
                        (self.Gammaun, self.model.Psiun.dot(matrix_power(self.model.Fun, j + 1 - self.m))))

                self.Theta_tilde = self.Thetabar - self.Gammaun.dot(self.Lambdaun) - dot(self.Itilde_ny, self.Bs_tilde)
            else:
                self.Theta_tilde = self.Thetabar - dot(self.Itilde_ny, self.Bs_tilde)

            # Hessian matrix
            self.H = multi_dot([self.Theta_tilde.transpose(), self.Qybar, self.Theta_tilde]) + \
                     self.Rbar + \
                     multi_dot([self.Lambda.transpose(), self.Qy_tilde, self.Lambda]) + \
                     multi_dot([self.Bs_tilde.transpose(), self.Sy, self.Bs_tilde])

        # Resolve stability first
        if self.config.integrating:
            self._set_integrating()
        elif self.config.unstable:
            self._set_unstable()
        # Adjust Hessian for zone
        if self.config.zone:
            self._set_zone_matrices()

    def _set_zone_matrices(self):
        self._solve = self._solve_zone

        if self.config.integrating:
            H12 = - dot(self.Lambdas.transpose(), self.Sy)
            H21 = H12.transpose()
            H22 = self.Sy
            self.H = vstack((hstack((self.H, H12)),
                             hstack((H21, H22))
                             ))
        else:
            H12 = - dot(self.Bs_tilde.transpose(), self.Sy)
            H21 = H12.transpose()
            H22 = self.Sy
            self.H = vstack((hstack((self.H, H12)),
                             hstack((H21, H22))
                             ))
        self.Aineq = hstack((self.Aineq, zeros((self.Aineq.shape[0], self.ny), dtype=float)))
        self.Aineq = vstack((self.Aineq,
                             hstack((zeros((self.ny, self.nu * self.m)), identity(self.ny))),
                             hstack((zeros((self.ny, self.nu * self.m)), - identity(self.ny)))
                             ))

    def _set_integrating(self, si=10.0):
        if not self.__si:
            self.Si = si * identity(self.ny)
        else:
            self.Si = diag(self.__si)

        self._save_i = lambda duuk: self._trendObj.historyData(ski=self.xi() + dot(self.Bi_tilde, duuk))

        self.H = self.H + multi_dot([self.Bi_tilde.transpose(), self.Si, self.Bi_tilde])

    def _set_unstable(self, sun=10.0):
        if not self.__sun:
            self.sun = sun
            self.__sun = self.sun * ones((self.nun,))
        self.Sun = diag(self.__sun)

        self._save_uns = lambda duuk: self._trendObj.historyData(
            skun=multi_dot([matrix_power(self.model.Fun, self.m), self.findxun, self.model.xmk]) + dot(self.Lambdaun,
                                                                                                       duuk))

        self.__sun = self.sun * ones((self.nun,))  # weights of unstable poles slacks

        self.H = self.H + multi_dot([self.Lambdaun.transpose(), self.Sun, self.Lambdaun])

    @property
    def c(self):
        clist = [c() for c in self._dict_c.values()]
        c = sum(clist)
        return c

    @property
    def cf(self):
        # Add components related to the inputs
        cf1 = 0
        for component in self._dict_cf1.values():
            cf1 = cf1 + component()
        # Components related to other decision variables
        cf2 = 0
        for component in self._dict_cf2.values():
            cf2 = cf2 + component()
        # Stack if needed
        cf = self._stack_cf(cf1, cf2)
        return cf

    def _set_ccf(self):
        # Components to include
        self._dict_cf1 = {}
        self._dict_cf2 = {}
        self._dict_c = {}

        # Applicable both for set-point tracking and zone control
        if self.config.unstable:
            cf1_uns = lambda: multi_dot(
                [multi_dot([matrix_power(self.model.Fun, self.m), self.xun()]).transpose(), self.Sun,
                 self.Lambdaun])
            c_uns = lambda: multi_dot([
                multi_dot([matrix_power(self.model.Fun, self.m), self.xun()]).transpose(), self.Sun,
                multi_dot([matrix_power(self.model.Fun, self.m), self.xun()])])

            # Add to dict
            self._dict_cf1['unstable'] = cf1_uns
            self._dict_c['unstable'] = c_uns

        if self.config.zone:
            self._stack_cf = lambda cf1, cf2: hstack((cf1, cf2))

            if self.config.integrating:
                # Stable
                cf_1_stable = lambda: multi_dot([self.xtilde().transpose(), self.Qybar, self.Theta_tilde]) + \
                                      multi_dot([self.xst().transpose(),
                                                 (matrix_power(self.model.Fst, self.m)).transpose(),
                                                 self.Qy_tilde,
                                                 self.Lambda])  # TODO: rename Lambda->Lambda_st & Lambdas->Lambda_s
                c_stable = lambda: multi_dot([self.xtilde().transpose(), self.Qybar, self.xtilde()]) + \
                                   multi_dot([self.xst().transpose(),
                                              matrix_power(self.model.Fst, self.m).transpose(),
                                              self.Qy_tilde,
                                              matrix_power(self.model.Fst, self.m), self.xst()])
                self._dict_cf1['stable'] = cf_1_stable
                self._dict_c['stable'] = c_stable

                # Integrating
                cf_1_base = lambda: multi_dot(
                    [(self.xs() + self.m * self.model.dt * self.xi()).transpose(),
                     self.Sy,
                     self.Lambdas]) + multi_dot([self.xi().transpose(), self.Si, self.Bi_tilde])

                cf_2_base = lambda: -dot((self.xs() + self.m * self.model.dt * self.xi()).transpose(), self.Sy)

                c_base = lambda: multi_dot(
                    [(self.xs() + self.m * self.model.dt * self.xi()).transpose(),
                     self.Sy,
                     (self.xs() + self.m * self.model.dt * self.xi())]) + \
                                 multi_dot([self.xi().transpose(), self.Si, self.xi()])

            else:  # Stable and unstable models
                cf_1_base = lambda: multi_dot([self.xtilde().transpose(), self.Qybar, self.Theta_tilde]) + \
                                    multi_dot([self.xst().transpose(),
                                               (matrix_power(self.model.Fst, self.m)).transpose(),
                                               self.Qy_tilde,
                                               self.Lambda]) + \
                                    multi_dot([self.xs().transpose(), self.Sy, self.Bs_tilde])
                # lambda cf2 (related to the calculated set-point)
                cf_2_base = lambda: - dot(self.xs().transpose(), self.Sy)

                # lambda c
                c_base = lambda: multi_dot([self.xtilde().transpose(), self.Qybar, self.xtilde()]) + \
                                 multi_dot([self.xst().transpose(),
                                            matrix_power(self.model.Fst, self.m).transpose(),
                                            self.Qy_tilde,
                                            matrix_power(self.model.Fst, self.m), self.xst()]) + \
                                 multi_dot([self.xs().transpose(), self.Sy, self.xs()])

            # Add to dict
            self._dict_cf1['zone'] = cf_1_base
            self._dict_cf2['zone'] = cf_2_base
            self._dict_c['zone'] = c_base

            # Economic
            if self.config.eco:
                if self.config.eco_tracking:
                    cf1_eco = lambda: (self.fk_1() - self._current_esp) * self.P * self.dfdelta_u
                    c_eco = lambda: (self.fk_1() - self._current_esp) * self.P * \
                                    (self.fk_1() - self._current_esp)
                else:  # self.config.eco_gradmin
                    cf1_eco = lambda: multi_dot(
                        [self.eco_func_gradient().transpose(), self.P, self.ddfeco,
                         self.Itilde_nu.transpose()])
                    c_eco = lambda: multi_dot(
                        [self.eco_func_gradient().transpose(), self.P,
                         self.eco_func_gradient()])
                # Add to dict
                self._dict_cf1['economic'] = cf1_eco
                self._dict_c['economic'] = c_eco

            # Input Targets
            if self.config.input_target:
                cf1_input_target = lambda: multi_dot(
                    [(self.uk_1 - self._current_input_target).transpose(), self.Su,
                     self.Mtilde[(self.m - 1) * self.nu:, :]])  # The last incremental control action is assumed null.
                c_input_target = lambda: multi_dot(
                    [(self.uk_1 - self._current_input_target).transpose(), self.Su,
                     self.uk_1 - self._current_input_target])

                # Add to list
                self._dict_cf1['input targets'] = cf1_input_target
                self._dict_c['input targets'] = c_input_target

        else:  # target tracking
            self._stack_cf = lambda cf1, cf2: cf1
            # For every set-point tracking problem
            cf_1_base = lambda: multi_dot([self.xtilde().transpose(), self.Qybar, self.Theta_tilde]) + \
                                multi_dot([self.xst().transpose(),
                                           (matrix_power(self.model.Fst, self.m)).transpose(),
                                           self.Qy_tilde,
                                           self.Lambda])  # TODO: rename Lambda->Lambda_st & Lambdas->Lambda_s
            c_base = lambda: multi_dot([self.xtilde().transpose(), self.Qybar, self.xtilde()]) + \
                             multi_dot([self.xst().transpose(),
                                        matrix_power(self.model.Fst, self.m).transpose(),
                                        self.Qy_tilde,
                                        matrix_power(self.model.Fst, self.m), self.xst()])

            # Add to dict
            self._dict_cf1['set-point tracking'] = cf_1_base
            self._dict_c['set-point tracking'] = c_base

            if self.config.integrating:
                cf1_integrating = lambda: multi_dot(
                    [(self.xs() + self.m * self.model.dt * self.xi() - self._current_ysp).transpose(),
                     self.Sy,
                     self.Lambdas]) + multi_dot([self.xi().transpose(), self.Si, self.Bi_tilde])

                c_integrating = lambda: multi_dot(
                    [(self.xs() + self.m * self.model.dt * self.xi() - self._current_ysp).transpose(),
                     self.Sy,
                     (self.xs() + self.m * self.model.dt * self.xi() - self._current_ysp)]) + \
                                        multi_dot([self.xi().transpose(), self.Si, self.xi()])

                # Add to dict
                self._dict_cf1['integrating'] = cf1_integrating
                self._dict_c['integrating'] = c_integrating
            else:
                # For stable and unstable poles
                cf1_st = lambda: multi_dot([(self.xs() - self._current_ysp).transpose(),
                                            self.Sy,
                                            self.Bs_tilde])

                c_st = lambda: multi_dot([(self.xs() - self._current_ysp).transpose(), self.Sy,
                                          (self.xs() - self._current_ysp)])

                # Add to dict
                self._dict_cf1['stable'] = cf1_st
                self._dict_c['stable'] = c_st

    def set_input_targets(self, qu=None, su=1.0):
        """
        Setting Economic Parameters. Related to term: sum ||u(k+j|k) - u_des||^2_qu + ||u(k+m-1|k) - u_des||^2_Su

        :param qu: (list)
        :param su: (list)
        :return:

        Following: Martins, M. A. F., Yamashita, A. S., Santoro, B. F., & Odloak, D. (2013). Robust model predictive
        control of integrating time delay processes. Journal of Process Control, 23(7), 917–932.
        """
        self.config.input_target = True

        # move weights
        if qu is not None:
            if len(qu) != self.nu:
                raise ValueError('len(qu) is incompatible with the number of inputs.')
            self.__qu = qu
        else:
            self.__qu = [r for r in self.__r]
        if isinstance(su, list):
            self.__su = su
        else:
            self.__su = [su * q for q in self.__qu]

        self.Qubar = diag(tile(self.__qu, [1, self.m])[0])  # dimension(nu * m x nu * m)
        self.Su = diag(self.__su)

        # if not self.config.zone:
        #     self.config.zone = True
        #     warn('Changing objective to zone control.')
        #     self._set_zone_matrices()

        self._save_input_target = lambda input_target: self._trendObj.historyData(input_target=input_target)

        self._H_eco = multi_dot(
            [(self.Mtilde - dot(self.Itilde_nu, self.Mtilde[(self.m - 1) * self.nu:, :])).transpose(),
             self.Qubar,
             (self.Mtilde - dot(self.Itilde_nu, self.Mtilde[(self.m - 1) * self.nu:, :]))]) + \
                      multi_dot([self.Mtilde[(self.m - 1) * self.nu:, :].transpose(),
                                 self.Su,
                                 self.Mtilde[(self.m - 1) * self.nu:, :]])

        self.H[0:self.nu * self.m, 0:self.nu * self.m] = self.H[0:self.nu * self.m, 0:self.nu * self.m] + self._H_eco

    def set_eco_tracking(self, dfdy, dfdu, fss=0.0, P=1.0):
        """
        """
        if dfdy.shape != (1, self.ny):
            raise ValueError('Incorrect dfdy gradient. Expected ' + str((1, self.ny)) + ' got ' + str(dfdy.shape) + '.')
        if dfdu.shape != (1, self.nu):
            raise ValueError('Incorrect dfdu gradient. Expected ' + str((1, self.nu)) + ' got ' + str(dfdu.shape) + '.')

        self.config.eco = True
        self.config.eco_tracking = True

        # Weight
        self.P = P  # scalar

        # Gradient
        self.fss = fss
        self.dfdy = dfdy
        self.dfdu = dfdu

        if self.config.integrating:
            # Gradient in respect to delta u
            self.dfdelta_u = self.dfdu.dot(self.Itilde_nu.transpose()) + self.dfdy.dot(self.Lambdas)
            # Approximation at k-1
            # TODO: Consider using set attr
            self.fk_1 = lambda: self.fss + dfdu.dot(self.uk_1) + dfdy.dot(
                self.xs() + self.m * self.model.dt * self.xi())
        else:
            # Gradient in respect to delta u
            self.dfdelta_u = self.dfdu.dot(self.Itilde_nu.transpose()) + self.dfdy.dot(self.Bs_tilde)
            # Approximation at k-1
            # TODO: Consider using set attr
            self.fk_1 = lambda: self.fss + dfdu.dot(self.uk_1) + dfdy.dot(self.xs())

        # Linear economic performance function
        self.fk = lambda duk: self.fk_1() + self.dfdelta_u.dot(duk)

        # Cost function
        self.H[0:self.nu * self.m, 0:self.nu * self.m] = self.H[0:self.nu * self.m, 0:self.nu * self.m] + \
                                                         self.P * self.dfdelta_u.transpose().dot(self.dfdelta_u)

        self._save_eco = lambda duk: self._trendObj.historyData(feco=self.fk(duk))

    def set_eco_gradmin(self, wy, wu, p):
        """
        Setting Economic Parameters. Related to term: ||dk + G duk||^2_P

        :param wy:
        :param wu:
        :param P:
        :return:

        Following: Alvarez, L. A., & Odloak, D. (2014). Reduction of the QP-MPC cascade structure to a single layer MPC.
        Journal of Process Control, 24(10), 1627–1638.
        """
        self.config.eco = True
        self.config.eco_gradmin = True

        # Input parameters
        self.P = diag(p)
        self._wy = array(wy, ndmin=2)
        self._wu = array(wu, ndmin=2)
        self._wc = dot(self._wy, self.model.B[0:self.ny, :]) + self._wu

        # Trend handling
        self._save_eco = lambda duk: self._trendObj.historyData(feco=self.eco_func(),
                                                                feco_h=self.predicted_eco_func(
                                                                    duk),
                                                                df_eco=self.eco_func_gradient())

        self.eco_func = lambda: (self._wy.dot(self.xs()) + self._wu.dot(self.uk_1) - self._current_esp) ** 2

        # Economic function
        if self.config.integrating:
            self.predicted_eco_func = lambda duk: (multi_dot(
                (self._wy,
                 self.xs() + self.m * self.model.dt * self.xi() + dot(self.Lambdas[:, :(self.m - 1) * self.nu],
                                                                      duk[: self.nu * (self.m - 1)])))
                                                   + dot(self._wu, self.uk_1 +
                                                         dot(self.Mtilde[self.nu * (self.m - 2): self.nu * (self.m - 1),
                                                             : self.nu * (self.m - 1)],
                                                             duk[: self.nu * (self.m - 1)]))
                                                   - self._current_esp) ** 2
        else:
            self.predicted_eco_func = lambda duk: (multi_dot(
                (self._wy, self.xs() + dot(self.Bs_tilde[:, :(self.m - 1) * self.nu], duk[: self.nu * (self.m - 1)])))
                                                   + dot(self._wu, self.uk_1 +
                                                         dot(self.Mtilde[self.nu * (self.m - 2): self.nu * (self.m - 1),
                                                             : self.nu * (self.m - 1)],
                                                             duk[: self.nu * (self.m - 1)]))
                                                   - self._current_esp) ** 2

        # Wrapper lambda, expected to do nothing if config.eco_gradmin is False
        self.update_gradmin = lambda duk: self.update_xs_pred(duk)

        self.eco_func_gradient = lambda: 2 * (
                self._wy.dot(self.xs_pred)
                + self._wu.dot(self.uk_1)
                - self._current_esp) * self._wc.transpose()

        self.ddfeco = 2 * dot(self._wc.transpose(), self._wc)

        if not self.config.zone:
            self.config.zone = True
            self._set_zone_matrices()

        self.H[0:self.nu * self.m, 0:self.nu * self.m] = self.H[0:self.nu * self.m, 0:self.nu * self.m] + \
                                                         multi_dot(
                                                             [self.Itilde_nu, self.ddfeco.transpose(), self.P,
                                                              self.ddfeco,
                                                              self.Itilde_nu.transpose()])

    def update_xs_pred(self, duk):
        self.xs_pred = self.findxs.dot(self.model.A.dot(self.model.xmk) + self.model.B.dot(duk))

    def initialConditions(self, u0, xmk, ysp=None):
        self.config._intialConditions = True

        self._current_ysp = None
        self._current_input_target = None
        self._current_esp = 0

        # Defining the constant and linear terms (c and cf) for the QP
        self._set_ccf()

        # TODO: use QP in initial condition
        if isinstance(u0, list):
            u0 = array(u0, ndmin=2).transpose()

        if isinstance(xmk, list):
            xmk = array(xmk, ndmin=2).transpose()
        self.model.initialConditions(xmk)
        ymk = self.model.ymk

        # if isinstance(ymk, list):
        #     ymk = array(ymk, ndmin=2).transpose()

        # Creating _trendObj handler
        default_trends = {"ymk": array([]).reshape(self.ny, 0),
                          "ysp": array([]).reshape(self.ny, 0),
                          "sky": array([]).reshape(self.ny, 0),
                          "uk": array([]).reshape(self.nu, 0),
                          "duk": array([]).reshape(self.nu, 0),
                          "umax": array([]).reshape(self.nu, 0),
                          "umin": array([]).reshape(self.nu, 0),
                          "fval": array([]).reshape(1, 0)}
        self._trendObj = Trend(default_trends)

        # Adding situational trends
        if self.config.zone:
            self._trendObj.mergeDict({"ymax": array([]).reshape(self.ny, 0),
                                      "ymin": array([]).reshape(self.ny, 0)})
        if self.config.eco:
            self._trendObj.mergeDict({"feco": array([]).reshape(1, 0),
                                      "feco_h": array([]).reshape(1, 0),
                                      "df_eco": array([]).reshape(self.nu, 0)})
        if self.config.integrating:
            self._trendObj.mergeDict({"ski": array([]).reshape(self.ny, 0)})

        if self.config.input_target:
            self._trendObj.mergeDict({"input_target": array([]).reshape(self.nu, 0)})
        if self.nun:
            self._trendObj.mergeDict({"skun": array([]).reshape(self.nun, 0)})

        # Assigning states
        if self.config.eco_gradmin:
            self.xs_pred = self.findxs.dot(self.model.xmk)

        # Storing data
        self.uk_1 = u0
        self._trendObj.historyData(uk=u0, ymk=ymk)
        if not self.config.zone:
            self._trendObj.historyData(ysp=ysp)

    def solve(self, ypk, umin, umax, dumax, **kwargs):
        # using the observer
        self.updateStates(ypk)

        # last implemented u
        self.uk_1 = self.trend["uk"][:, -1:]

        # Including  constraints
        # Upper bound
        UB = dot(self.Itilde_nu, dumax)
        # lower bound
        LB = dot(-self.Itilde_nu, dumax)

        return self._solve(UB, LB, umin, umax, **kwargs)

    def _solve_zone(self, UB, LB, umin, umax, ymin, ymax, input_target, esp=None, **kwargs):
        """
        Evaluation of control moves

        :param xmk (array): states of the system
        :param ysp (list):set-point of outputs
        :param umax(list):
        :param umin (list):
        :param dumax (list):
        :return: duuk, uk

        """

        self._current_input_target = input_target
        self._current_esp = esp

        # Inequalities matrices Aineq*x <= Bineq
        self.Bineq = vstack((dot(self.Itilde_nu, umax - self.uk_1),
                             dot(self.Itilde_nu, self.uk_1 - umin),
                             UB,
                             -LB,
                             ymax,
                             -ymin))

        # Estimating the solution
        if self.trend['duk'].shape[1] == 0:
            self.estimated_solution = zeros((self.nu * self.m + self.ny,))
            self.estimated_solution[self.nu * self.m:] = (ymax + ymin)[:, 0] / 2

        # Solving
        solution = self.solver()

        if (self.Bineq - dot(self.Aineq, solution) < -1e-8).any():
            warn('The inequality constraints were not satisfied.')

        duk = solution[0:self.nu * self.m]

        ysp = solution[self.nu * self.m:]

        # Evaluating objective function
        fval = multi_dot([solution.transpose(), self.H, solution]) + 2 * multi_dot([self.cf, solution]) + self.c

        # Estimating solution at k + 1|k
        self.estimated_solution[0: self.nu * (self.m - 1)] = solution[self.nu: self.nu * self.m, 0]
        self.estimated_solution[self.nu * (self.m - 1): self.nu * self.m] = 0.0
        self.estimated_solution[self.nu * self.m:] = solution[self.nu * self.m:, 0]

        # Updating economic function (for config.eco_gradmin only)
        self.update_gradmin(duk[0:self.nu])

        # Evaluating slacks
        sky = self.xs() + dot(self.Bs_tilde, duk) - ysp  # slacks of the outputs

        # Evaluating uk
        uk = self.uk_1 + duk[0:self.nu]

        # Saving _trendObj data # TODO: restructure
        self._trendObj.historyData(duk=duk[0:self.nu], uk=uk, sky=sky, fval=fval,
                                   umax=umax, umin=umin, ymin=ymin, ymax=ymax, ysp=ysp)

        self._save_input_target(input_target)
        self._save_eco(duk)
        self._save_i(duk)
        self._save_uns(duk)

        # Updating the model
        self.model.update(duk[0:self.nu])
        self._trendObj.historyData(ymk=self.model.ymk)
        return duk[0:self.nu], uk

    def _solve_tracking(self, UB, LB, umin, umax, ysp, input_target=None, esp=None, **kwargs):
        """
        Evaluation of control moves

        :param xmk (array): states of the system
        :param ysp (list):set-point of outputs
        :param umax(list):
        :param umin (list):
        :param dumax (list):
        :return: duuk, uk
        """

        # State dependent matrices
        self._current_ysp = ysp
        self._current_input_target = input_target
        self._current_esp = esp

        # Inequalities matrices Aineq*x <= Bineq
        self.Bineq = vstack((dot(self.Itilde_nu, (umax - self.uk_1)),
                             dot(self.Itilde_nu, (self.uk_1 - umin)),
                             UB,
                             -LB))
        # TODO: estimated solution
        self.estimated_solution = None
        # if self.trend['duk'].shape[1] == 0:
        #     self.estimated_solution = zeros((self.nu * self.m,))

        # Solving
        duk = self.solver()

        # Estimating solution at k + 1|k
        #self.estimated_solution[0: self.nu * (self.m - 1)] = duk[self.nu: self.nu * self.m, 0]
        #self.estimated_solution[self.nu * (self.m - 1): self.nu * self.m] = 0.0

        # Evaluating objective function
        fval = multi_dot([duk.transpose(), self.H, duk]) + 2 * multi_dot([self.cf, duk]) + self.c  # control cost value

        # Updating economic function (for config.eco_gradmin only)
        self.update_gradmin(duk[0:self.nu])

        # Evaluating slacks
        sky = self.xs() + dot(self.Bs_tilde, duk) - ysp  # slacks of the outputs

        # Evaluating uk
        # assumes self.model to not be positional
        uk = self.uk_1 + duk[0:self.nu]

        # Saving _trendObj data
        self._trendObj.historyData(duk=duk[0:self.nu], uk=uk, sky=sky, fval=fval, umax=umax, umin=umin, ysp=ysp)

        self._save_input_target(input_target)
        self._save_eco(duk)
        self._save_i(duk)
        self._save_uns(duk)

        # Updating the model
        ymk = self.model.update(duk[0:self.nu])
        self._trendObj.historyData(ymk=self.model.ymk)
        return duk[0:self.nu], uk

    def updateStates(self, ypk):
        de = ypk - self.model.ymk
        self.model.xmk = self.model.xmk + dot(self.kalman.gain, de)

    def Kalman(self, w=.1e-3, v=.1e-5, niter=100):
        self.kalman = Kalman(self.model.A, self.model.C, w, v, niter)
        self.gain = self.kalman.gain

    def _solver_cvxopt(self):
        with catch_warnings():
            # cvxopt (1.2.3) causes numpy (1.18.1) to throw a FutureWarning
            simplefilter(action='ignore', category=FutureWarning)

            solution = qp(matrix(self.H), matrix(self.cf.transpose()), matrix(self.Aineq), matrix(self.Bineq),
                          initvals=self.estimated_solution, options=self.qp_opt)
        return array(solution['x'])

    def _solver_quadprog(self):
        solution = solve_qp(self.H, - self.cf.flatten(), - self.Aineq.transpose(), - self.Bineq.flatten())
        return array(solution[0], ndmin=2).transpose()


class ClosedLoop:
    class ConfigPlots:
        @property
        def path(self):
            return os.path.abspath(os.path.join(self.folder, self.subfolder))

        def __init__(self):
            self.folder = ''  # Default: same folder as the script from which ClosedLoop is called
            self.subfolder = ''  # Default: No subfolders

            self.filename = {"Inputs": "Uk",
                             "Outputs": "Yk",
                             "Cost Function": "Vk",
                             "Economic Function": "F eco",
                             "Skun": "Skun",
                             "Ski": "Ski"}

            # self.legends = {"Inputs": ['Input', 'Target', 'Constraints'],
            #                 "Outputs": ['Output', 'Set-point', 'Constraints'],
            #                 }
            self.costLabel = {"Cost Function": '$V_{k}$', "Delta": "$\Delta V_{k}$",
                              "Derivative": r'$\mathdefault{\frac{dV}{dt}}$'}

            # plot **kwargs, here used for drawstyle and color
            self.style = {"Inputs": {'drawstyle': 'steps-post', 'color': 'k', 'label': 'Input'},
                          "Input target": {'ls': '--', 'color': 'b', 'label': 'Input Target'},
                          "Input Constraints": {'ls': '--', 'color': 'r', 'label': 'Constraints'},
                          "Outputs": {'label': 'Output'},
                          "Output Targets": {'drawstyle': 'steps-post', 'ls': '--', 'color': 'r', 'label': 'Setpoint'},
                          "Output Constraints": {'drawstyle': 'steps-post', 'ls': '--', 'color': 'k',
                                                 'label': 'Constraints'},
                          "Cost Function": {'ls': '-', 'color': 'k'},
                          "Economic function": {'ls': '-', 'color': 'k'},
                          "Unstable modes slacks": {'ls': '-', 'color': 'k'},
                          "Integrating modes slacks": {'ls': '-', 'color': 'k'}}
            self.nominal = True
            # The second subplot displayed with the cost function can be either its "delta" or "derivative".
            self.costComplement = "delta"
            self.hideInputConstraints = False

            # show and save are overwritten by ClosedLoop.results()
            self.save = True
            self.show = False

    def __init__(self, system, controller):
        self.system = system
        self.controller = controller
        self.configPlot = self.ConfigPlots()

    def initialConditions(self, system_state=None, controller_state=None):
        # Default at equilibrium point
        system_state = system_state if system_state is not None else [0] * self.system.nx
        controller_state = controller_state if controller_state is not None else [0] * self.controller.nx
        # Initial Conditions
        self.system.initialConditions(system_state)
        self.controller.model.initialConditions(controller_state)

    def simulation(self, tf, u0=None, ypk=None, umin=None, umax=None, dumax=None, spec_change=None, ysp=None,
                   ymin=None, ymax=None, input_target=None, esp=None, show=False, save=True):
        self.configPlot.show = show
        self.configPlot.save = save
        # Time
        # Finding nsim such that tf is greater than, or equal to, its specified value
        nsim: int = int(ceil(tf / self.controller.model.dt)) + 1
        # Removing values specified after tf
        if spec_change is None:
            spec_change = []
            warn('No spec_change was provided, assuming an empty list.')
        if spec_change is not None:
            for time in spec_change[::-1]:
                if time > tf:
                    warn("Changes after tf has been removed: " + str(time) + " > " + str(tf))
                    spec_change.remove(time)

        # Controlled Variables
        ypk = ypk or self.system.trend['ymk'][:, -1:]
        if self.controller.config.zone:
            if ymin is None or ymax is None:
                raise ValueError('Zone has not been specified, this controller requires ymin and ymax.')
            if spec_change is not None:
                if len(ymin) - 1 != len(spec_change) or len(ymax) - 1 != len(spec_change):
                    raise ValueError('Incorrect number of zone constraints.')
                for i in range(len(spec_change) + 1):
                    if len(ymin[i]) != self.controller.ny:
                        raise ValueError(f'Incorrect number of outputs at index {i} in the lower zone constraint.')
                    if len(ymax[i]) != self.controller.ny:
                        raise ValueError(f'Incorrect number of outputs at index {i} in the upper zone constraint.')
                    # TODO: Ajustar os erros
                    # TODO: Avaliar se a variável foi fornecida ou é None. Do contrário há o erro:
                    #  if len(ymin) - 1 != len(spec_change) or len(ymax) - 1 != len(spec_change):
                    #  TypeError: object of type 'NoneType' has no len()
        else:
            if ysp is None:
                raise ValueError('Set-point has not been specified, this controller requires ysp.')
            elif any([len(sp) != self.controller.ny for sp in ysp]):
                raise ValueError('The number of set-points does not match the number of outputs.')

            if spec_change is not None:
                if len(ysp) - 1 != len(spec_change):
                    raise ValueError('Incorrect number of set-point values or set-point changes.')

        ymin = Trend.trend_tile(ymin, nsim, dt=self.controller.model.dt, y_change=spec_change)
        ymax = Trend.trend_tile(ymax, nsim, dt=self.controller.model.dt, y_change=spec_change)
        if ymin.dtype.name != 'object':
            if (ymin > ymax).any():
                raise ValueError('Infeasible constraint: ymin > ymax')

        ysp, self.ysp_change = Trend.trend_tile(ysp, nsim, dt=self.controller.model.dt, y_change=spec_change,
                                                get_change_instants=True)

        # Manipulated variables
        u0 = array(u0, ndmin=2).transpose() if u0 is not None else array([0] * self.controller.nu, ndmin=2).transpose()
        umin = array(umin, ndmin=2).transpose()
        umax = array(umax, ndmin=2).transpose()
        dumax = array(dumax, ndmin=2).transpose()

        if not self.system.nu == u0.shape[0]:
            raise ValueError('Incorrect number of initial inputs (u0)')
        if not self.system.nu == umax.shape[0]:
            raise ValueError('Incorrect number of upper input constraints (umax)')
        if not self.system.nu == umin.shape[0]:
            raise ValueError('Incorrect number of lower input constraints (umin)')
        if not self.system.nu == dumax.shape[0]:
            raise ValueError('Incorrect number of input rate constraints (dumax)')

        # TODO: This tile logic should be moved to Trend.trend_tile()
        if input_target is None:
            input_target = [0] * self.controller.nu
            input_target = Trend.trend_tile(input_target, nsim)
        elif isinstance(input_target, list) and not isinstance(input_target[0], list):
            input_target = Trend.trend_tile([input_target], nsim)
        elif len(input_target) - 1 == len(spec_change):
            input_target = Trend.trend_tile(input_target, nsim, dt=self.controller.model.dt, y_change=spec_change)
        else:
            raise ValueError('Incorrect number of input_target set-point values.')

        # Economic Set-point
        if esp is None:
            if self.controller.config.eco:
                raise ValueError('An economic set-point target is required.')
            esp = Trend.trend_tile(esp, nsim)
        elif isinstance(esp, int) or isinstance(esp, float):
            esp = Trend.trend_tile(esp, nsim)
        elif len(esp) - 1 == len(spec_change):
            esp = array(esp, ndmin=2).transpose()
            esp = Trend.trend_tile(esp, nsim, dt=self.controller.model.dt, y_change=spec_change)
        else:
            raise ValueError('Incorrect number of economic set-point targets.')

        # Initial condition at t=0
        self.controller.initialConditions(u0, self.controller.model.xmk, ysp[:, 0:1])

        # Simulation of other nsim-1 points
        self.controller.config.ready()
        for k in range(1, nsim):

            # Caculating the control action
            duk1, uk1 = self.controller.solve(ypk, umin=umin, umax=umax, dumax=dumax, ysp=ysp[:, k:k + 1],
                                              ymin=ymin[:, k:k + 1], ymax=ymax[:, k:k + 1],
                                              input_target=input_target[:, k:k + 1],
                                              esp=esp[:, k:k + 1])

            # Applying the control action
            if self.system.input_type == "Incremental":
                ypk = self.system.update(duk1)
            else:
                ypk = self.system.update(uk1)

        self.results()

    def results(self):
        # TODO: toggle constraints

        # Using ConfigResults
        # self.configPlot.show = True

        time = self.controller.model.dt * array([i for i in range(self.controller.trend["uk"].shape[1])])

        # Manipulated Variables
        self._plotmanager = PlotManager(folder=self.configPlot.folder)
        self._plotmanager.plot(time, self.controller.trend["uk"] + self.controller.model.uss,
                               config_plot=self.configPlot.style["Inputs"],
                               subfolder=self.configPlot.subfolder,
                               filename=self.configPlot.filename["Inputs"])
        self._plotmanager.plot_on_top(time[1:], self.controller.trend["umax"] + self.controller.model.uss,
                                      config_plot=self.configPlot.style["Input Constraints"])
        self._plotmanager.plot_on_top(time[1:], self.controller.trend["umin"] + self.controller.model.uss,
                                      config_plot={**self.configPlot.style["Input Constraints"], **{'label': None}})
        # Manipulated Variables - Targets
        if self.controller.config.input_target:
            self._plotmanager.plot_on_top(time[1:], self.controller.trend["input_target"] + self.controller.model.uss,
                                          config_plot=self.configPlot.style["Input target"])

        # # Manipulated Variables - Label
        self._plotmanager.label(self.controller.model.labels['time'], self.controller.model.labels['inputs'])

        # self._plotmanager.save_close()

        # Controlled Variables
        self._plotmanager.plot(time, self.system.trend['ymk'] + self.controller.model.yss,
                               config_plot=self.configPlot.style["Outputs"],
                               subfolder=self.configPlot.subfolder,
                               filename=self.configPlot.filename["Outputs"])
        # self._output_results.plot(time, self.system._trendObj.ymk[0:,:-1] + self.controller.model.yss)
        # self._output_results.plot(time, self.system.trend['ymk'] + self.controller.model.yss)

        # Controlled Variables - Zone/Set-point tracking
        if self.controller.config.zone:
            self._plotmanager.plot_on_top(time[1:],
                                          self.controller.trend["ysp"] + self.controller.model.yss,
                                          config_plot=self.configPlot.style["Output Targets"])
            self._plotmanager.plot_on_top(time[1:],
                                          self.controller.trend["ymin"] + self.controller.model.yss,
                                          config_plot=self.configPlot.style["Output Constraints"])
            self._plotmanager.plot_on_top(time[1:],
                                          self.controller.trend["ymax"] + self.controller.model.yss,
                                          config_plot=self.configPlot.style["Output Constraints"])
        else:
            self._plotmanager.plot_on_top(time, self.controller.trend["ysp"] + self.controller.model.yss,
                                          config_plot=self.configPlot.style["Output Targets"])
        # Controlled Variables - Label
        self._plotmanager.label(self.controller.model.labels['time'], self.controller.model.labels['outputs'])

        # Cost Function
        Vk = self.controller.trend["fval"]

        # dVk[i]       = Vk[i+1]     - Vk[i], corresponding to:
        # dVk(k+1|k+1) = Vk(k+1|k+1) - Vk(k|k),
        # therefore dVk is one step ahead of Vk
        dVk = diff(Vk)

        if self.ysp_change is not None:
            for i in self.ysp_change:
                # time vector starts at 0
                # First Vk is calculated at t = time(k)
                # First dVk  is calculated at t = time(k+1)
                dVk[0][i - 2] = 0

        t = [time[1:], time[2:]]
        costF = [Vk[0][:], dVk[0][:]]
        if self.configPlot.costComplement == "delta":
            second_label = self.configPlot.costLabel["Delta"]
            style = self.configPlot.style["Cost Function"]
        else:
            second_label = self.configPlot.costLabel["Derivative"]
            costF[1] = Vk[0][:], dVk[0][:] / self.controller.model.dt
            style = self.configPlot.style["Cost Function"]

        self._plotmanager.plot(t, costF,
                               config_plot=style,
                               subfolder=self.configPlot.subfolder,
                               filename=self.configPlot.filename["Cost Function"])
        self._plotmanager.label(self.controller.model.labels['time'],
                                [self.configPlot.costLabel["Cost Function"], second_label])

        if self.controller.config.eco:
            if self.controller.config.eco_tracking:
                self._plotmanager.plot(time[1:], self.controller.trend["feco"],
                                       config_plot=self.configPlot.style["Cost Function"],
                                       subfolder=self.configPlot.subfolder,
                                       filename=self.configPlot.filename["Economic Function"])
            else:
                self._plotmanager.plot(time[0:-1], self.controller.trend["feco"],
                                       config_plot=self.configPlot.style["Cost Function"],
                                       subfolder=self.configPlot.subfolder,
                                       filename=self.configPlot.filename["Economic Function"])
                self._plotmanager.plot_on_top(
                    time[0: - 1 - self.controller.m] + self.controller.model.dt * self.controller.m,
                    self.controller.trend["feco_h"][:, :-self.controller.m])
            self._plotmanager.label(self.controller.model.labels['time'], '$Feco_{k}$')

        if self.controller.config.unstable:
            skun = self.controller.trend["skun"]
            self._plotmanager.plot(time[1:], skun, config_plot=self.configPlot.style["Unstable modes slacks"],
                                   subfolder=self.configPlot.subfolder,
                                   filename=self.configPlot.filename["Skun"])

            if skun.shape[0] == 1:
                slack_label = ['$\delta_{un}$']
            else:
                slack_label = ['$\delta_{un' + str(i + 1) + '}$' for i in range(skun.shape[0])]

            self._plotmanager.label(self.controller.model.labels['time'], slack_label)

        if self.controller.config.integrating:
            # Used slacks
            used = (self.controller.trend["ski"] != 0).any(1)  # TODO: adjustable tol
            nsi = sum(used)
            if nsi:
                slack = self.controller.trend["ski"][used, :]
                self._plotmanager.plot(time[1:], slack, config_plot=self.configPlot.style["Integrating modes slacks"],
                                       subfolder=self.configPlot.subfolder,
                                       filename=self.configPlot.filename["Ski"])
                if nsi == 1:
                    slack_label = ['$\delta_{i}$']
                else:
                    slack_label = ['$\delta_{i' + str(i + 1) + '}$' for i in range(nsi)]

                self._plotmanager.label(self.controller.model.labels['time'], slack_label)

        # # Save and/or Close
        self._plotmanager.show(to_save=self.configPlot.save, to_show=self.configPlot.show)

        # Performance indicators
        self.getPerformance()

    def getPerformance(self, time_vector=None):
        dt = self.controller.model.dt
        if time_vector is None:
            time_vector = dt * array([i for i in range(self.controller.trend["uk"].shape[1])])

        if self.controller.config.zone:
            # ysp calculated after the first ymk
            E = self.system.trend["ymk"][:, 1:] - self.controller.trend["ysp"]
            self.performance = {'IAE': sum(sum(dt * abs(E))),
                                'ISE': sum(sum(dt * E ** 2)),
                                'ITAE': sum(dt * dot(abs(E), time_vector[1:])),
                                'ITSE': sum(dt * dot(E ** 2, time_vector[1:]))}
        else:
            # ysp assigned from the start
            E = self.system.trend["ymk"] - self.controller.trend["ysp"]
            self.performance = {'IAE': sum(sum(dt * abs(E))),
                                'ISE': sum(sum(dt * E ** 2)),
                                'ITAE': sum(dt * dot(abs(E), time_vector)),
                                'ITSE': sum(dt * dot(E ** 2, time_vector))}

        return self.performance
