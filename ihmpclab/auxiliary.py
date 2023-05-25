import os
from typing import Union
from warnings import warn

# Matrices (using numpy v. 1.18.1)
from numpy import array, identity, tile, dot, hstack, diff, ceil, sort
from numpy.linalg import multi_dot, inv

# Results (using matplotlib v. 3.1.3)
from matplotlib.pyplot import figure, close, rc, show, ioff, axes, text
from matplotlib.ticker import (AutoMinorLocator, )
from matplotlib import use

use('Agg')

# TODO: detect whether the user has the compiler
# rc('text', usetex=True) # this line requires Latex compiler
rc('font', **{'family': 'serif', 'weight': 'normal', 'variant': 'normal', 'size': 10})
# Set the font used for MathJax - more on this later
rc('mathtext', **{'fontset': 'cm'})


class PlotManager:
    class LineData:
        def __init__(self, x, y, config_plot=None):
            self.x = x
            self.y = y
            self.config_plot = config_plot

    class PlotPile:
        # FigureGroup
        def __init__(self, config_fig=None, autosplit=True, subplots_per_fig=3, folder='', subfolder='',
                     filename: Union[str, list] = 'Figure'):

            self.config_fig = config_fig  # Figure configuration
            self.folder = folder
            self.subfolder = subfolder
            self.filename = filename

            # Boolean to use multiple images if the maximum number of subplots is reached
            self.autosplit = autosplit
            self.subplots_per_fig = subplots_per_fig  # Maximum for subplots per figure

            # Hidden defaults for matplotlib
            self.tick_params = {'which': 'both', 'direction': 'out', 'bottom': True, 'left': True, 'width': 1}
            self.grid_params = {'which': 'both', 'color': '#CCCCCC'}

            # Hidden defaults for other methods
            self.dir = []
            self.lines = []
            self.xlabel = ''
            self.ylabel = ''
            self.labelled_xy = False

        def plot(self, x, y, config_plot=None):
            self.lines.append(PlotManager.LineData(x, y, config_plot=config_plot))

        @property
        def nvar(self):
            # The first plot defines the total number of subplots
            if isinstance(self.lines[0].y, list):
                return len(self.lines[0].y)
            else:
                return self.lines[0].y.shape[0]

        def split(self):
            if self.autosplit:
                nfull = int(self.nvar / self.subplots_per_fig)

                # Figures containing the maximum number of subplots
                self.nsub = [self.subplots_per_fig] * nfull
                remainder = self.nvar - self.subplots_per_fig * nfull
                if remainder > 0:
                    self.nsub.append(remainder)
                self.nfig = len(self.nsub)
            else:
                self.nfig = 1
                self.nsub = self.nvar

        def show(self, to_save=True):
            self.split()  # Determines how to split the subplots into figures

            if to_save:
                self.getDir()  # Finds directories by joining the folder path and file names (requires split() )
                savefig = self.save
            else:
                savefig = lambda a, b: None
            ylabel_counter = 0
            current_sub = 0

            for fig_index in range(self.nfig):
                # Each figure with nsub subplots
                f = figure(**self.config_fig)
                axes = f.subplots(self.nsub[fig_index], 1)

                if self.nsub[fig_index] == 1:
                    # For a single subplot axes is not an iterable by default
                    axes = [axes]

                # self.add_legend(axes[0])
                for subindex in range(self.nsub[fig_index]):
                    for line in self.lines:
                        axes[subindex].plot(line.x[current_sub],
                                            line.y[current_sub],
                                            **line.config_plot)
                        # axes[subindex].set_ylabel(line.config_plot['label'])
                        self.config_axis(axes[subindex])
                    current_sub += 1

                # Y and X axes labels
                if self.labelled_xy:
                    # xlabel
                    axes[-1].set_xlabel(self.xlabel)
                    # ylabels
                    for subindex in range(self.nsub[fig_index]):
                        axes[subindex].set_ylabel(self.ylabel[subindex + ylabel_counter])
                    ylabel_counter += self.nsub[fig_index]

                # Legend labels
                if not all([line.config_plot['label'] is None for line in self.lines]):
                    self.add_legend(axes[0])

                # Saving
                savefig(f, fig_index)

        def save(self, current_figure, fig_index):
            current_figure.savefig(self.dir[fig_index])

        def config_axis(self, axis):
            # axis.grid()
            axis.xaxis.set_minor_locator(AutoMinorLocator())
            axis.yaxis.set_minor_locator(AutoMinorLocator())
            axis.tick_params(**self.tick_params)
            axis.grid(**self.grid_params)
            # axis.set_aspect(.5)

        def add_legend(self, axis):
            # handles=axis.lines # uses every label

            # adds non-duplicates
            handles = []
            labels = []
            for index, line in enumerate(axis.lines):
                if self.lines[index].config_plot['label'] is not None and line._label not in labels:
                    handles.append(line)
                    labels.append(line._label)

            # for index, line in enumerate(self.lines):
            #     if line.config_plot['label'] is not None:
            #         handles.append(axis.lines[index])

            axis.legend(handles=handles, loc='lower center', ncol=len(handles),
                        bbox_to_anchor=(0, 1.02, 1, 0.2),
                        fancybox=True, shadow=True)

        def getDir(self):
            folderpath = os.path.join(self.folder, self.subfolder)
            if self.folder != '':
                if not os.path.isdir(self.folder):
                    os.mkdir(self.folder)
                    os.mkdir(folderpath)
                elif not os.path.isdir(folderpath):
                    os.mkdir(folderpath)

            if isinstance(self.filename, str):
                if self.nfig == 1:
                    directories = [os.path.join(folderpath, self.filename)]
                else:
                    directories = [os.path.join(folderpath, self.filename) + ' ' + str(i) for i in range(self.nfig)]
            else:
                # self.filename is a list
                directories = [os.path.join(folderpath, filename) for filename in self.filename]
            self.dir = directories

        def label(self, xlabel: str = '', ylabel: Union[str, list] = ''):
            self.labelled_xy = True
            if isinstance(ylabel, str):
                ylabel = [ylabel]
            if len(ylabel) != self.nvar:
                warn('\n\nIncorrect number of labels.'
                     '\nExpected ' + str(self.nvar) + ', got ' + str(len(ylabel)) + '.')
                ylabel = self.nvar * ['']
            self.xlabel = xlabel
            self.ylabel = ylabel

    def __init__(self, folder=''):
        self.folder = folder
        self.pile = []

        self.default_config_fig = {'dpi': 200, 'clear': True, 'frameon': True, 'tight_layout': True,
                                   'figsize': [6.4, 4.8]}
        self.default_config_plot = {'ls': '-', 'label': None}

    def plot(self, x, y, config_fig=None, config_plot=None, subfolder='', filename: Union[str, list] = 'Figure'):
        x, y = self.correctData(x, y)

        if config_fig is not None:
            config_fig = {**self.default_config_fig, **config_fig}
        else:
            config_fig = self.default_config_fig

        # Starts a new stack
        self.pile.append(
            self.PlotPile(config_fig=config_fig, folder=self.folder, subfolder=subfolder, filename=filename))

        # Add plot to own stack
        self.plot_on_top(x, y, config_plot=config_plot)

    def plot_on_top(self, x, y, config_plot=None):
        config_plot = config_plot or self.default_config_plot

        if config_plot is not None:
            config_plot = {**self.default_config_plot, **config_plot}
        else:
            config_plot = self.default_config_plot

        x, y = self.correctData(x, y)
        self.pile[-1].plot(x, y, config_plot=config_plot)

    def correctData(self, x, y):
        if isinstance(x, list):
            return x, y

        x = array(x, ndmin=2)
        y = array(y, ndmin=2)
        if (x.ndim == 1 or x.shape[0] == 1) and y.shape[0] > 1:
            x = tile(x, [y.shape[0], 1])
        return x, y

    def label(self, xlabel: str = '', ylabel: Union[str, list] = ''):
        self.pile[-1].label(xlabel, ylabel)

    def show(self, to_save=True, to_show=True):
        # Show/Save
        ioff()  # Turns off interactive mode, results are shown only when show() is called
        for stack in self.pile:
            stack.show(to_save=to_save)
        if to_show:
            show()


class Kalman:
    def __init__(self, A, C, w=.1e-4, v=.1e-3, niter=100, dynamic=False):
        self.A = A
        self.C = C
        nx = self.A.shape[0]
        ny = self.C.shape[0]

        self.W = w * identity(nx)
        self.V = v * identity(ny)

        self.P = identity(nx)

        self.getGain(niter)

    def getGain(self, niter):
        for i in range(niter):
            self.P = multi_dot([self.A, self.P, self.A.transpose()]) \
                     - multi_dot([self.A, self.P, self.C.transpose(),
                                  dot(inv(self.V + multi_dot([self.C, self.P, self.C.transpose()])),
                                      multi_dot([self.C, self.P, self.A.transpose()]))
                                  ]) + self.W
        self.gain = multi_dot([self.A, self.P, self.C.transpose(),
                               inv(self.V + multi_dot([self.C, self.P, self.C.transpose()]))
                               ])

    # def update(self):
    #     self.P = multi_dot([self.A, self.P, self.A.transpose()]) + self.W
    #     self.gain = dot(self.P, self.C.transpose()).dot(inv(multi_dot([self.C, self.P, self.C.transpose()])))
    #     self.P = dot(identity(self.A.shape[0]) - self.gain.dot(self.C), self.P)
    #     return self.gain


# TODO: static method

class Trend:
    def __init__(self, first_dict):
        self.trends = {}
        self.mergeDict(first_dict)

    def historyData(self, **new_values):
        for key in new_values:
            self.trends[key] = hstack((self.trends[key], new_values[key]))

    def mergeDict(self, new_dict):
        self.trends.update(new_dict)

    def get(self, key):
        return self.trends[key]

    @staticmethod
    def trend_tile(y, nsim, dt=None, y_change=None, get_change_instants=False):
        if y_change is None:
            trend = tile(array(y, ndmin=2).transpose(), nsim)
            return trend
        else:

            y = array(y, ndmin=2).transpose()

            # Sorted array in time samples
            y_change = ceil(array(y_change) / dt)
            y_change = sort(y_change)

            # Number of time samples for each value
            len_y = hstack((0,
                            y_change,
                            nsim
                            ))
            len_y = diff(len_y)

            # Use as list of ints easier indexation
            len_y = [int(i) for i in len_y]
            y_change = [int(i) for i in y_change]

            # stacking tiles
            trend = tile(y[:, 0:1], len_y[0])
            for k in range(1, len(y_change) + 1):
                trend = hstack((trend,
                                tile(y[:, k:k + 1], len_y[k])
                                ))
        if get_change_instants:
            return trend, y_change
        else:
            return trend
