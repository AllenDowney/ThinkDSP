"""This file contains code for use with "Think Stats",
by Allen B. Downey, available from greenteapress.com

Copyright 2010 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

import math
import matplotlib
import matplotlib.pyplot as pyplot
import numpy as np

# customize some matplotlib attributes
#matplotlib.rc('figure', figsize=(4, 3))

#matplotlib.rc('font', size=14.0)
#matplotlib.rc('axes', labelsize=22.0, titlesize=22.0)
#matplotlib.rc('legend', fontsize=20.0)

#matplotlib.rc('xtick.major', size=6.0)
#matplotlib.rc('xtick.minor', size=3.0)

#matplotlib.rc('ytick.major', size=6.0)
#matplotlib.rc('ytick.minor', size=3.0)


class Brewer(object):
    """Encapsulates a nice sequence of colors.

    Shades of blue that look good in color and can be distinguished
    in grayscale (up to a point).
    
    Borrowed from http://colorbrewer2.org/
    """
    color_iter = None

    colors = ['#081D58',
              '#253494',
              '#225EA8',
              '#1D91C0',
              '#41B6C4',
              '#7FCDBB',
              '#C7E9B4',
              '#EDF8B1',
              '#FFFFD9']

    # lists that indicate which colors to use depending on how many are used
    which_colors = [[],
                    [1],
                    [1, 3],
                    [0, 2, 4],
                    [0, 2, 4, 6],
                    [0, 2, 3, 5, 6],
                    [0, 2, 3, 4, 5, 6],
                    [0, 1, 2, 3, 4, 5, 6],
                    ]

    @classmethod
    def Colors(cls):
        """Returns the list of colors.
        """
        return cls.colors

    @classmethod
    def ColorGenerator(cls, n):
        """Returns an iterator of color strings.

        n: how many colors will be used
        """
        for i in cls.which_colors[n]:
            yield cls.colors[i]
        raise StopIteration('Ran out of colors in Brewer.ColorGenerator')

    @classmethod
    def InitializeIter(cls, num):
        """Initializes the color iterator with the given number of colors."""
        cls.color_iter = cls.ColorGenerator(num)

    @classmethod
    def ClearIter(cls):
        """Sets the color iterator to None."""
        cls.color_iter = None

    @classmethod
    def GetIter(cls):
        """Gets the color iterator."""
        return cls.color_iter


def PrePlot(num=None, rows=1, cols=1):
    """Takes hints about what's coming.

    num: number of lines that will be plotted
    """
    if num:
        Brewer.InitializeIter(num)

    # TODO: get sharey and sharex working.  probably means switching
    # to subplots instead of subplot.
    # also, get rid of the gray background.

    if rows > 1 or cols > 1:
        pyplot.subplots(rows, cols, sharey=True)
        global SUBPLOT_ROWS, SUBPLOT_COLS
        SUBPLOT_ROWS = rows
        SUBPLOT_COLS = cols
    

def SubPlot(plot_number):
    pyplot.subplot(SUBPLOT_ROWS, SUBPLOT_COLS, plot_number)


class InfiniteList(list):
    """A list that returns the same value for all indices."""
    def __init__(self, val):
        """Initializes the list.

        val: value to be stored
        """
        list.__init__(self)
        self.val = val

    def __getitem__(self, index):
        """Gets the item with the given index.

        index: int

        returns: the stored value
        """
        return self.val


def Underride(d, **options):
    """Add key-value pairs to d only if key is not in d.

    If d is None, create a new dictionary.

    d: dictionary
    options: keyword args to add to d
    """
    if d is None:
        d = {}

    for key, val in options.iteritems():
        d.setdefault(key, val)

    return d


def Clf():
    """Clears the figure and any hints that have been set."""
    Brewer.ClearIter()
    pyplot.clf()
    

def Figure(**options):
    """Sets options for the current figure."""
    Underride(options, figsize=(6, 8))
    pyplot.figure(**options)
    

def Plot(xs, ys, style='', **options):
    """Plots a line.

    Args:
      xs: sequence of x values
      ys: sequence of y values
      style: style string passed along to pyplot.plot
      options: keyword args passed to pyplot.plot
    """
    color_iter = Brewer.GetIter()

    if color_iter:
        try:
            options = Underride(options, color=color_iter.next())
        except StopIteration:
            print 'Warning: Brewer ran out of colors.'
            Brewer.ClearIter()
        
    options = Underride(options, linewidth=3, alpha=0.8)
    pyplot.plot(xs, ys, style, **options)


def Scatter(xs, ys, **options):
    """Makes a scatter plot.

    xs: x values
    ys: y values
    options: options passed to pyplot.scatter
    """
    options = Underride(options, color='blue', alpha=0.2, 
                        s=30, edgecolors='none')
    pyplot.scatter(xs, ys, **options)


def Pmf(pmf, **options):
    """Plots a Pmf or Hist as a line.

    Args:
      pmf: Hist or Pmf object
      options: keyword args passed to pyplot.plot
    """
    xs, ps = pmf.Render()
    if pmf.name:
        options = Underride(options, label=pmf.name)
    Plot(xs, ps, **options)


def Pmfs(pmfs, **options):
    """Plots a sequence of PMFs.

    Options are passed along for all PMFs.  If you want different
    options for each pmf, make multiple calls to Pmf.
    
    Args:
      pmfs: sequence of PMF objects
      options: keyword args passed to pyplot.plot
    """
    for pmf in pmfs:
        Pmf(pmf, **options)


def Hist(hist, **options):
    """Plots a Pmf or Hist with a bar plot.

    Args:
      hist: Hist or Pmf object
      options: keyword args passed to pyplot.bar
    """
    # find the minimum distance between adjacent values
    xs, fs = hist.Render()
    width = min(Diff(xs))

    if hist.name:
        options = Underride(options, label=hist.name)

    options = Underride(options, 
                        align='center',
                        linewidth=0,
                        width=width)

    pyplot.bar(xs, fs, **options)


def Hists(hists, **options):
    """Plots two histograms as interleaved bar plots.

    Options are passed along for all PMFs.  If you want different
    options for each pmf, make multiple calls to Pmf.

    Args:
      hists: list of two Hist or Pmf objects
      options: keyword args passed to pyplot.plot
    """
    for hist in hists:
        Hist(hist, **options)


def Diff(t):
    """Compute the differences between adjacent elements in a sequence.

    Args:
        t: sequence of number

    Returns:
        sequence of differences (length one less than t)
    """
    diffs = [t[i+1] - t[i] for i in range(len(t)-1)]
    return diffs


def Cdf(cdf, complement=False, transform=None, **options):
    """Plots a CDF as a line.

    Args:
      cdf: Cdf object
      complement: boolean, whether to plot the complementary CDF
      transform: string, one of 'exponential', 'pareto', 'weibull', 'gumbel'
      options: keyword args passed to pyplot.plot

    Returns:
      dictionary with the scale options that should be passed to
      myplot.Save or myplot.Show
    """
    xs, ps = cdf.Render()
    scale = dict(xscale='linear', yscale='linear')

    if transform == 'exponential':
        complement = True
        scale['yscale'] = 'log'

    if transform == 'pareto':
        complement = True
        scale['yscale'] = 'log'
        scale['xscale'] = 'log'

    if complement:
        ps = [1.0-p for p in ps]

    if transform == 'weibull':
        xs.pop()
        ps.pop()
        ps = [-math.log(1.0-p) for p in ps]
        scale['xscale'] = 'log'
        scale['yscale'] = 'log'

    if transform == 'gumbel':
        xs.pop(0)
        ps.pop(0)
        ps = [-math.log(p) for p in ps]
        scale['yscale'] = 'log'

    if cdf.name:
        options = Underride(options, label=cdf.name)

    Plot(xs, ps, **options)
    return scale


def Cdfs(cdfs, complement=False, transform=None, **options):
    """Plots a sequence of CDFs.
    
    cdfs: sequence of CDF objects
    complement: boolean, whether to plot the complementary CDF
    transform: string, one of 'exponential', 'pareto', 'weibull', 'gumbel'
    options: keyword args passed to pyplot.plot
    """
    for cdf in cdfs:
        Cdf(cdf, complement, transform, **options)


def Contour(obj, pcolor=False, contour=True, imshow=False, **options):
    """Makes a contour plot.
    
    d: map from (x, y) to z, or object that provides GetDict
    pcolor: boolean, whether to make a pseudocolor plot
    contour: boolean, whether to make a contour plot
    imshow: boolean, whether to use pyplot.imshow
    options: keyword args passed to pyplot.pcolor and/or pyplot.contour
    """
    try:
        d = obj.GetDict()
    except AttributeError:
        d = obj

    Underride(options, linewidth=3, cmap=matplotlib.cm.Blues)

    xs, ys = zip(*d.iterkeys())
    xs = sorted(set(xs))
    ys = sorted(set(ys))

    X, Y = np.meshgrid(xs, ys)
    func = lambda x, y: d.get((x, y), 0)
    func = np.vectorize(func)
    Z = func(X, Y)

    x_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
    axes = pyplot.gca()
    axes.xaxis.set_major_formatter(x_formatter)

    if pcolor:
        pyplot.pcolormesh(X, Y, Z, **options)
    if contour:
        cs = pyplot.contour(X, Y, Z, **options)
        pyplot.clabel(cs, inline=1, fontsize=10)
    if imshow:
        extent = xs[0], xs[-1], ys[0], ys[-1]
        pyplot.imshow(Z, extent=extent, **options)
        

def Pcolor(xs, ys, zs, pcolor=True, contour=False, **options):
    """Makes a pseudocolor plot.
    
    xs:
    ys:
    zs:
    pcolor: boolean, whether to make a pseudocolor plot
    contour: boolean, whether to make a contour plot
    options: keyword args passed to pyplot.pcolor and/or pyplot.contour
    """
    Underride(options, linewidth=3, cmap=matplotlib.cm.Blues)

    X, Y = np.meshgrid(xs, ys)
    Z = zs

    x_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
    axes = pyplot.gca()
    axes.xaxis.set_major_formatter(x_formatter)

    if pcolor:
        pyplot.pcolormesh(X, Y, Z, **options)

    if contour:
        cs = pyplot.contour(X, Y, Z, **options)
        pyplot.clabel(cs, inline=1, fontsize=10)
        

def Config(**options):
    """Configures the plot.

    Pulls options out of the option dictionary and passes them to
    title, xlabel, ylabel, xscale, yscale, xticks, yticks, axis, legend,
    and loc.
    """
    title = options.get('title', '')
    pyplot.title(title)

    xlabel = options.get('xlabel', '')
    pyplot.xlabel(xlabel)

    ylabel = options.get('ylabel', '')
    pyplot.ylabel(ylabel)

    if 'xscale' in options:
        pyplot.xscale(options['xscale'])

    if 'xticks' in options:
        pyplot.xticks(options['xticks'])

    if 'yscale' in options:
        pyplot.yscale(options['yscale'])

    if 'yticks' in options:
        pyplot.yticks(options['yticks'])

    if 'axis' in options:
        pyplot.axis(options['axis'])

    loc = options.get('loc', 0)
    legend = options.get('legend', True)
    if legend:
        pyplot.legend(loc=loc)


def Show(**options):
    """Shows the plot.

    For options, see Config.

    options: keyword args used to invoke various pyplot functions
    """
    # TODO: figure out how to show more than one plot
    Config(**options)
    pyplot.show()


def Save(root=None, formats=None, **options):
    """Saves the plot in the given formats.

    For options, see Config.

    Args:
      root: string filename root
      formats: list of string formats
      options: keyword args used to invoke various pyplot functions
    """
    Config(**options)

    if formats is None:
        formats = ['pdf', 'eps']

    if root:
        for fmt in formats:
            SaveFormat(root, fmt)
    Clf()


def SaveFormat(root, fmt='eps'):
    """Writes the current figure to a file in the given format.

    Args:
      root: string filename root
      fmt: string format
    """
    filename = '%s.%s' % (root, fmt)
    print 'Writing', filename
    pyplot.savefig(filename, format=fmt, dpi=300)


# provide aliases for calling functons with lower-case names
preplot = PrePlot
subplot = SubPlot
clf = Clf
figure = Figure
plot = Plot
scatter = Scatter
pmf = Pmf
pmfs = Pmfs
hist = Hist
hists = Hists
diff = Diff
cdf = Cdf
cdfs = Cdfs
contour = Contour
pcolor = Pcolor
config = Config
show = Show
save = Save


def main():
    color_iter = Brewer.ColorGenerator(7)
    for color in color_iter:
        print color

if __name__ == '__main__':
    main()
