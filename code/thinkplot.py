"""This file contains code for use with "Think Stats",
by Allen B. Downey, available from greenteapress.com

Copyright 2014 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function

import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import warnings

# customize some matplotlib attributes
#matplotlib.rc('figure', figsize=(4, 3))

#matplotlib.rc('font', size=14.0)
#matplotlib.rc('axes', labelsize=22.0, titlesize=22.0)
#matplotlib.rc('legend', fontsize=20.0)

#matplotlib.rc('xtick.major', size=6.0)
#matplotlib.rc('xtick.minor', size=3.0)

#matplotlib.rc('ytick.major', size=6.0)
#matplotlib.rc('ytick.minor', size=3.0)


class _Brewer(object):
    """Encapsulates a nice sequence of colors.

    Shades of blue that look good in color and can be distinguished
    in grayscale (up to a point).

    Borrowed from http://colorbrewer2.org/
    """
    color_iter = None

    colors = ['#f7fbff', '#deebf7', '#c6dbef',
              '#9ecae1', '#6baed6', '#4292c6',
              '#2171b5','#08519c','#08306b'][::-1]

    # lists that indicate which colors to use depending on how many are used
    which_colors = [[],
                    [1],
                    [1, 3],
                    [0, 2, 4],
                    [0, 2, 4, 6],
                    [0, 2, 3, 5, 6],
                    [0, 2, 3, 4, 5, 6],
                    [0, 1, 2, 3, 4, 5, 6],
                    [0, 1, 2, 3, 4, 5, 6, 7],
                    [0, 1, 2, 3, 4, 5, 6, 7, 8],
                    ]

    current_figure = None

    @classmethod
    def Colors(cls):
        """Returns the list of colors.
        """
        return cls.colors

    @classmethod
    def ColorGenerator(cls, num):
        """Returns an iterator of color strings.

        n: how many colors will be used
        """
        for i in cls.which_colors[num]:
            yield cls.colors[i]
        raise StopIteration('Ran out of colors in _Brewer.')

    @classmethod
    def InitIter(cls, num):
        """Initializes the color iterator with the given number of colors."""
        cls.color_iter = cls.ColorGenerator(num)
        fig = plt.gcf()
        cls.current_figure = fig

    @classmethod
    def ClearIter(cls):
        """Sets the color iterator to None."""
        cls.color_iter = None
        cls.current_figure = None

    @classmethod
    def GetIter(cls, num):
        """Gets the color iterator."""
        fig = plt.gcf()
        if fig != cls.current_figure:
            cls.InitIter(num)
            cls.current_figure = fig

        if cls.color_iter is None:
            cls.InitIter(num)

        return cls.color_iter


def _UnderrideColor(options):
    """If color is not in the options, chooses a color.
    """
    if 'color' in options:
        return options

    # get the current color iterator; if there is none, init one
    color_iter = _Brewer.GetIter(5)

    try:
        options['color'] = next(color_iter)
    except StopIteration:
        # if you run out of colors, initialize the color iterator
        # and try again
        warnings.warn('Ran out of colors.  Starting over.')
        _Brewer.ClearIter()
        _UnderrideColor(options)

    return options


def PrePlot(num=None, rows=None, cols=None):
    """Takes hints about what's coming.

    num: number of lines that will be plotted
    rows: number of rows of subplots
    cols: number of columns of subplots
    """
    if num:
        _Brewer.InitIter(num)

    if rows is None and cols is None:
        return

    if rows is not None and cols is None:
        cols = 1

    if cols is not None and rows is None:
        rows = 1

    # resize the image, depending on the number of rows and cols
    size_map = {(1, 1): (8, 6),
                (1, 2): (12, 6),
                (1, 3): (12, 6),
                (1, 4): (12, 5),
                (1, 5): (12, 4),
                (2, 2): (10, 10),
                (2, 3): (16, 10),
                (3, 1): (8, 10),
                (4, 1): (8, 12),
                }

    if (rows, cols) in size_map:
        fig = plt.gcf()
        fig.set_size_inches(*size_map[rows, cols])

    # create the first subplot
    if rows > 1 or cols > 1:
        ax = plt.subplot(rows, cols, 1)
        global SUBPLOT_ROWS, SUBPLOT_COLS
        SUBPLOT_ROWS = rows
        SUBPLOT_COLS = cols
    else:
        ax = plt.gca()

    return ax


def SubPlot(plot_number, rows=None, cols=None, **options):
    """Configures the number of subplots and changes the current plot.

    rows: int
    cols: int
    plot_number: int
    options: passed to subplot
    """
    rows = rows or SUBPLOT_ROWS
    cols = cols or SUBPLOT_COLS
    return plt.subplot(rows, cols, plot_number, **options)


def _Underride(d, **options):
    """Add key-value pairs to d only if key is not in d.

    If d is None, create a new dictionary.

    d: dictionary
    options: keyword args to add to d
    """
    if d is None:
        d = {}

    for key, val in options.items():
        d.setdefault(key, val)

    return d


def Clf():
    """Clears the figure and any hints that have been set."""
    global LOC
    LOC = None
    _Brewer.ClearIter()
    plt.clf()
    fig = plt.gcf()
    fig.set_size_inches(8, 6)


def Figure(**options):
    """Sets options for the current figure."""
    _Underride(options, figsize=(6, 8))
    plt.figure(**options)


def Plot(obj, ys=None, style='', **options):
    """Plots a line.

    Args:
      obj: sequence of x values, or Series, or anything with Render()
      ys: sequence of y values
      style: style string passed along to plt.plot
      options: keyword args passed to plt.plot
    """
    options = _UnderrideColor(options)
    label = getattr(obj, 'label', '_nolegend_')
    options = _Underride(options, linewidth=3, alpha=0.7, label=label)

    xs = obj
    if ys is None:
        if hasattr(obj, 'Render'):
            xs, ys = obj.Render()
        if isinstance(obj, pd.Series):
            ys = obj.values
            xs = obj.index

    if ys is None:
        plt.plot(xs, style, **options)
    else:
        plt.plot(xs, ys, style, **options)


def Vlines(xs, y1, y2, **options):
    """Plots a set of vertical lines.

    Args:
      xs: sequence of x values
      y1: sequence of y values
      y2: sequence of y values
      options: keyword args passed to plt.vlines
    """
    options = _UnderrideColor(options)
    options = _Underride(options, linewidth=1, alpha=0.5)
    plt.vlines(xs, y1, y2, **options)


def Hlines(ys, x1, x2, **options):
    """Plots a set of horizontal lines.

    Args:
      ys: sequence of y values
      x1: sequence of x values
      x2: sequence of x values
      options: keyword args passed to plt.vlines
    """
    options = _UnderrideColor(options)
    options = _Underride(options, linewidth=1, alpha=0.5)
    plt.hlines(ys, x1, x2, **options)


def axvline(x, **options):
    """Plots a vertical line.

    Args:
      x: x location
      options: keyword args passed to plt.axvline
    """
    options = _UnderrideColor(options)
    options = _Underride(options, linewidth=1, alpha=0.5)
    plt.axvline(x, **options)


def axhline(y, **options):
    """Plots a horizontal line.

    Args:
      y: y location
      options: keyword args passed to plt.axhline
    """
    options = _UnderrideColor(options)
    options = _Underride(options, linewidth=1, alpha=0.5)
    plt.axhline(y, **options)


def tight_layout(**options):
    """Adjust subplots to minimize padding and margins.
    """
    options = _Underride(options,
                         wspace=0.1, hspace=0.1,
                         left=0, right=1,
                         bottom=0, top=1)
    plt.tight_layout()
    plt.subplots_adjust(**options)


def FillBetween(xs, y1, y2=None, where=None, **options):
    """Fills the space between two lines.

    Args:
      xs: sequence of x values
      y1: sequence of y values
      y2: sequence of y values
      where: sequence of boolean
      options: keyword args passed to plt.fill_between
    """
    options = _UnderrideColor(options)
    options = _Underride(options, linewidth=0, alpha=0.5)
    plt.fill_between(xs, y1, y2, where, **options)


def Bar(xs, ys, **options):
    """Plots a line.

    Args:
      xs: sequence of x values
      ys: sequence of y values
      options: keyword args passed to plt.bar
    """
    options = _UnderrideColor(options)
    options = _Underride(options, linewidth=0, alpha=0.6)
    plt.bar(xs, ys, **options)


def Scatter(xs, ys=None, **options):
    """Makes a scatter plot.

    xs: x values
    ys: y values
    options: options passed to plt.scatter
    """
    options = _Underride(options, color='blue', alpha=0.2,
                         s=30, edgecolors='none')

    if ys is None and isinstance(xs, pd.Series):
        ys = xs.values
        xs = xs.index

    plt.scatter(xs, ys, **options)


def HexBin(xs, ys, **options):
    """Makes a scatter plot.

    xs: x values
    ys: y values
    options: options passed to plt.scatter
    """
    options = _Underride(options, cmap=matplotlib.cm.Blues)
    plt.hexbin(xs, ys, **options)


def Pdf(pdf, **options):
    """Plots a Pdf, Pmf, or Hist as a line.

    Args:
      pdf: Pdf, Pmf, or Hist object
      options: keyword args passed to plt.plot
    """
    low, high = options.pop('low', None), options.pop('high', None)
    n = options.pop('n', 101)
    xs, ps = pdf.Render(low=low, high=high, n=n)
    options = _Underride(options, label=pdf.label)
    Plot(xs, ps, **options)


def Pdfs(pdfs, **options):
    """Plots a sequence of PDFs.

    Options are passed along for all PDFs.  If you want different
    options for each pdf, make multiple calls to Pdf.

    Args:
      pdfs: sequence of PDF objects
      options: keyword args passed to plt.plot
    """
    for pdf in pdfs:
        Pdf(pdf, **options)


def Hist(hist, **options):
    """Plots a Pmf or Hist with a bar plot.

    The default width of the bars is based on the minimum difference
    between values in the Hist.  If that's too small, you can override
    it by providing a width keyword argument, in the same units
    as the values.

    Args:
      hist: Hist or Pmf object
      options: keyword args passed to plt.bar
    """
    # find the minimum distance between adjacent values
    xs, ys = hist.Render()

    # see if the values support arithmetic
    try:
        xs[0] - xs[0]
    except TypeError:
        # if not, replace values with numbers
        labels = [str(x) for x in xs]
        xs = np.arange(len(xs))
        plt.xticks(xs+0.5, labels)

    if 'width' not in options:
        try:
            options['width'] = 0.9 * np.diff(xs).min()
        except TypeError:
            warnings.warn("Hist: Can't compute bar width automatically."
                            "Check for non-numeric types in Hist."
                            "Or try providing width option."
                            )

    options = _Underride(options, label=hist.label)
    options = _Underride(options, align='center')
    if options['align'] == 'left':
        options['align'] = 'edge'
    elif options['align'] == 'right':
        options['align'] = 'edge'
        options['width'] *= -1

    Bar(xs, ys, **options)


def Hists(hists, **options):
    """Plots two histograms as interleaved bar plots.

    Options are passed along for all PMFs.  If you want different
    options for each pmf, make multiple calls to Pmf.

    Args:
      hists: list of two Hist or Pmf objects
      options: keyword args passed to plt.plot
    """
    for hist in hists:
        Hist(hist, **options)


def Pmf(pmf, **options):
    """Plots a Pmf or Hist as a line.

    Args:
      pmf: Hist or Pmf object
      options: keyword args passed to plt.plot
    """
    xs, ys = pmf.Render()
    low, high = min(xs), max(xs)

    width = options.pop('width', None)
    if width is None:
        try:
            width = np.diff(xs).min()
        except TypeError:
            warnings.warn("Pmf: Can't compute bar width automatically."
                          "Check for non-numeric types in Pmf."
                          "Or try providing width option.")
    points = []

    lastx = np.nan
    lasty = 0
    for x, y in zip(xs, ys):
        if (x - lastx) > 1e-5:
            points.append((lastx, 0))
            points.append((x, 0))

        points.append((x, lasty))
        points.append((x, y))
        points.append((x+width, y))

        lastx = x + width
        lasty = y
    points.append((lastx, 0))
    pxs, pys = zip(*points)

    align = options.pop('align', 'center')
    if align == 'center':
        pxs = np.array(pxs) - width/2.0
    if align == 'right':
        pxs = np.array(pxs) - width

    options = _Underride(options, label=pmf.label)
    Plot(pxs, pys, **options)


def Pmfs(pmfs, **options):
    """Plots a sequence of PMFs.

    Options are passed along for all PMFs.  If you want different
    options for each pmf, make multiple calls to Pmf.

    Args:
      pmfs: sequence of PMF objects
      options: keyword args passed to plt.plot
    """
    for pmf in pmfs:
        Pmf(pmf, **options)


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
      options: keyword args passed to plt.plot

    Returns:
      dictionary with the scale options that should be passed to
      Config, Show or Save.
    """
    xs, ps = cdf.Render()
    xs = np.asarray(xs)
    ps = np.asarray(ps)

    scale = dict(xscale='linear', yscale='linear')

    for s in ['xscale', 'yscale']:
        if s in options:
            scale[s] = options.pop(s)

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
        xs = np.delete(xs, -1)
        ps = np.delete(ps, -1)
        ps = [-math.log(1.0-p) for p in ps]
        scale['xscale'] = 'log'
        scale['yscale'] = 'log'

    if transform == 'gumbel':
        xs = np.delete(xs, 0)
        ps = np.delete(ps, 0)
        ps = [-math.log(p) for p in ps]
        scale['yscale'] = 'log'

    options = _Underride(options, label=cdf.label)
    Plot(xs, ps, **options)
    return scale


def Cdfs(cdfs, complement=False, transform=None, **options):
    """Plots a sequence of CDFs.

    cdfs: sequence of CDF objects
    complement: boolean, whether to plot the complementary CDF
    transform: string, one of 'exponential', 'pareto', 'weibull', 'gumbel'
    options: keyword args passed to plt.plot
    """
    for cdf in cdfs:
        Cdf(cdf, complement, transform, **options)


def Contour(obj, pcolor=False, contour=True, imshow=False, **options):
    """Makes a contour plot.

    d: map from (x, y) to z, or object that provides GetDict
    pcolor: boolean, whether to make a pseudocolor plot
    contour: boolean, whether to make a contour plot
    imshow: boolean, whether to use plt.imshow
    options: keyword args passed to plt.pcolor and/or plt.contour
    """
    try:
        d = obj.GetDict()
    except AttributeError:
        d = obj

    _Underride(options, linewidth=3, cmap=matplotlib.cm.Blues)

    xs, ys = zip(*d.keys())
    xs = sorted(set(xs))
    ys = sorted(set(ys))

    X, Y = np.meshgrid(xs, ys)
    func = lambda x, y: d.get((x, y), 0)
    func = np.vectorize(func)
    Z = func(X, Y)

    x_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
    axes = plt.gca()
    axes.xaxis.set_major_formatter(x_formatter)

    if pcolor:
        plt.pcolormesh(X, Y, Z, **options)
    if contour:
        cs = plt.contour(X, Y, Z, **options)
        plt.clabel(cs, inline=1, fontsize=10)
    if imshow:
        extent = xs[0], xs[-1], ys[0], ys[-1]
        plt.imshow(Z, extent=extent, **options)


def Pcolor(xs, ys, zs, pcolor=True, contour=False, **options):
    """Makes a pseudocolor plot.

    xs:
    ys:
    zs:
    pcolor: boolean, whether to make a pseudocolor plot
    contour: boolean, whether to make a contour plot
    options: keyword args passed to plt.pcolor and/or plt.contour
    """
    _Underride(options, linewidth=3, cmap=matplotlib.cm.Blues)

    X, Y = np.meshgrid(xs, ys)
    Z = zs

    x_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
    axes = plt.gca()
    axes.xaxis.set_major_formatter(x_formatter)

    if pcolor:
        plt.pcolormesh(X, Y, Z, **options)

    if contour:
        cs = plt.contour(X, Y, Z, **options)
        plt.clabel(cs, inline=1, fontsize=10)


def Text(x, y, s, **options):
    """Puts text in a figure.

    x: number
    y: number
    s: string
    options: keyword args passed to plt.text
    """
    options = _Underride(options,
                         fontsize=16,
                         verticalalignment='top',
                         horizontalalignment='left')
    plt.text(x, y, s, **options)


LEGEND = True
LOC = None

def Config(**options):
    """Configures the plot.

    Pulls options out of the option dictionary and passes them to
    the corresponding plt functions.
    """
    names = ['title', 'xlabel', 'ylabel', 'xscale', 'yscale',
             'xticks', 'yticks', 'axis', 'xlim', 'ylim']

    for name in names:
        if name in options:
            getattr(plt, name)(options[name])

    global LEGEND
    LEGEND = options.get('legend', LEGEND)

    # see if there are any elements with labels;
    # if not, don't draw a legend
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()

    if LEGEND and len(labels) > 0:
        global LOC
        LOC = options.get('loc', LOC)
        frameon = options.get('frameon', True)

        try:
            plt.legend(loc=LOC, frameon=frameon)
        except UserWarning:
            pass

    # x and y ticklabels can be made invisible
    val = options.get('xticklabels', None)
    if val is not None:
        if val == 'invisible':
            ax = plt.gca()
            labels = ax.get_xticklabels()
            plt.setp(labels, visible=False)

    val = options.get('yticklabels', None)
    if val is not None:
        if val == 'invisible':
            ax = plt.gca()
            labels = ax.get_yticklabels()
            plt.setp(labels, visible=False)

def set_font_size(title_size=16, label_size=16, ticklabel_size=14, legend_size=14):
    """Set font sizes for the title, labels, ticklabels, and legend.
    """
    def set_text_size(texts, size):
        for text in texts:
            text.set_size(size)

    ax = plt.gca()

    # TODO: Make this function more robust if any of these elements
    # is missing.

    # title
    ax.title.set_size(title_size)

    # x axis
    ax.xaxis.label.set_size(label_size)
    set_text_size(ax.xaxis.get_ticklabels(), ticklabel_size)

    # y axis
    ax.yaxis.label.set_size(label_size)
    set_text_size(ax.yaxis.get_ticklabels(), ticklabel_size)

    # legend
    legend = ax.get_legend()
    if legend is not None:
        set_text_size(legend.texts, legend_size)


def bigger_text():
    sizes = dict(title_size=16, label_size=16, ticklabel_size=14, legend_size=14)
    set_font_size(**sizes)


def Show(**options):
    """Shows the plot.

    For options, see Config.

    options: keyword args used to invoke various plt functions
    """
    clf = options.pop('clf', True)
    Config(**options)
    plt.show()
    if clf:
        Clf()


def Plotly(**options):
    """Shows the plot.

    For options, see Config.

    options: keyword args used to invoke various plt functions
    """
    clf = options.pop('clf', True)
    Config(**options)
    import plotly.plotly as plotly
    url = plotly.plot_mpl(plt.gcf())
    if clf:
        Clf()
    return url


def Save(root=None, formats=None, **options):
    """Saves the plot in the given formats and clears the figure.

    For options, see Config.

    Note: With a capital S, this is the original save, maintained for
    compatibility.  New code should use save(), which works better
    with my newer code, especially in Jupyter notebooks.

    Args:
      root: string filename root
      formats: list of string formats
      options: keyword args used to invoke various plt functions
    """
    clf = options.pop('clf', True)

    save_options = {}
    for option in ['bbox_inches', 'pad_inches']:
        if option in options:
            save_options[option] = options.pop(option)

    # TODO: falling Config inside Save was probably a mistake, but removing
    # it will require some work
    Config(**options)

    if formats is None:
        formats = ['pdf', 'png']

    try:
        formats.remove('plotly')
        Plotly(clf=False)
    except ValueError:
        pass

    if root:
        for fmt in formats:
            SaveFormat(root, fmt, **save_options)
    if clf:
        Clf()


def save(root, formats=None, **options):
    """Saves the plot in the given formats and clears the figure.

    For options, see plt.savefig.

    Args:
      root: string filename root
      formats: list of string formats
      options: keyword args passed to plt.savefig
    """
    if formats is None:
        formats = ['pdf', 'png']

    try:
        formats.remove('plotly')
        Plotly(clf=False)
    except ValueError:
        pass

    for fmt in formats:
        SaveFormat(root, fmt, **options)


def SaveFormat(root, fmt='eps', **options):
    """Writes the current figure to a file in the given format.

    Args:
      root: string filename root
      fmt: string format
    """
    _Underride(options, dpi=300)
    filename = '%s.%s' % (root, fmt)
    print('Writing', filename)
    plt.savefig(filename, format=fmt, **options)


# provide aliases for calling functions with lower-case names
preplot = PrePlot
subplot = SubPlot
clf = Clf
figure = Figure
plot = Plot
vlines = Vlines
hlines = Hlines
fill_between = FillBetween
text = Text
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


def main():
    color_iter = _Brewer.ColorGenerator(7)
    for color in color_iter:
        print(color)


if __name__ == '__main__':
    main()
