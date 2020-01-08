""" Analyze ODE tolerances on SBML test suite

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2020-01-07
:Copyright: 2020, Karr Lab
:License: MIT
"""

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
import matplotlib
import matplotlib.pyplot as plt
import numpy
import os
import pandas

FILE = 'test_tols_all.txt'

def analyze(file):
    atols = set()
    rtols = set()
    cases = set()
    data = {}
    with open(file, 'r') as fh:
        for line in fh:
            fields = line.strip().split('\t')
            if len(fields) == 5:
                # atol	rtol	case_num	verified    run_time
                verified = fields[3]
                if verified in ['True', 'False']:
                    atol = float(fields[0])
                    rtol = float(fields[1])
                    atols.add(atol)
                    rtols.add(rtol)
                    case = fields[2]
                    cases.add(case)
                    verified = True if fields[3] == 'True' else False
                    run_time = float(fields[4])
                    data[(atol, rtol, case)] = (verified, run_time)

    # get fraction of cases verified vs. atol & rtol
    num_solved_array = numpy.zeros((len(atols), len(rtols)))    # shape: row, column
    num_solved_df = pandas.DataFrame(num_solved_array,
                                     index=sorted(atols),
                                     columns=sorted(rtols))
    for (atol, rtol, case), values in data.items():
        verified, run_time = values
        if verified:
            num_solved_df.at[atol, rtol] = num_solved_df.at[atol, rtol] + 1

    # get ODE run time
    compute_time = numpy.zeros((len(atols), len(rtols)))
    compute_time_df = pandas.DataFrame(compute_time,
                                     index=sorted(atols),
                                     columns=sorted(rtols))
    for (atol, rtol, _), values in data.items():
        verified, run_time = values
        if verified:
            compute_time_df.at[atol, rtol] = compute_time_df.at[atol, rtol] + run_time

    return len(cases), num_solved_df / len(cases), compute_time_df

# make heatmaps of tolerances vs. validation & compute time
def plot():
    file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'tests', 'testing',
                        'verification_results', 'ode_tuning', FILE)
    pathname = os.path.normpath(file)
    num_cases, fraction_solved_df, compute_time_df = analyze(pathname)

    columns = list(fraction_solved_df.columns)
    rows = list(fraction_solved_df.index)

    fig, axes = plt.subplots(1, 2)
    size = 6
    font = {'family' : 'normal',
            'size'   : size}
    matplotlib.rc('font', **font)
    im, cbar = heatmap(fraction_solved_df.values, rows, columns, ax=axes[0],
                       size=size, cmap="YlGn", cbarlabel="Fraction cases validated",
                       xlabel='rel-tol', ylabel='abs-tol',
                       title=f'Fraction of {num_cases} SBML test cases verified')
    texts = annotate_heatmap(im, size=size, valfmt="{x:.2f}")

    im, cbar = heatmap(compute_time_df.values, rows, columns, ax=axes[1],
                       size=size, cmap="RdBu", cbarlabel="Compute time (sec)",
                       xlabel='rel-tol', ylabel='abs-tol',
                       title=f'Total cpu time (sec) for {num_cases} cases')
    texts = annotate_heatmap(im, size=size, valfmt="{x:.0f}")

    fig.tight_layout()
    plot_file = os.path.join(os.path.dirname(pathname), FILE + '.pdf')
    fig.savefig(plot_file)
    plt.close(fig)
    print("Wrote: {}".format(plot_file))

# from https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html
def heatmap(data, row_labels, col_labels, ax=None, size=None,
            cbar_kw={}, cbarlabel="", xlabel=None, ylabel=None, title=None, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(numpy.arange(data.shape[1]))
    ax.set_yticks(numpy.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    if size:
        ax.tick_params(axis='both', which='major', labelsize=size)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(numpy.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(numpy.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, numpy.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

plot()
