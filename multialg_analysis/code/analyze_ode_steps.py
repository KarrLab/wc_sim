""" Code for plotting ODE integration step data

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2019-11-20
:Copyright: 2019, Karr Lab
:License: MIT
"""

from pprint import pprint
import pandas
from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages
import os

def mk_floats(row):
    new_row = []
    for v in row:
        try:
            new_row.append(float(v))
        except:
            new_row.append(v)
    return new_row

# get data
def get_data():
    datafile = os.path.join(os.path.dirname(__file__), '..', 'data', 'run_ode_solver_w_internal.tsv')
    with open(datafile, 'r') as fh:
        while(True):
            line = fh.readline()   # discard header lines
            if 'mode\ttime' in line:
                break
        headers = line.strip().split('\t')
        rows = []
        while(True):
            line = fh.readline()   # get data lines
            if 'end data' in line:
                break
            float_line = mk_floats(line.strip().split('\t'))
            row = dict(zip(headers, float_line
            ))
            rows.append(row)
    data = pandas.DataFrame(rows)
    return data

def plot(data):
    # plotting options
    pyplot.rc('font', size=6)
    fig = pyplot.figure()
    fig.suptitle(f'Details of ODE solution of [compt_1]: spec_type_0 ==> spec_type_1 @ k * spec_type_0 / Avogadro / volume_compt_1')
    axes = fig.add_subplot(1, 1, 1)
    modes = ['internal step', 'external step']
    linewidth = 0.5
    markersize = 2
    internal_plot_kwargs = dict(color='blue', linewidth=linewidth, markersize=markersize,
                                marker='o', label='internal step')
    external_plot_kwargs = dict(color='red', linewidth=linewidth, markersize=markersize,
                                marker='+', label='external step')
    plot_kwargs = dict(zip(modes, [internal_plot_kwargs, external_plot_kwargs]))

    species = 'population_1'
    for mode in modes:
        step_data = data[data['mode'] == mode]
        times = step_data['time']
        pop = step_data[species]
        axes.plot(times, pop, **plot_kwargs[mode])
        axes.set_xlabel('Time (s)')
        axes.set_ylabel(f'{species} (copy number)')
        axes.legend()

    plots_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'plots')
    figure_name = f'ode_modes_species_trajectories'
    filename = os.path.join(plots_dir, figure_name + '.pdf')
    fig.savefig(filename)
    print(f"wrote '{filename}'")
    pyplot.close(fig)

data = get_data()
plot(data)
