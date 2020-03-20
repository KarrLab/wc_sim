""" Use wc_sim to simulate a model of the translation
and metabolism of a bacterium

:Author: Jonathan Karr <karr@mssm.edu>
:Date: 2018-10-23
:Copyright: 2018, Karr Lab
:License: MIT
"""

from matplotlib import pyplot
from wc_sim.simulation import Simulation
from wc_sim.run_results import RunResults
import numpy
import wc_lang
import wc_lang.io

model_filename = 'model.xlsx'
results_parent_dirname = 'results'
checkpoint_period = 100.
time_max = 3600. * 8.

# read model
model = wc_lang.io.Reader().run(model_filename)[wc_lang.Model][0]

# run simulation
seed = 100
sim = Simulation(model)
_, results_dirname = sim.run(time_max=time_max,
                             seed=seed,
                             results_dir=results_parent_dirname,
                             checkpoint_period=checkpoint_period)
results = RunResults(results_dirname)

# plot results


def plot(model, results, filename):
    # get expected results
    mean_doubling_time = model.parameters.get_one(id='mean_doubling_time').value

    cytosol = model.compartments.get_one(id='c')
    extracellular_space = model.compartments.get_one(id='e')

    species_type = model.species_types.get_one(id='prot')
    species = species_type.species.get_one(compartment=cytosol)
    exp_avg_prot = species.concentration.value

    # get simulation results
    pops = results.get('populations')
    time = pops.index
    ala_e = pops['ala[e]']
    h2o_e = pops['h2o[e]']
    ala_c = pops['ala[c]']
    h2o_c = pops['h2o[c]']
    prot_c = pops['prot[c]']
    cell_100_ag = pops['cell_100_ag[c]']

    states = results.get('aggregate_states')
    mass = states['c']['mass']
    vol_e = 1e-12
    vol_c = states['c']['volume']

    # initialize figure
    fig1, axes1 = pyplot.subplots(nrows=3, ncols=2)
    fig2, axes2 = pyplot.subplots(nrows=3, ncols=1)

    # plot alanine predictions
    axes1[0][0].plot(time / 3600, ala_e / vol_e / 6.022e23 * 1e3)
    axes1[0][0].set_xlim((time[0] / 3600, time[-1] / 3600))
    axes1[0][0].set_ylim((10.0-0.1, 10+0.1))
    axes1[0][0].ticklabel_format(useOffset=False)
    #axes1[0][0].set_xlabel('Time (h)')
    axes1[0][0].set_ylabel('Ext Ala (mM)')

    axes1[0][1].plot(time / 3600, ala_c / (cell_100_ag * 6.71e-17 / 511) / 6.022e23 * 1e3)
    axes1[0][1].set_xlim((time[0] / 3600, time[-1] / 3600))
    #axes1[0][1].set_ylim((10.0-0.1, 10+0.1))
    axes1[0][1].ticklabel_format(useOffset=False)
    #axes1[0][1].set_xlabel('Time (h)')
    axes1[0][1].set_ylabel('Cyt Ala (mM)')

    # plot water predictions
    axes1[1][0].plot(time / 3600, h2o_e / vol_e / 6.022e23 * 1e0)
    axes1[1][0].set_xlim((time[0] / 3600, time[-1] / 3600))
    axes1[1][0].set_ylim((55.0-0.1, 55+0.1))
    axes1[1][0].ticklabel_format(useOffset=False)
    #axes1[1][0].set_xlabel('Time (h)')
    axes1[1][0].set_ylabel('Ext water (M)')

    axes1[1][1].plot(time / 3600, h2o_c / (cell_100_ag * 6.71e-17 / 511) / 6.022e23 * 1e0)
    axes1[1][1].set_xlim((time[0] / 3600, time[-1] / 3600))
    #axes1[1][1].set_ylim((1300., 1700.))
    axes1[1][1].ticklabel_format(useOffset=False)
    #axes1[1][1].set_xlabel('Time (h)')
    axes1[1][1].set_ylabel('Cyt water (M)')

    # plot protein predictions
    axes1[2][0].set_visible(False)

    axes1[2][1].plot(time / 3600, prot_c, label='Sim')
    axes1[2][1].plot(time / 3600, 1500 * numpy.exp(numpy.log(2) / mean_doubling_time * time), label='Exp')
    axes1[2][1].set_xlim((time[0] / 3600, time[-1] / 3600))
    #axes1[2][1].set_ylim((1300., 3400.))
    axes1[2][1].set_xlabel('Time (h)')
    axes1[2][1].set_ylabel('Protein (molecules)')
    axes1[2][1].legend(loc='upper left')

    # plot cell mass
    axes2[0].plot(time / 3600, cell_100_ag / 10)
    axes2[0].set_xlim((time[0] / 3600, time[-1] / 3600))
    axes2[0].set_ylim((45, 105))
    axes2[0].ticklabel_format(useOffset=False)
    #axes2[0].set_xlabel('Time (h)')
    axes2[0].set_ylabel('Cell mass (fg)')

    # plot mass
    axes2[1].plot(time / 3600, mass * 1e15)
    axes2[1].set_xlim((time[0] / 3600, time[-1] / 3600))
    #axes2[1].set_ylim((51.1-0.01, 51.1+0.01))
    axes2[1].ticklabel_format(useOffset=False)
    #axes2[1].set_xlabel('Time (h)')
    axes2[1].set_ylabel('Cyt mass (fg)')

    # plot volume
    axes2[2].plot(time / 3600, vol_c * 1e18)
    axes2[2].set_xlim((time[0] / 3600, time[-1] / 3600))
    axes2[2].set_ylim((67-0.01, 67+0.01))
    axes2[2].ticklabel_format(useOffset=False)
    axes2[2].set_xlabel('Time (h)')
    axes2[2].set_ylabel('Cyt volume (aL)')

    # save and close figure
    fig1.savefig(filename.format('species'))
    fig2.savefig(filename.format('mass'))
    pyplot.close(fig1)
    pyplot.close(fig2)


plot(model, results, 'results-{}.pdf')
