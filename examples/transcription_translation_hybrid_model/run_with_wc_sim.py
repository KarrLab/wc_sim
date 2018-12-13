""" Use wc_sim to simulate a model of the transcription,
translation, and RNA and protein degradation of 3 genes, each
with 1 RNA and 1 protein species

:Author: Jonathan Karr <jonrkarr@gmail.com>
:Date: 2018-10-23
:Copyright: 2018, Karr Lab
:License: MIT
"""

from matplotlib import pyplot
from wc_sim.multialgorithm.simulation import Simulation
from wc_sim.multialgorithm.run_results import RunResults
import numpy
import wc_lang.io

model_filename = 'model.xlsx'
results_parent_dirname = 'results'
checkpoint_period = 100.

# read model
model = wc_lang.io.Reader().run(model_filename)

# run simulation
sim = Simulation(model)
_, results_dirname = sim.run(end_time=model.parameters.get_one(id='cell_cycle_len').value,
                             results_dir=results_parent_dirname,
                             checkpoint_period=checkpoint_period)
results = RunResults(results_dirname)

# plot results


def plot(model, results, filename):
    # get expected results
    c = model.compartments.get_one(id='c')

    species_type = model.species_types.get_one(id='rna')
    species = species_type.species.get_one(compartment=c)
    exp_avg_rna = species.concentration.value

    species_type = model.species_types.get_one(id='prot')
    species = species_type.species.get_one(compartment=c)
    exp_avg_prot = species.concentration.value

    # get simulation results
    pops = results.get('populations')
    time = pops.index
    rna = pops['rna[c]']
    prot = pops['prot[c]']

    states = results.get('aggregate_states')
    mass = states['c']['mass']
    vol = states['c']['volume']

    # initialize figure
    fig1, axes1 = pyplot.subplots(nrows=2, ncols=1)
    fig2, axes2 = pyplot.subplots(nrows=2, ncols=1)

    # plot RNA predictions
    avg = numpy.mean(rna)
    axes1[0].plot(time / 3600, rna, label='<{0:.1f}>'.format(avg))
    axes1[0].plot([0, time[-1] / 3600], [avg, avg], label='Avg: {}'.format(exp_avg_rna))
    axes1[0].set_xlim((time[0] / 3600, time[-1] / 3600))
    axes1[0].set_ylim((0., 10.0))
    #axes1[0].set_xlabel('Time (h)')
    axes1[0].set_ylabel('RNA (molecules)')
    axes1[0].legend(loc='upper right')

    # plot protein predictions
    avg = numpy.mean(prot)
    axes1[1].plot(time / 3600, prot, label='<{0:.0f}>'.format(avg))
    axes1[1].plot([0, time[-1] / 3600], [avg, avg], label='Avg: {}'.format(exp_avg_prot))
    axes1[1].set_xlim((time[0] / 3600, time[-1] / 3600))
    axes1[1].set_ylim((1300., 1700.))
    axes1[1].set_xlabel('Time (h)')
    axes1[1].set_ylabel('Protein (molecules)')
    axes1[1].legend(loc='upper right')

    # plot mass
    axes2[0].plot(time / 3600, mass * 1e15)
    axes2[0].set_xlim((time[0] / 3600, time[-1] / 3600))
    axes2[0].set_ylim((51.1-0.01, 51.1+0.01))
    axes2[0].ticklabel_format(useOffset=False)
    #axes2[0].set_xlabel('Time (h)')
    axes2[0].set_ylabel('Mass (fg)')

    # plot volume
    axes2[1].plot(time / 3600, vol * 1e18)
    axes2[1].set_xlim((time[0] / 3600, time[-1] / 3600))
    axes2[1].set_ylim((67-0.01, 67+0.01))
    axes2[1].ticklabel_format(useOffset=False)
    axes2[1].set_xlabel('Time (h)')
    axes2[1].set_ylabel('Volume (aL)')

    # save and close figure
    fig1.savefig(filename.format('species'))
    fig2.savefig(filename.format('mass'))
    pyplot.close(fig1)
    pyplot.close(fig2)


plot(model, results, 'wc_sim_results-{}.pdf')
