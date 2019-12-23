""" Use custom code to simulate a model of the transcription,
translation, and RNA and protein degradation of 3 genes, each
with 1 RNA and 1 protein species

:Author: Jonathan Karr <karr@mssm.edu>
:Date: 2018-10-23
:Copyright: 2018, Karr Lab
:License: MIT
"""

#!/usr/local/bin/python3.6

from matplotlib import pyplot
import copy
import numpy
import scipy.integrate

rxns = {
    'transcription': {
        'parts': [{'species': 'rna', 'coefficient': 1}],
        'rate_law': 'k_transcription',
        'framework': 'ssa',
    },
    'translation': {
        'parts': [{'species': 'prot', 'coefficient': 1}],
        'rate_law': 'k_translation * (rna / (km_translation + rna))',
        'framework': 'ode',
    },
    'rna_deg': {
        'parts': [{'species': 'rna', 'coefficient': -1}],
        'rate_law': 'k_rna_deg * rna',
        'framework': 'ssa',
    },
    'prot_deg': {
        'parts': [{'species': 'prot', 'coefficient': -1}],
        'rate_law': 'k_prot_deg * prot',
        'framework': 'ode',
    },
}

rate_params = {
    'k_transcription': 2. * numpy.log(2) / (3 * 60),
    'k_translation': 1500 * 2. * numpy.log(2) / (18 * 60 * 60),
    'km_translation': 2.,
    'k_rna_deg': numpy.log(2) / (3 * 60),
    'k_prot_deg': numpy.log(2) / (18 * 60 * 60),
}

init_species = {
    'rna': 1,
    'prot': 1375,
}

exp_avg_species = {
    'rna': 2,
    'prot': 1500,
}

init_time = 0.
time_max = 8 * 60 * 60.
checkpoint_period = 1.

numpy.random.seed(seed=0)


def sim_ssa(rxns, rate_params, init_species, init_time, time_max, checkpoint_period):
    """ Simulate model with SSA

    Args:
        rxns (:obj:`dict`): reaction participants and rate laws
        rate_params (:obj:`dict`): values of parameters of rate laws
        init_species (:obj:`dict` of :obj:`int`):
            initial copy numbers of RNAs and proteins
        init_time (:obj:`float`): initial time (seconds)
        time_max (:obj:`float`): simulation length (seconds)
        checkpoint_period (:obj:`float`): interval to log simulation results (seconds)

    Returns:
        :obj:`dict` of :obj:`numpy.ndarray` of :obj:`int`: copy number dynamics of RNAs and proteins
        :obj:`numpy.ndarray` of :obj:`float`: time (seconds)
    """
    rxns = list(rxns.values())

    props = numpy.full((len(rxns),), numpy.nan)
    species = copy.deepcopy(init_species)
    time = init_time

    n_log = int((time_max - init_time) / checkpoint_period)
    species_log = {}
    time_log = numpy.full((n_log, ), init_time)
    for species_id, init_val in species.items():
        species_log[species_id] = numpy.full((n_log, ), init_val)

    while time < time_max:
        locals = {**species, **rate_params}
        props = [eval(rxn['rate_law'], locals) for rxn in rxns]
        tot_prop = numpy.sum(props)
        if tot_prop == 0:
            break

        dt = numpy.random.exponential(1 / tot_prop)
        time += dt
        i_log = int(numpy.ceil((time - init_time) / checkpoint_period))
        time_log[i_log:] = time

        rxn = numpy.random.choice(rxns, p=props / tot_prop)
        for part in rxn['parts']:
            species[part['species']] += part['coefficient']
            species_log[part['species']][i_log:] = species[part['species']]

    return (species_log, time_log)


def sim_ode(rxns, rate_params, init_species, init_time, time_max, checkpoint_period):
    """ Simulate model with ODE integration

    Args:
        rxns (:obj:`dict`): reaction participants and rate laws
        rate_params (:obj:`dict`): values of parameters of rate laws
        init_species (:obj:`dict` of :obj:`int`):
            initial copy numbers of RNAs and proteins
        init_time (:obj:`float`): initial time (seconds)
        time_max (:obj:`float`): simulation length (seconds)
        checkpoint_period (:obj:`float`): interval to log simulation results (seconds)

    Returns:
        :obj:`dict` of :obj:`numpy.ndarray` of :obj:`int`: copy number dynamics of RNAs and proteins
        :obj:`numpy.ndarray` of :obj:`float`: time (seconds)
    """
    time_log = numpy.linspace(init_time, time_max, num=int((time_max-init_time)/checkpoint_period) + 1)
    species_ids = list(init_species.keys())
    init_ys = numpy.array([init_species[species_id] for species_id in species_ids])

    def func(ys, time, species_ids, rxns, rate_params):
        species = {species_id: y for species_id, y in zip(species_ids, ys)}
        locals = {**species, **rate_params}
        dys = numpy.zeros(ys.shape)
        for rxn in rxns.values():
            rate = eval(rxn['rate_law'], locals)
            for part in rxn['parts']:
                i_species = species_ids.index(part['species'])
                dys[i_species] += rate * part['coefficient']
        return dys

    ys_log = scipy.integrate.odeint(func, init_ys, time_log, args=(species_ids, rxns, rate_params))
    species_log = {species_id: ys_log[:, i_species] for i_species, species_id in enumerate(species_ids)}

    return (species_log, time_log)


def sim_hyb(rxns, rate_params, init_species, init_time, time_max, checkpoint_period):
    """ Simulate model with SSA/ODE hybrid

    Args:
        rxns (:obj:`dict`): reaction participants and rate laws
        rate_params (:obj:`dict`): values of parameters of rate laws
        init_species (:obj:`dict` of :obj:`int`):
            initial copy numbers of RNAs and proteins
        init_time (:obj:`float`): initial time (seconds)
        time_max (:obj:`float`): simulation length (seconds)
        checkpoint_period (:obj:`float`): interval to log simulation results (seconds)

    Returns:
        :obj:`dict` of :obj:`numpy.ndarray` of :obj:`int`: copy number dynamics of RNAs and proteins
        :obj:`numpy.ndarray` of :obj:`float`: time (seconds)
    """
    rxns_ssa = []
    rxns_ode = {}
    for rxn_id, rxn in rxns.items():
        if rxn['framework'] == 'ssa':
            rxns_ssa.append(rxn)
        else:
            rxns_ode[rxn_id] = rxn

    props = numpy.full((len(rxns_ssa),), numpy.nan)
    species = copy.deepcopy(init_species)
    time = init_time

    n_log = int((time_max - init_time) / checkpoint_period)
    species_log = {}
    time_log = numpy.full((n_log, ), init_time)
    for species_id, init_val in species.items():
        species_log[species_id] = numpy.full((n_log, ), init_val)

    while time < time_max:
        # select and fire next SSA reaction
        locals = {**species, **rate_params}
        props = [eval(rxn['rate_law'], locals) for rxn in rxns_ssa]
        tot_prop = numpy.sum(props)
        if tot_prop == 0:
            break

        time_step = numpy.random.exponential(1 / tot_prop)
        time += time_step
        i_log = int(numpy.ceil((time - init_time) / checkpoint_period))
        time_log[i_log:] = time

        rxn = numpy.random.choice(rxns_ssa, p=props / tot_prop)
        for part in rxn['parts']:
            species[part['species']] += part['coefficient']
            species_log[part['species']][i_log:] = species[part['species']]

        # interpolate ODE
        ode_species_log, _ = sim_ode(rxns_ode, rate_params, species, 0, time_step, time_step)
        for species_id in species.keys():
            species[species_id] = ode_species_log[species_id][-1]
            species_log[species_id][i_log:] = species[species_id]

    return (species_log, time_log)


def plot(
        species_ssa, time_ssa,
        species_ode, time_ode,
        species_hyb, time_hyb,
        exp_avg_species, filename):
    fig, axes = pyplot.subplots(nrows=2, ncols=1)

    # RNA
    avg = numpy.mean(species_ssa['rna'])
    axes[0].plot(time_ssa / 3600, species_ssa['rna'], label='SSA <{0:.1f}>'.format(avg))

    avg = numpy.mean(species_ode['rna'])
    axes[0].plot(time_ode / 3600, species_ode['rna'], label='ODE <{0:.1f}>'.format(avg))

    avg = numpy.mean(species_hyb['rna'])
    axes[0].plot(time_hyb / 3600, species_hyb['rna'], label='Hybrid <{0:.1f}>'.format(avg))

    axes[0].plot([0, time_ssa[-1] / 3600], [avg, avg], label='Avg: {}'.format(exp_avg_species['rna']))
    axes[0].set_xlim((time_ssa[0] / 3600, time_ssa[-1] / 3600))
    axes[0].set_ylim((0., 10.0))
    #axes[0].set_xlabel('Time (h)')
    axes[0].set_ylabel('RNA (molecules)')
    axes[0].legend(loc='upper right')

    # Protein
    avg = numpy.mean(species_ssa['prot'])
    axes[1].plot(time_ssa / 3600, species_ssa['prot'], label='SSA <{0:.0f}>'.format(avg))

    avg = numpy.mean(species_ode['prot'])
    axes[1].plot(time_ode / 3600, species_ode['prot'], label='ODE <{0:.0f}>'.format(avg))

    avg = numpy.mean(species_hyb['prot'])
    axes[1].plot(time_hyb / 3600, species_hyb['prot'], label='Hybrid <{0:.0f}>'.format(avg))

    axes[1].plot([0, time_ssa[-1] / 3600], [avg, avg], label='Avg: {}'.format(exp_avg_species['prot']))
    axes[1].set_xlim((time_ssa[0] / 3600, time_ssa[-1] / 3600))
    axes[1].set_ylim((1300., 1700.))
    axes[1].set_xlabel('Time (h)')
    axes[1].set_ylabel('Protein (molecules)')
    axes[1].legend(loc='upper right')

    fig.savefig(filename)
    pyplot.close(fig)


ssa_species_log, ssa_time_log = sim_ssa(rxns, rate_params, init_species, init_time, time_max, checkpoint_period)
ode_species_log, ode_time_log = sim_ode(rxns, rate_params, init_species, init_time, time_max, checkpoint_period)
hyb_species_log, hyb_time_log = sim_hyb(rxns, rate_params, init_species, init_time, time_max, checkpoint_period)
plot(ssa_species_log, ssa_time_log,
     ode_species_log, ode_time_log,
     hyb_species_log, hyb_time_log,
     exp_avg_species, 'custom_results.pdf')
