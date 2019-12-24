""" Utilities for testing

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2019-10-26
:Copyright: 2019, Karr Lab
:License: MIT
"""

from collections import defaultdict
from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages
import math
import numpy as np
import os
import tempfile
import time

from wc_lang import Model
from wc_lang.io import Reader
from wc_onto import onto
from wc_sim.dynamic_components import DynamicModel
from wc_sim.run_results import RunResults
from wc_sim.simulation import Simulation
import obj_tables
import wc_lang


def read_model_for_test(model_filename, set_std_devs_to_0=True, integration_framework=None):
    """ Read a model and prepare it for tests

    Args:
        model_filename (:obj:`str`): `wc_lang` model file
        set_std_devs_to_0 (:obj:`bool`, optional): if set, set all standard deviations in distributions
            to 0
        integration_framework (:obj:`str`): if provided, set all submodels to this integration framework

    Returns:
        :obj:`Model`: a whole-cell model
    """
    # read model while ignoring missing models
    all_models = Reader().run(model_filename, ignore_extra_models=True)
    if set_std_devs_to_0:
        # set all standard deviations to 0
        models_with_std_devs = (wc_lang.InitVolume,
                                wc_lang.Ph,
                                wc_lang.DistributionInitConcentration,
                                wc_lang.Parameter,
                                wc_lang.Observation,
                                wc_lang.Conclusion)
        for model, instances in all_models.items():
            if model in models_with_std_devs:
                for instance in instances:
                    instance.std = 0
    test_wc_model = all_models[Model][0]
    if integration_framework:
        for submodel in test_wc_model.submodels:
            submodel.framework = onto[integration_framework]
    return test_wc_model


def check_simul_results(test_case, dynamic_model, results_dir, integration_framework=None,
                        expected_initial_values=None,
                        expected_times=None, expected_species_trajectories=None,
                        expected_property_trajectories=None, rel_tol=1e-7):
    """ Evaluate whether a simulation predicted the expected results

    The expected trajectories must contain expected values at the times of the simulation's checkpoints.

    Args:
        test_case (:obj:`unittest.TestCase`): the `test_case`
        dynamic_model (:obj:`DynamicModel`): the simulation's :obj:`DynamicModel`
        results_dir (:obj:`str`): simulation results directory
        integration_framework (:obj:`str`, optional): the integration framework used to simulate the model
        expected_initial_values (:obj:`dict`, optional): expected initial values for each compartment
            in the test; indexed by compartment id, and attribute names for :obj:`DynamicCompartment`\ s
        expected_times (:obj:`iterator`, optional): expected time sequence for all trajectories
        expected_species_trajectories (:obj:`dict`, optional): expected trajectories of species
            copy numbers in the test; provides an iterator for each species, indexed by the species id
        expected_property_trajectories (:obj:`dict`, optional): expected trajectories of aggregate properties
             for each compartment in the test; provides an iterator for each property; indexed by
             compartment id. Properties must be keyed as structured by `AccessState.get_checkpoint_state()`
             and `RunResults.convert_checkpoints()`
        rel_tol (:obj:`float`, optional): simulation predictions are compared approximately; `rel_tol` is
            relative tolerance, as used in `math.isclose()` and `numpy.testing.assert_allclose()`;
            default `rel_tol` is `1e-7`; absolute tolerance is set to 0
    """

    # tolerances
    rtol = rel_tol
    atol = 0
    if integration_framework:
        tolerances = {
            'deterministic_simulation_algorithm': {
                'rtol': 1e-7,
                'atol': 0
            },
            'ordinary_differential_equations': {
                'rtol': 5e-3,
                'atol': 10
            }
        }
        rtol = tolerances[integration_framework]['rtol']
        atol = tolerances[integration_framework]['atol']

    model_and_integration = f"In model {dynamic_model.id}, integrated by {integration_framework},"

    # test initial values
    if expected_initial_values:
        for compartment_id in expected_initial_values:
            dynamic_compartment = dynamic_model.dynamic_compartments[compartment_id]
            for initial_attribute, expected_value in expected_initial_values[compartment_id].items():
                simulated_value = getattr(dynamic_compartment, initial_attribute)
                test_case.assertTrue(math.isclose(simulated_value, expected_value, rel_tol=rtol),
                                     msg=f"{model_and_integration} {initial_attribute} "
                                     f"{simulated_value}, is not within {rtol} of {expected_value}")

    # get results
    if expected_species_trajectories or expected_property_trajectories:
        run_results = RunResults(results_dir)
        populations_df = run_results.get('populations')
        aggregate_states_df = run_results.get('aggregate_states')

    # test expected_times
    if expected_times and not np.isnan(expected_times).all():
        np.testing.assert_array_equal(populations_df.index, expected_times,
                                      err_msg=f"{model_and_integration} time sequence for populations "
                                      f"differs from expected time sequence:")
        np.testing.assert_array_equal(aggregate_states_df.index, expected_times,
                                      err_msg=f"{model_and_integration} time sequence for aggregate "
                                      f"states differs from expected time sequence:")

    # test expected trajectories of species
    if expected_species_trajectories:
        for species_id, expected_trajectory in expected_species_trajectories.items():
            if np.isnan(expected_trajectory).all():
                continue
            simulated_trajectory = populations_df[species_id]
            np.testing.assert_allclose(simulated_trajectory, expected_trajectory, equal_nan=False,
                                       rtol=rtol,
                                       atol=atol,
                                       err_msg=f"{model_and_integration} trajectory for {species_id} "
                                           f"not almost equal to expected trajectory:")

    # test expected trajectories of properties of compartments
    if expected_property_trajectories:
        # get all properties
        properties = set()
        for property_array_dict in expected_property_trajectories.values():
            properties.update(property_array_dict.keys())
        # check property trajectories
        for compartment_id in expected_property_trajectories:
            for property in properties:
                if compartment_id not in aggregate_states_df:
                    continue
                simulated_trajectory = run_results.get_properties(compartment_id, property)
                expected_trajectory = expected_property_trajectories[compartment_id][property]
                if np.isnan(expected_trajectory).all():
                    continue
                # todo: investigate possible numpy bug: without list(), this fails with
                # "TypeError: ufunc 'isfinite' not supported for the input types, ..." but ndarray works elsewhere
                # todo: when fixed, remove list(), and below too
                np.testing.assert_allclose(list(simulated_trajectory.to_numpy()), expected_trajectory,
                                           equal_nan=False,
                                           rtol=rtol,
                                           atol=atol,
                                           err_msg=f"{model_and_integration} trajectory of {property} "
                                               f"of compartment {compartment_id} "
                                               f"not almost equal to expected trajectory:")
        # confirm that compartment densities are constants
        density_ratios = (('mass', 'volume'),
                          ('accounted mass', 'accounted volume'))
        for mass_property, volume_property in density_ratios:
            if {mass_property, volume_property} <= properties:
                for compartment_id in expected_property_trajectories:
                    if compartment_id not in aggregate_states_df:
                        continue
                    mass_trajectory = run_results.get_properties(compartment_id, mass_property).to_numpy()
                    volume_trajectory = run_results.get_properties(compartment_id, volume_property).to_numpy()
                    density_trajectory = mass_trajectory / volume_trajectory
                    constant_density = np.full_like(density_trajectory, density_trajectory[0])
                    err_msg = (f"{model_and_integration} density trajectory for "
                        f"'{mass_property}' / '{volume_property}' of compartment {compartment_id} "
                        f"is not constant:")
                    np.testing.assert_allclose(list(density_trajectory),
                                               list(constant_density),
                                               equal_nan=False,
                                               rtol=rtol,
                                               atol=atol,
                                               err_msg=err_msg)


def species_id_to_pop_attr(species_id):
    # obtain an attribute name for the population of a species
    return f"pop_{species_id.replace('[', '_').replace(']', '_')}"


def define_trajectory_classes(model):
    # define the expected trajectory classes for model, SpeciesTrajectory & AggregateTrajectory
    # based on init_schema from obj_tables.utils
    # could put a list of species to ignore in Parameters, or, better yet, in a TestConfig worksheet

    # SpeciesTrajectory
    cls_specs = {}
    cls_name = 'SpeciesTrajectory'
    cls_specs[cls_name] = cls = {
        'name': cls_name,
        'attrs': {},
        'tab_format': obj_tables.TableFormat.row,
        'verbose_name': cls_name,
        'verbose_name_plural': 'SpeciesTrajectories',
        'desc': 'Trajectories of species populations',
    }
    # attributes: time, population for each species
    cls['attrs']['time'] = {
        'name': 'time',
        'type': obj_tables.FloatAttribute()
    }
    for species in model.get_species():
        attr_name = species_id_to_pop_attr(species.id)
        cls['attrs'][attr_name] = {
            'name': attr_name,
            'type': obj_tables.FloatAttribute()
        }

    # AggregateTrajectory
    cls_name = 'AggregateTrajectory'
    cls_specs[cls_name] = cls = {
        'name': cls_name,
        'attrs': {},
        'tab_format': obj_tables.TableFormat.row,
        'verbose_name': cls_name,
        'verbose_name_plural': 'AggregateTrajectories',
        'desc': 'Trajectories of aggregate quantities',
    }
    # attributes: time, cell aggregates, aggregates for compartments
    cls['attrs']['time'] = {
        'name': 'time',
        'type': obj_tables.FloatAttribute()
    }
    for aggregate_value in DynamicModel.AGGREGATE_VALUES:
        attr_name = f'cell {aggregate_value}'
        attr_name.replace(' ', '_')
        cls['attrs'][attr_name] = {
            'name': attr_name,
            'type': obj_tables.FloatAttribute()
        }
    for compartment in model.get_compartments():
        # ignore compartments that aren't cellular
        if compartment.biological_type == onto['WC:cellular_compartment']:
            for aggregate_value in DynamicModel.AGGREGATE_VALUES:
                attr_name = f'compartment {compartment.id} {aggregate_value}'
                attr_name.replace(' ', '_')
                cls['attrs'][attr_name] = {
                    'name': attr_name,
                    'type': obj_tables.FloatAttribute()
                }

    # create the classes
    trajectory_classes = {}
    for cls_spec in cls_specs.values():
        meta_attrs = {
            'table_format': cls_spec['tab_format'],
            'description': cls_spec['desc'],
        }
        if cls_spec['verbose_name']:
            meta_attrs['verbose_name'] = cls_spec['verbose_name']
        if cls_spec['verbose_name_plural']:
            meta_attrs['verbose_name_plural'] = cls_spec['verbose_name_plural']

        attrs = {
            '__doc__': cls_spec['desc'],
            'Meta': type('Meta', (obj_tables.Model.Meta, ), meta_attrs),
        }
        for attr_spec in cls_spec['attrs'].values():
            attr = attr_spec['type']
            attrs[attr_spec['name']] = attr

        cls = type(cls_spec['name'], (obj_tables.Model, ), attrs)
        trajectory_classes[cls_spec['name']] = cls

    return(trajectory_classes)


def verify_independently_solved_model(test_case, model_filename, results_dir):

    # read model while ignoring missing models, with std dev = 0
    for integration_framework in ['deterministic_simulation_algorithm', 'ordinary_differential_equations']:
        # empty results_dir
        for file in os.listdir(results_dir):
            file_path = os.path.join(results_dir, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)

        model = read_model_for_test(model_filename, integration_framework=f'WC:{integration_framework}')

        # simulate model
        end_time = model.parameters.get_one(id='end_time').value
        checkpoint_period = model.parameters.get_one(id='checkpoint_period').value
        args = dict(results_dir=results_dir,
                    checkpoint_period=checkpoint_period,
                    ode_time_step=1.)
        simulation = Simulation(model)
        start_time = time.perf_counter()
        _, results_dir = simulation.run(end_time=end_time, **args)
        elapsed_rt = time.perf_counter() - start_time
        print(f'ran {os.path.basename(model_filename)} with {integration_framework} in '
              f'{elapsed_rt:.2e} (sec)')

        # test dynamics
        # read expected trajectories
        trajectories = define_trajectory_classes(model)
        SpeciesTrajectory = trajectories['SpeciesTrajectory']
        AggregateTrajectory = trajectories['AggregateTrajectory']
        trajectory_model_classes = list(trajectories.values())
        expected_trajectories = \
            obj_tables.io.Reader().run(model_filename, models=trajectory_model_classes,
                                       ignore_extra_models=True, ignore_attribute_order=True)

        # get expected species trajectories from model workbook
        expected_species_trajectories = {}
        species_ids = [species.id for species in model.get_species()]
        for species_id in species_ids:
            expected_species_trajectories[species_id] = []
        for species_id in species_ids:
            for expected_species_trajectory in expected_trajectories[SpeciesTrajectory]:
                expected_pop = getattr(expected_species_trajectory, species_id_to_pop_attr(species_id))
                expected_species_trajectories[species_id].append(expected_pop)
        expected_trajectory_times = []
        for expected_species_trajectory in expected_trajectories[SpeciesTrajectory]:
            expected_trajectory_times.append(expected_species_trajectory.time)

        # plot trajectories
        plots_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'results'))
        os.makedirs(plots_dir, exist_ok=True)
        plots = plot_expected_vs_simulated(simulation.dynamic_model,
                                           integration_framework,
                                           results_dir,
                                           trajectory_times=expected_trajectory_times,
                                           plots_dir=plots_dir,
                                           expected_species_trajectories=expected_species_trajectories)

        # get aggregate trajectories from model workbook
        expected_aggregate_trajectories = {}
        for dyn_compartment_id in simulation.dynamic_model.dynamic_compartments:
            expected_aggregate_trajectories[dyn_compartment_id] = defaultdict(list)
        expected_trajectory_times = []
        for attr in AggregateTrajectory.Meta.local_attributes.values():
            attr_name = attr.name
            if attr_name != 'time':
                if attr_name.startswith('compartment'):
                    compartment_id = attr_name.split(' ')[1]
                    for expected_aggregate_trajectory in expected_trajectories[AggregateTrajectory]:
                        aggregate_prop = ' '.join(attr_name.split(' ')[2:])
                        expected_aggregate_trajectories[compartment_id][aggregate_prop].append(
                            getattr(expected_aggregate_trajectory, attr.name))
            else:
                for expected_aggregate_trajectory in expected_trajectories[AggregateTrajectory]:
                    expected_trajectory_times.append(expected_aggregate_trajectory.time)

        # plot trajectories
        plots = plot_expected_vs_simulated(simulation.dynamic_model,
                                        integration_framework,
                                        results_dir,
                                        trajectory_times=expected_trajectory_times,
                                        plots_dir=plots_dir,
                                        expected_property_trajectories=expected_aggregate_trajectories)

        # compare expected & simulated trajectories
        check_simul_results(test_case,
                            simulation.dynamic_model,
                            results_dir,
                            integration_framework,
                            expected_times=expected_trajectory_times,
                            expected_species_trajectories=expected_species_trajectories,
                            expected_property_trajectories=expected_aggregate_trajectories)

def plot_expected_vs_simulated(dynamic_model,
                            integration_algorithm,
                            results_dir,
                            trajectory_times,
                            plots_dir=None,
                            expected_species_trajectories=None,
                            expected_property_trajectories=None):
    """ Plot expected vs. simulated trajectories

    Args:
        dynamic_model (:obj:`DynamicModel`): the simulation's :obj:`DynamicModel`
        integration_algorithm (:obj:`str`): the integration algorithm
        results_dir (:obj:`str`): simulation results directory
        trajectory_times (:obj:`iterator`): time sequence for all trajectories
        plots_dir (:obj:`str`, optional): directory for plot files
        expected_species_trajectories (:obj:`dict`, optional): expected trajectories of species
            copy numbers in the test; provides an iterator for each species, indexed by the species id
        expected_property_trajectories (:obj:`dict`, optional): expected trajectories of aggregate properties
             for each compartment in the test; provides an iterator for each property; indexed by
             compartment id. Properties must be keyed as structured by `AccessState.get_checkpoint_state()`
             and `RunResults.convert_checkpoints()`

    Returns:
        :obj:`str`: the directory containing plots
    """
    model_id = dynamic_model.id
    run_results = RunResults(results_dir)
    populations_df = run_results.get('populations')
    aggregate_states_df = run_results.get('aggregate_states')

    if not plots_dir:
        # make dir
        plots_dir = tempfile.mkdtemp()

    # calculate nrows, ncols
    def nrows_ncols(num_subplots):
        nrows = ncols = math.ceil(math.sqrt(num_subplots))
        return nrows, ncols

    # plotting options
    pyplot.rc('font', size=6)
    # linestyles, designed so that simulated and expected curves which are equal will both be visible
    loosely_dashed = (0, (4, 6))
    dashdotted = (0, (2, 3, 3, 2))
    linewidth = 0.8
    simulated_plot_kwargs = dict(color='blue', label='simulated', linestyle=loosely_dashed,
                                 linewidth=linewidth)
    expected_plot_kwargs = dict(color='red', label='expected', linestyle=dashdotted,
                                linewidth=linewidth)

    # plot expected vs. simulated trajectories of species
    if trajectory_times and expected_species_trajectories:
        # calculate num subplots
        num_subplots = 0
        for expected_trajectory in expected_species_trajectories.values():
            # 1 subplot for each species
            if not np.isnan(expected_trajectory).all():
                num_subplots += 1
        if not num_subplots:
            print(f"no expected data for plot of {model_id} species trajectories using {integration_algorithm}")
        else:
            nrows, ncols = nrows_ncols(num_subplots)
            fig = pyplot.figure()
            fig.suptitle(f'Species copy numbers of model {model_id} using {integration_algorithm}')
            index = 0
            for species_id, expected_trajectory in expected_species_trajectories.items():
                if np.isnan(expected_trajectory).all():
                    continue
                # plot expected_trajectory vs. simulated_trajectory
                simulated_trajectory = populations_df[species_id]
                index += 1
                axes = fig.add_subplot(nrows, ncols, index)
                axes.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
                axes.plot(trajectory_times, simulated_trajectory, **simulated_plot_kwargs)
                axes.plot(trajectory_times, expected_trajectory, **expected_plot_kwargs)
                axes.set_xlabel('Time (s)')
                y_label = f'{species_id} (copy number)'
                axes.set_ylabel(y_label)
                axes.legend()
            figure_name = f'{model_id}_using_{integration_algorithm}_species_trajectories'
            filename = os.path.join(plots_dir, figure_name + '.pdf')
            fig.savefig(filename)
            pyplot.close(fig)

    # plot expected vs. simulated trajectories of properties of compartments
    units_for_properties = {
        'volume': 'l',
        'mass': 'g',
    }
    def get_units(property):
        rv = 'unknown'
        for prop, units in units_for_properties.items():
            if prop in property:
                rv = units
        return rv

    if trajectory_times and expected_property_trajectories:
        # get all properties
        properties = set()
        for property_array_dict in expected_property_trajectories.values():
            properties.update(property_array_dict.keys())

        if not len(properties):
            print(f"no properties for plot of {model_id} using {integration_algorithm} properties trajectories")
        else:
            nrows, ncols = nrows_ncols(len(properties))
            fig = pyplot.figure()
            fig.suptitle(f'Properties of model {model_id} using {integration_algorithm}')
            index = 0
            for property in properties:
                for compartment_id in expected_property_trajectories:
                    dynamic_compartment = dynamic_model.dynamic_compartments[compartment_id]
                    if compartment_id not in aggregate_states_df:
                        continue
                    # plot expected_trajectory vs. simulated_trajectory
                    simulated_trajectory = run_results.get_properties(compartment_id, property)
                    expected_trajectory = expected_property_trajectories[compartment_id][property]
                    index += 1
                    axes = fig.add_subplot(nrows, ncols, index)
                    axes.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
                    axes.plot(trajectory_times, simulated_trajectory, **simulated_plot_kwargs)
                    axes.plot(trajectory_times, expected_trajectory, **expected_plot_kwargs)
                    axes.set_xlabel('Time (s)')
                    y_label = f"{property} - {compartment_id} ({get_units(property)})"
                    axes.set_ylabel(y_label)
                    axes.legend()
                figure_name = f'{model_id}_using_{integration_algorithm}_properties'
                filename = os.path.join(plots_dir, figure_name + '.pdf')
                fig.savefig(filename)
                pyplot.close(fig)
    return plots_dir
