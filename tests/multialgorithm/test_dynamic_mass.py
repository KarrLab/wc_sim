""" Tests of constant and dynamically changing mass

* (One compartment)
* (Two species types)
* Two species
    
    * One whose copy number changes dynamically
    * One whose copy number doesn't change (isn't involved in any reactions)

* (One submodel)
* Two reactions which produce and consume the dynamically changing species

:Author: Jonathan Karr <karr@mssm.edu>
:Date: 2019-01-17
:Copyright: 2019, Karr Lab
:License: MIT
"""

from test.support import EnvironmentVarGuard
from wc_sim.multialgorithm.run_results import RunResults
from wc_sim.multialgorithm.simulation import Simulation
from wc_utils.util.ontology import wcm_ontology
import numpy
import numpy.testing
import scipy.constants
import shutil
import tempfile
import unittest
import wc_lang
import wc_lang.util
import wc_sim


class DynamicMassTestCase(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def gen_model(self):
        # tests constant mass (and constant volume)
        model = model = wc_lang.Model(id='model', version='0.0.1', wc_lang_version='0.0.1')

        # compartments and species
        st_constant = model.species_types.create(id='st_constant', charge=0., molecular_weight=1.)
        st_dynamic = model.species_types.create(id='st_dynamic', charge=0., molecular_weight=0.)
        comp = model.compartments.create(id='comp', std_init_volume=0.)
        spec_constant = model.species.create(species_type=st_constant, compartment=comp)
        spec_dynamic = model.species.create(species_type=st_dynamic, compartment=comp)
        spec_constant.id = spec_constant.gen_id()
        spec_dynamic.id = spec_dynamic.gen_id()
        conc_constant = model.distribution_init_concentrations.create(
            species=spec_constant, std=0., units=wc_lang.ConcentrationUnit.molecule)
        conc_dynamic = model.distribution_init_concentrations.create(
            species=spec_dynamic, std=0., units=wc_lang.ConcentrationUnit.molecule)
        conc_constant.id = conc_constant.gen_id()
        conc_dynamic.id = conc_dynamic.gen_id()

        density = comp.init_density = model.parameters.create(id='density', value=1., units='g l^-1')
        volume = model.functions.create(id='volume', units='l')
        volume.expression, error = wc_lang.FunctionExpression.deserialize(f'{comp.id} / {density.id}', {
            wc_lang.Compartment: {comp.id: comp},
            wc_lang.Parameter: {density.id: density}})
        assert error is None, str(error)

        # submodels
        submdl_syn = model.submodels.create(id='submodel_synthesis', framework=wcm_ontology['WCM:stochastic_simulation_algorithm'])
        submdl_deg = model.submodels.create(id='submodel_degradation', framework=wcm_ontology['WCM:stochastic_simulation_algorithm'])

        # reactions
        rxn_syn = model.reactions.create(id='rxn_synthesis', submodel=submdl_syn)
        rxn_syn.participants.create(species=spec_dynamic, coefficient=1)
        rl_syn = rxn_syn.rate_laws.create()
        rl_syn.id = rl_syn.gen_id()
        k_syn = model.parameters.create(id='k_syn', units='s^-1')
        rl_syn.expression, error = wc_lang.RateLawExpression.deserialize(k_syn.id, {
            wc_lang.Parameter: {
                k_syn.id: k_syn,
            },
        })
        assert error is None, str(error)

        rxn_deg = model.reactions.create(id='rxn_degradation', submodel=submdl_deg)
        rxn_deg.participants.create(species=spec_dynamic, coefficient=-1)
        rl_deg = rxn_deg.rate_laws.create()
        rl_deg.id = rl_deg.gen_id()
        k_deg = model.parameters.create(id='k_deg', units='s^-1 molecule^-1')
        rl_deg.expression, error = wc_lang.RateLawExpression.deserialize(f'{k_deg.id} * {spec_dynamic.id}', {
            wc_lang.Parameter: {
                k_deg.id: k_deg,
            },
            wc_lang.Species: {
                spec_dynamic.id: spec_dynamic,
            }
        })
        assert error is None, str(error)

        # return model
        return model

    def simulate(self, model, end_time=100.):
        st_constant = model.species_types.get_one(id='st_constant')
        st_dynamic = model.species_types.get_one(id='st_dynamic')
        comp = model.compartments.get_one(id='comp')
        spec_constant = model.species.get_one(species_type=st_constant, compartment=comp)
        spec_dynamic = model.species.get_one(species_type=st_dynamic, compartment=comp)

        # simulate
        env = EnvironmentVarGuard()
        env.set('CONFIG__DOT__wc_lang__DOT__validation__DOT__validate_element_charge_balance', '0')
        with env:
            simulation = Simulation(model)
        _, results_dirname = simulation.run(end_time=end_time,
                                            time_step=1.,
                                            checkpoint_period=1.,
                                            results_dir=self.tempdir)

        # get results
        results = RunResults(results_dirname)

        agg_states = results.get('aggregate_states')
        comp_mass = agg_states[(comp.id, 'mass')]

        pops = results.get('populations')
        time = pops.index
        pop_constant = pops[spec_constant.id]
        pop_dynamic = pops[spec_dynamic.id]

        return (time, pop_constant, pop_dynamic, comp_mass)

    def test_constant_mass(self):
        model = self.gen_model()
        st_constant = model.species_types.get_one(id='st_constant')
        st_dynamic = model.species_types.get_one(id='st_dynamic')
        comp = model.compartments.get_one(id='comp')
        density = model.parameters.get_one(id='density')
        spec_constant = model.species.get_one(species_type=st_constant, compartment=comp)
        spec_dynamic = model.species.get_one(species_type=st_dynamic, compartment=comp)

        # set quantitative values
        comp.mean_init_volume = 100e-15
        st_constant.molecular_weight = 1.
        st_dynamic.molecular_weight = 0.
        spec_constant.distribution_init_concentration.mean = 1e6
        spec_dynamic.distribution_init_concentration.mean = 10.
        init_mass = (spec_constant.distribution_init_concentration.mean * st_constant.molecular_weight +
                     spec_dynamic.distribution_init_concentration.mean * st_dynamic.molecular_weight
                     ) / scipy.constants.Avogadro
        init_density = density.value = init_mass / comp.mean_init_volume
        model.parameters.get_one(id='k_syn').value = 1.
        model.parameters.get_one(id='k_deg').value = 1.

        # simulate
        time, pop_constant, pop_dynamic, comp_mass = self.simulate(model)

        # verify results
        numpy.testing.assert_equal(pop_constant, numpy.full((101,), spec_constant.distribution_init_concentration.mean))
        numpy.testing.assert_equal(comp_mass, numpy.full((101,), spec_constant.distribution_init_concentration.mean
                                                         * st_constant.molecular_weight
                                                         / scipy.constants.Avogadro))
        self.assertEqual(density.value, init_density)

    def test_exponentially_increase_to_steady_state_mass(self):
        model = self.gen_model()
        st_constant = model.species_types.get_one(id='st_constant')
        st_dynamic = model.species_types.get_one(id='st_dynamic')
        comp = model.compartments.get_one(id='comp')
        density = model.parameters.get_one(id='density')
        spec_constant = model.species.get_one(species_type=st_constant, compartment=comp)
        spec_dynamic = model.species.get_one(species_type=st_dynamic, compartment=comp)

        # set quantitative values
        comp.mean_init_volume = 100e-15
        st_constant.molecular_weight = 0.
        st_dynamic.molecular_weight = 1.
        spec_constant.distribution_init_concentration.mean = 0.
        spec_dynamic.distribution_init_concentration.mean = 10.
        init_mass = (spec_constant.distribution_init_concentration.mean * st_constant.molecular_weight +
                     spec_dynamic.distribution_init_concentration.mean * st_dynamic.molecular_weight
                     ) / scipy.constants.Avogadro
        init_density = density.value = init_mass / comp.mean_init_volume
        model.parameters.get_one(id='k_syn').value = 5.
        model.parameters.get_one(id='k_deg').value = 5e-2

        # simulate
        time, pop_constant, pop_dynamic, comp_mass = self.simulate(model, end_time=200.)

        # verify results
        numpy.testing.assert_equal(pop_constant, numpy.full((201,), spec_constant.distribution_init_concentration.mean))
        numpy.testing.assert_equal(comp_mass[0], init_mass)
        self.assertEqual(density.value, init_density)

        spec_dynamic_ss = model.parameters.get_one(id='k_syn').value / model.parameters.get_one(id='k_deg').value
        std_spec_dynamic_ss = numpy.sqrt(spec_dynamic_ss)
        self.assertGreater(numpy.mean(pop_dynamic[101:]), spec_dynamic_ss - 3 * std_spec_dynamic_ss)
        self.assertLess(numpy.mean(pop_dynamic[101:]), spec_dynamic_ss + 3 * std_spec_dynamic_ss)

        mass_ss = spec_dynamic_ss * st_dynamic.molecular_weight / scipy.constants.Avogadro
        std_mass_ss = std_spec_dynamic_ss * st_dynamic.molecular_weight / scipy.constants.Avogadro
        self.assertGreater(numpy.mean(comp_mass[101:]), mass_ss - 3 * std_mass_ss)
        self.assertLess(numpy.mean(comp_mass[101:]), mass_ss + 3 * std_mass_ss)

    def test_exponentially_descend_to_steady_state_mass(self):
        model = self.gen_model()
        st_constant = model.species_types.get_one(id='st_constant')
        st_dynamic = model.species_types.get_one(id='st_dynamic')
        comp = model.compartments.get_one(id='comp')
        density = model.parameters.get_one(id='density')
        spec_constant = model.species.get_one(species_type=st_constant, compartment=comp)
        spec_dynamic = model.species.get_one(species_type=st_dynamic, compartment=comp)

        # set quantitative values
        comp.mean_init_volume = 100e-15
        st_constant.molecular_weight = 0.
        st_dynamic.molecular_weight = 1.
        spec_constant.distribution_init_concentration.mean = 0.
        spec_dynamic.distribution_init_concentration.mean = 100.
        init_mass = (spec_constant.distribution_init_concentration.mean * st_constant.molecular_weight +
                     spec_dynamic.distribution_init_concentration.mean * st_dynamic.molecular_weight
                     ) / scipy.constants.Avogadro
        init_density = density.value = init_mass / comp.mean_init_volume
        model.parameters.get_one(id='k_syn').value = 1.
        model.parameters.get_one(id='k_deg').value = 5e-2

        # simulate
        time, pop_constant, pop_dynamic, comp_mass = self.simulate(model, end_time=200.)

        # verify results
        numpy.testing.assert_equal(pop_constant, numpy.full((201,), spec_constant.distribution_init_concentration.mean))
        numpy.testing.assert_equal(comp_mass[0], init_mass)
        self.assertEqual(density.value, init_density)

        spec_dynamic_ss = model.parameters.get_one(id='k_syn').value / model.parameters.get_one(id='k_deg').value
        std_spec_dynamic_ss = numpy.sqrt(spec_dynamic_ss)
        self.assertGreater(numpy.mean(pop_dynamic[101:]), spec_dynamic_ss - 3 * std_spec_dynamic_ss)
        self.assertLess(numpy.mean(pop_dynamic[101:]), spec_dynamic_ss + 3 * std_spec_dynamic_ss)

        mass_ss = spec_dynamic_ss * st_dynamic.molecular_weight / scipy.constants.Avogadro
        std_mass_ss = std_spec_dynamic_ss * st_dynamic.molecular_weight / scipy.constants.Avogadro
        self.assertGreater(numpy.mean(comp_mass[101:]), mass_ss - 3 * std_mass_ss)
        self.assertLess(numpy.mean(comp_mass[101:]), mass_ss + 3 * std_mass_ss)
