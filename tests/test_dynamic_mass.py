""" Tests of constant and dynamically changing mass

:Author: Jonathan Karr <karr@mssm.edu>
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2019-01-17
:Copyright: 2019, Karr Lab
:License: MIT
"""

from matplotlib import pyplot
from scipy.constants import Avogadro as NA
from test.support import EnvironmentVarGuard
import numpy
import numpy.testing
import os
import shutil
import tempfile
import unittest

from wc_onto import onto
from wc_sim.run_results import RunResults
from wc_sim.simulation import Simulation
from wc_utils.util.units import unit_registry
import wc_lang
import wc_lang.io
import wc_lang.util


class TwoSpeciesTestCase(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        self.model = self.gen_model()
        self.st_constant = self.model.species_types.get_one(id='st_constant')
        self.st_dynamic = self.model.species_types.get_one(id='st_dynamic')
        self.comp = self.model.compartments.get_one(id='comp')
        self.volume = self.model.functions.get_one(id='volume')
        self.density = self.model.parameters.get_one(id='density')
        self.spec_constant = self.model.species.get_one(species_type=self.st_constant, compartment=self.comp)
        self.spec_dynamic = self.model.species.get_one(species_type=self.st_dynamic, compartment=self.comp)
        # species populations, because the units for these concentrations are molecules
        self.spec_constant_pop = self.spec_constant.distribution_init_concentration
        self.spec_dynamic_pop = self.spec_dynamic.distribution_init_concentration
        self.k_syn_dynamic = self.model.parameters.get_one(id='k_syn_dynamic')
        self.k_deg_dynamic = self.model.parameters.get_one(id='k_deg_dynamic')

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def gen_model(self):
        """ Generate this model:

        * One compartment, comp
        * Two submodels, submodel_synthesis and submodel_degradation
        * Two species types
        * Two species

            * One whose copy number increases at a constant rate, st_constant[comp]
            * One whose copy number changes dynamically, st_dynamic[comp]

        * Four reactions which produce and consume the dynamically changing species

            * In submodel_synthesis

                * rxn_synthesis_constant: -> st_constant[comp]; k_syn_constant
                * rxn_synthesis_dynamic: -> st_dynamic[comp]; k_syn_dynamic

            * In submodel_degradation

                * rxn_degradation_constant: st_constant[comp] -> ; k_deg_constant * st_constant[comp]
                * rxn_degradation_dynamic: st_dynamic[comp] -> ; k_deg_dynamic * st_dynamic[comp]

        * Default values

            * k_syn_constant = 0
            * k_deg_constant = 0
        """
        model = wc_lang.Model(id='model', version='0.0.1', wc_lang_version='0.0.1')

        # species, compartment and initial concentrations
        st_constant = model.species_types.create(id='st_constant',
                                                 structure=wc_lang.ChemicalStructure(charge=0., molecular_weight=1.))
        st_dynamic = model.species_types.create(id='st_dynamic',
                                                structure=wc_lang.ChemicalStructure(charge=0., molecular_weight=0.))
        comp = model.compartments.create(id='comp', init_volume=wc_lang.InitVolume(std=0.))
        spec_constant = model.species.create(species_type=st_constant, compartment=comp)
        spec_dynamic = model.species.create(species_type=st_dynamic, compartment=comp)
        spec_constant.id = spec_constant.gen_id()
        spec_dynamic.id = spec_dynamic.gen_id()
        conc_constant = model.distribution_init_concentrations.create(
                                                                      species=spec_constant, std=0.,
                                                                      units=unit_registry.parse_units('molecule'))
        conc_dynamic = model.distribution_init_concentrations.create(
                                                                     species=spec_dynamic, std=0.,
                                                                     units=unit_registry.parse_units('molecule'))
        conc_constant.id = conc_constant.gen_id()
        conc_dynamic.id = conc_dynamic.gen_id()

        # density and volume
        density = comp.init_density = model.parameters.create(id='density', value=1.,
                                                              units=unit_registry.parse_units('g l^-1'))
        volume = model.functions.create(id='volume', units=unit_registry.parse_units('l'))
        volume.expression, error = wc_lang.FunctionExpression.deserialize(f'{comp.id} / {density.id}', {
                                                                          wc_lang.Compartment: {comp.id: comp},
                                                                          wc_lang.Parameter: {density.id: density}})
        assert error is None, str(error)

        # submodels
        submdl_syn = model.submodels.create(id='submodel_synthesis',
                                            framework=onto['WC:stochastic_simulation_algorithm'])
        submdl_deg = model.submodels.create(id='submodel_degradation',
                                            framework=onto['WC:stochastic_simulation_algorithm'])

        # reactions
        rxn_syn_constant = model.reactions.create(id='rxn_synthesis_constant', submodel=submdl_syn)
        rxn_syn_constant.participants.create(species=spec_constant, coefficient=1)
        rl_syn_constant = model.rate_laws.create(reaction=rxn_syn_constant)
        rl_syn_constant.id = rl_syn_constant.gen_id()
        k_syn_constant = model.parameters.create(id='k_syn_constant', value=0., units=unit_registry.parse_units('s^-1'))
        rl_syn_constant.expression, error = wc_lang.RateLawExpression.deserialize(k_syn_constant.id, {
            wc_lang.Parameter: {
                k_syn_constant.id: k_syn_constant,
            },
        })
        assert error is None, str(error)

        rxn_deg_constant = model.reactions.create(id='rxn_degradation_constant', submodel=submdl_deg)
        rxn_deg_constant.participants.create(species=spec_constant, coefficient=-1)
        rl_deg_constant = model.rate_laws.create(reaction=rxn_deg_constant)
        rl_deg_constant.id = rl_deg_constant.gen_id()
        k_deg_constant = model.parameters.create(id='k_deg_constant', value=0.,
                                                 units=unit_registry.parse_units('s^-1 molecule^-1'))
        rl_deg_constant.expression, error = \
            wc_lang.RateLawExpression.deserialize(f'{k_deg_constant.id} * {spec_constant.id}', {
            wc_lang.Parameter: {
                k_deg_constant.id: k_deg_constant,
            },
            wc_lang.Species: {
                spec_constant.id: spec_constant,
            }
        })
        assert error is None, str(error)

        rxn_syn_dynamic = model.reactions.create(id='rxn_synthesis_dynamic', submodel=submdl_syn)
        rxn_syn_dynamic.participants.create(species=spec_dynamic, coefficient=1)
        rl_syn_dynamic = model.rate_laws.create(reaction=rxn_syn_dynamic)
        rl_syn_dynamic.id = rl_syn_dynamic.gen_id()
        k_syn_dynamic = model.parameters.create(id='k_syn_dynamic', units=unit_registry.parse_units('s^-1'))
        rl_syn_dynamic.expression, error = wc_lang.RateLawExpression.deserialize(k_syn_dynamic.id, {
            wc_lang.Parameter: {
                k_syn_dynamic.id: k_syn_dynamic,
            },
        })
        assert error is None, str(error)

        rxn_deg_dynamic = model.reactions.create(id='rxn_degradation_dynamic', submodel=submdl_deg)
        rxn_deg_dynamic.participants.create(species=spec_dynamic, coefficient=-1)
        rl_deg_dynamic = model.rate_laws.create(reaction=rxn_deg_dynamic)
        rl_deg_dynamic.id = rl_deg_dynamic.gen_id()
        k_deg_dynamic = model.parameters.create(id='k_deg_dynamic',
                                                units=unit_registry.parse_units('s^-1 molecule^-1'))
        rl_deg_dynamic.expression, error = \
            wc_lang.RateLawExpression.deserialize(f'{k_deg_dynamic.id} * {spec_dynamic.id}', {
                wc_lang.Parameter: {
                    k_deg_dynamic.id: k_deg_dynamic,
                },
                wc_lang.Species: {
                    spec_dynamic.id: spec_dynamic,
                }
            })
        assert error is None, str(error)

        # other parameter
        model.parameters.create(id='Avogadro', value=NA,
                                units=unit_registry.parse_units('molecule mol^-1'))

        return model

    def simulate(self, model, end_time=100.):
        # simulate
        env = EnvironmentVarGuard()
        env.set('CONFIG__DOT__wc_lang__DOT__validation__DOT__validate_element_charge_balance', '0')
        with env:
            simulation = Simulation(model)
        _, results_dirname = simulation.run(end_time=end_time,
                                            ode_time_step=1.,
                                            checkpoint_period=1.,
                                            results_dir=self.tempdir)

        # get results
        results = RunResults(results_dirname)

        agg_states = results.get('aggregate_states')
        comp_mass = agg_states[(self.comp.id, 'mass')]
        comp_vol = results.get('functions')[self.volume.id]

        pops = results.get('populations')
        time = pops.index
        pop_constant = pops[self.spec_constant.id]
        pop_dynamic = pops[self.spec_dynamic.id]

        return (time, pop_constant, pop_dynamic, comp_mass, comp_vol)

    def test_exponentially_increase_to_steady_state_count_with_constant_mass(self):
        # set quantitative values
        self.comp.init_volume.mean = 100e-15
        self.st_constant.structure.molecular_weight = 1.
        self.st_dynamic.structure.molecular_weight = 0.
        self.spec_constant_pop.mean = 1e6
        self.spec_dynamic_pop.mean = 10.
        init_mass = (self.spec_constant_pop.mean * self.st_constant.structure.molecular_weight +
                     self.spec_dynamic_pop.mean * self.st_dynamic.structure.molecular_weight
                     ) / NA
        init_density = self.density.value = init_mass / self.comp.init_volume.mean
        self.k_syn_dynamic.value = 1.
        self.k_deg_dynamic.value = 2e-2

        # simulate
        time, pop_constant, pop_dynamic, comp_mass, comp_vol = self.simulate(self.model, end_time=200.)

        # verify results
        numpy.testing.assert_equal(pop_constant, numpy.full((201,),
                                   self.spec_constant_pop.mean))
        numpy.testing.assert_equal(comp_mass,
                                   numpy.full((201,),
                                              self.spec_constant_pop.mean *
                                              self.st_constant.structure.molecular_weight / NA))
        self.assertEqual(self.density.value, init_density)

        spec_dynamic_ss = self.k_syn_dynamic.value / self.k_deg_dynamic.value
        std_spec_dynamic_ss = numpy.sqrt(spec_dynamic_ss)
        self.assertGreater(numpy.mean(pop_dynamic[101:]), spec_dynamic_ss - 3 * std_spec_dynamic_ss)
        self.assertLess(numpy.mean(pop_dynamic[101:]), spec_dynamic_ss + 3 * std_spec_dynamic_ss)

    def test_exponentially_increase_to_steady_state_count_and_mass(self):
        # set quantitative values
        self.comp.init_volume.mean = 100e-15
        self.st_constant.structure.molecular_weight = 0.
        self.st_dynamic.structure.molecular_weight = 1.
        self.spec_constant_pop.mean = 0.
        self.spec_dynamic_pop.mean = 10.
        init_mass = (self.spec_constant_pop.mean * self.st_constant.structure.molecular_weight +
                     self.spec_dynamic_pop.mean * self.st_dynamic.structure.molecular_weight
                     ) / NA
        init_density = self.density.value = init_mass / self.comp.init_volume.mean
        self.k_syn_dynamic.value = 5.
        self.k_deg_dynamic.value = 5e-2

        # simulate
        time, pop_constant, pop_dynamic, comp_mass, comp_vol = self.simulate(self.model, end_time=200.)

        # verify results
        numpy.testing.assert_equal(pop_constant, numpy.full((201,),
                                   self.spec_constant_pop.mean))
        numpy.testing.assert_equal(comp_mass[0], init_mass)
        self.assertEqual(self.density.value, init_density)

        spec_dynamic_ss = self.k_syn_dynamic.value / self.k_deg_dynamic.value
        std_spec_dynamic_ss = numpy.sqrt(spec_dynamic_ss)
        self.assertGreater(numpy.mean(pop_dynamic[101:]), spec_dynamic_ss - 3 * std_spec_dynamic_ss)
        self.assertLess(numpy.mean(pop_dynamic[101:]), spec_dynamic_ss + 3 * std_spec_dynamic_ss)

        mass_ss = spec_dynamic_ss * self.st_dynamic.structure.molecular_weight / NA
        std_mass_ss = std_spec_dynamic_ss * self.st_dynamic.structure.molecular_weight / NA
        self.assertGreater(numpy.mean(comp_mass[101:]), mass_ss - 3 * std_mass_ss)
        self.assertLess(numpy.mean(comp_mass[101:]), mass_ss + 3 * std_mass_ss)

    def test_exponentially_descend_to_steady_state_count_and_mass(self):
        """
        Reactions:

        * In submodel_synthesis:

            * rxn_synthesis_constant: -> st_constant[comp]; 0
            * rxn_synthesis_dynamic: -> st_dynamic[comp]; 1

        * In submodel_degradation:

            * rxn_degradation_constant: st_constant[comp] -> ; 0
            * rxn_degradation_dynamic: st_dynamic[comp] -> ; 0.05 * st_dynamic[comp]
        """
        # set quantitative values
        self.comp.init_volume.mean = 100e-15
        self.st_constant.structure.molecular_weight = 0.
        self.st_dynamic.structure.molecular_weight = 1.
        self.spec_constant_pop.mean = 0.
        self.spec_dynamic_pop.mean = 100.
        init_mass = (self.spec_constant_pop.mean * self.st_constant.structure.molecular_weight +
                     self.spec_dynamic_pop.mean * self.st_dynamic.structure.molecular_weight) / NA
        init_density = self.density.value = init_mass / self.comp.init_volume.mean
        self.k_syn_dynamic.value = 1.
        self.k_deg_dynamic.value = 5e-2

        # simulate
        time, pop_constant, pop_dynamic, comp_mass, comp_vol = self.simulate(self.model, end_time=200.)

        # verify results
        numpy.testing.assert_equal(pop_constant,
                                   numpy.full((201,), self.spec_constant_pop.mean))
        numpy.testing.assert_equal(comp_mass[0], init_mass)
        self.assertEqual(self.density.value, init_density)

        spec_dynamic_ss = self.k_syn_dynamic.value / self.k_deg_dynamic.value
        std_spec_dynamic_ss = numpy.sqrt(spec_dynamic_ss)
        self.assertGreater(numpy.mean(pop_dynamic[101:]), spec_dynamic_ss - 3 * std_spec_dynamic_ss)
        self.assertLess(numpy.mean(pop_dynamic[101:]), spec_dynamic_ss + 3 * std_spec_dynamic_ss)

        mass_ss = spec_dynamic_ss * self.st_dynamic.structure.molecular_weight / NA
        std_mass_ss = std_spec_dynamic_ss * self.st_dynamic.structure.molecular_weight / NA
        self.assertGreater(numpy.mean(comp_mass[101:]), mass_ss - 3 * std_mass_ss)
        self.assertLess(numpy.mean(comp_mass[101:]), mass_ss + 3 * std_mass_ss)

    def test_exponentially_increase_to_steady_state_mass_and_descend_to_concentration(self):
        """
        Reactions, including new rate law below:

        * In submodel_synthesis:

            * rxn_synthesis_constant: -> st_constant[comp]; k_syn_constant
            * rxn_synthesis_dynamic: -> st_dynamic[comp]; k_syn_dynamic * volume * NA

        * In submodel_degradation:

            * rxn_degradation_constant: st_constant[comp] -> ; k_deg_constant * st_constant[comp]
            * rxn_degradation_dynamic: st_dynamic[comp] -> ; k_deg_dynamic * st_dynamic[comp]
        """
        # change rate law
        rxn_syn_dynamic = self.model.reactions.get_one(id='rxn_synthesis_dynamic')
        rl_syn_dynamic = rxn_syn_dynamic.rate_laws[0]
        self.k_syn_dynamic.units = unit_registry.parse_units('mol s^-1 l^-1 molecule^-1')
        Avogadro = self.model.parameters.get_one(id='Avogadro')
        rl_syn_dynamic.expression.parameters = []
        rl_syn_dynamic.expression, error = \
            wc_lang.RateLawExpression.deserialize(f'{self.k_syn_dynamic.id} * {self.volume.id} * {Avogadro.id}', {
                wc_lang.Parameter: {
                    self.k_syn_dynamic.id: self.k_syn_dynamic,
                    Avogadro.id: Avogadro,
                },
                wc_lang.Function: {
                    self.volume.id: self.volume,
                },
            })
        assert error is None, str(error)

        # set quantitative values
        self.comp.init_volume.mean = 100e-15
        self.st_constant.structure.molecular_weight = 1.
        self.st_dynamic.structure.molecular_weight = 0.
        self.spec_constant_pop.mean = 10.
        self.spec_dynamic_pop.mean = 100.
        init_mass = (self.spec_constant_pop.mean * self.st_constant.structure.molecular_weight +
                     self.spec_dynamic_pop.mean * self.st_dynamic.structure.molecular_weight) / NA
        init_density = self.density.value = init_mass / self.comp.init_volume.mean
        k_syn_constant = self.model.parameters.get_one(id='k_syn_constant')
        k_deg_constant = self.model.parameters.get_one(id='k_deg_constant')
        k_syn_constant.value = 1.
        k_deg_constant.value = 2e-2

        # analyze steady-state (ss)
        # mass:
        # in ss, spec_constant concentration in molecules is given by:
        # k_syn_constant.value / k_deg_constant.value = 50
        spec_constant_ss_pop = k_syn_constant.value / k_deg_constant.value
        std_spec_constant_ss_pop = numpy.sqrt(spec_constant_ss_pop)
        # mass of spec_constant is given by:
        # 50 * self.st_constant.structure.molecular_weight / NA
        mass_spec_constant_ss = spec_constant_ss_pop * self.st_constant.structure.molecular_weight / NA
        std_mass_spec_constant_ss = std_spec_constant_ss_pop * self.st_constant.structure.molecular_weight / NA
        # entire ss mass equals mass of spec_constant because st_dynamic.structure.molecular_weight = 0:
        mass_ss = mass_spec_constant_ss
        std_mass_ss = std_mass_spec_constant_ss

        vol_ss = mass_ss / self.density.value
        std_vol_ss = std_mass_ss / self.density.value

        self.k_syn_dynamic.value = 1. / (NA * vol_ss)
        self.k_deg_dynamic.value = 5e-2
        k_syn_dynamic = self.k_syn_dynamic
        k_deg_dynamic = self.k_deg_dynamic

        # in ss, spec_dynamic population (molecule) is:
        spec_dynamic_ss_pop = self.k_syn_dynamic.value * vol_ss * NA / self.k_deg_dynamic.value
        std_spec_dynamic_ss_pop = numpy.sqrt(spec_dynamic_ss_pop)

        init_vol = self.comp.init_volume.mean

        spec_dynamic_ss_conc = spec_dynamic_ss_pop / (vol_ss * NA)
        std_spec_dynamic_ss_conc = std_spec_dynamic_ss_pop / (vol_ss * NA)

        # simulate
        end_time = 1000.
        time, pop_constant, pop_dynamic, comp_mass, comp_vol = self.simulate(self.model, end_time=end_time)
        conc_dynamic = pop_dynamic / comp_vol / NA

        # verify results
        numpy.testing.assert_equal(comp_mass[0], init_mass)
        self.assertEqual(self.density.value, init_density)
        # confirm that density remains constant
        comp_density = comp_mass / comp_vol
        numpy.testing.assert_allclose(list(comp_density.to_numpy()), [self.density.value]*int(end_time + 1))

        fig, axes = pyplot.subplots(nrows=2, ncols=3)

        axes[0][0].plot(time, comp_mass)
        axes[0][0].set_xlabel('Time (s)')
        axes[0][0].set_ylabel('Mass (g)')

        axes[0][1].plot(time, comp_vol)
        axes[0][1].set_xlabel('Time (s)')
        axes[0][1].set_ylabel('Volume (l)')

        axes[0][2].plot(time, comp_mass/comp_vol)
        axes[0][2].set_xlabel('Time (s)')
        axes[0][2].set_ylabel('Density (g/l)')

        axes[1][0].plot(time, pop_dynamic)
        axes[1][0].set_xlabel('Time (s)')
        axes[1][0].set_ylabel('Dynamic copy number')

        axes[1][1].plot(time, comp_vol/self.comp.init_volume.mean)
        axes[1][1].set_xlabel('Time (s)')
        axes[1][1].set_ylabel('Fold change volume')

        axes[1][2].plot(time, comp_mass/init_mass)
        axes[1][2].set_xlabel('Time (s)')
        axes[1][2].set_ylabel('Fold change mass')

        dirname = os.path.join(os.path.dirname(__file__), 'results')
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        filename = os.path.join(dirname,
                                os.path.basename(__file__).replace(
                                '.py', 'exponentially_increase_to_steady_state_mass_and_concentration.pdf'))
        fig.savefig(filename)
        pyplot.close(fig)

        mid_point = (end_time / 2) + 1
        self.assertGreater(numpy.mean(comp_mass[mid_point:]), mass_ss - 3 * std_mass_ss)
        self.assertLess(numpy.mean(comp_mass[mid_point:]), mass_ss + 3 * std_mass_ss)

        self.assertGreater(numpy.mean(comp_vol[mid_point:]), vol_ss - 3 * std_vol_ss)
        self.assertLess(numpy.mean(comp_vol[mid_point:]), vol_ss + 3 * std_vol_ss)

        self.assertGreater(numpy.mean(conc_dynamic[mid_point:]),
                           spec_dynamic_ss_conc - 3 * std_spec_dynamic_ss_conc)
        self.assertLess(numpy.mean(conc_dynamic[mid_point:]),
                        spec_dynamic_ss_conc + 3 * std_spec_dynamic_ss_conc)

        third_quarter = int((3 * end_time / 4) + 1)
        self.assertGreater(numpy.mean(pop_dynamic[third_quarter:]),
                           spec_dynamic_ss_pop - 3 * std_spec_dynamic_ss_pop)
        self.assertLess(numpy.mean(pop_dynamic[third_quarter:]),
                        spec_dynamic_ss_pop + 3 * std_spec_dynamic_ss_pop)


class MetabolismAndGeneExpressionTestCase(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test(self):
        model_filename = os.path.join(os.path.dirname(__file__), 'fixtures', 'MetabolismAndGeneExpression.xlsx')
        model = wc_lang.io.Reader().run(model_filename)[wc_lang.Model][0]

        simulation = Simulation(model)
        _, results_dirname = simulation.run(end_time=8 * 3600,
                                            ode_time_step=1.,
                                            checkpoint_period=100.,
                                            results_dir=self.tempdir)

        # get results
        results = RunResults(results_dirname)

        agg_states = results.get('aggregate_states')
        mass = agg_states[('c', 'mass')]

        pops = results.get('populations')
        time = pops.index
        funcs = results.get('functions')
        volume_c = funcs['volume_c']
        volume_e = funcs['volume_e']

        # assert
        self.assertGreater(mass.values[-1] / mass.values[0], 1.75)
        self.assertLess(mass.values[-1] / mass.values[0], 2.25)

        ala_c_fold_change = pops['ala[c]'].values[-1] / pops['ala[c]'].values[0]
        amp_c_fold_change = pops['amp[c]'].values[-1] / pops['amp[c]'].values[0]
        ala_e_fold_change = pops['ala[e]'].values[-1] / pops['ala[e]'].values[0]
        amp_e_fold_change = pops['amp[e]'].values[-1] / pops['amp[e]'].values[0]

        self.assertGreater(ala_c_fold_change, 1.75)
        self.assertGreater(amp_c_fold_change, 1.75)
        self.assertLess(ala_c_fold_change, 2.25)
        self.assertLess(amp_c_fold_change, 2.25)

        self.assertGreater(ala_e_fold_change, 0.95)
        self.assertGreater(amp_e_fold_change, 0.95)
        self.assertLess(ala_e_fold_change, 1.0)
        self.assertLess(amp_e_fold_change, 1.0)

        prot_fold_change = (pops['protein_rnapol[c]'].values[-1]
                            + pops['protein_ribosome[c]'].values[-1]
                            + pops['protein_rnase[c]'].values[-1]
                            + pops['protein_protease[c]'].values[-1]) \
            / (pops['protein_rnapol[c]'].values[0]
               + pops['protein_ribosome[c]'].values[0]
               + pops['protein_rnase[c]'].values[0]
               + pops['protein_protease[c]'].values[0])
        self.assertGreater(prot_fold_change, 1.5)
        self.assertLess(prot_fold_change, 4.)

        # plot
        dirname = os.path.join(os.path.dirname(__file__), 'results')
        if not os.path.isdir(dirname):
            os.mkdir(dirname)

        fig, axes = pyplot.subplots(nrows=2, ncols=3)

        # mass
        axes[0][0].plot(time / 3600, 1e18 * mass)
        axes[0][0].set_ylabel('Mass (fg)')

        # volume
        axes[1][0].plot(time / 3600, 1e15 * volume_c)
        axes[1][0].set_xlabel('Time (h)')
        axes[1][0].set_ylabel('Volume (al)')

        # metabolites
        axes[0][1].plot(time / 3600, 1e3 * numpy.stack((
                        (pops['ala[e]'] / volume_e / NA).values,
                        (pops['amp[e]'] / volume_e / NA).values)).T)
        axes[0][1].set_title('Extracellular metabolites')
        axes[0][1].set_ylabel('mM')

        axes[1][1].plot(time / 3600, 1e3 * numpy.stack((
                        (pops['ala[c]'] / volume_c / NA).values,
                        (pops['amp[c]'] / volume_c / NA).values)).T)
        axes[1][1].set_title('Cytosol metabolites')
        axes[1][1].set_ylabel('mM')

        # RNA
        axes[0][2].plot(time / 3600, numpy.stack((
            pops['rna_rnapol[c]'].values,
            pops['rna_ribosome[c]'].values,
            pops['rna_rnase[c]'].values,
            pops['rna_protease[c]'].values)).T)
        axes[0][2].set_title('RNA')
        axes[0][2].set_ylabel('Count (molecule)')

        # protein
        axes[1][2].plot(time / 3600, numpy.stack((
            pops['protein_rnapol[c]'].values,
            pops['protein_ribosome[c]'].values,
            pops['protein_rnase[c]'].values,
            pops['protein_protease[c]'].values)).T)
        axes[1][2].set_title('Protein')
        axes[1][2].set_ylabel('Count (molecule)')

        # save figure
        filename = os.path.join(dirname, 'MetabolismAndGeneExpression.pdf')
        fig.savefig(filename)
        pyplot.close(fig)
