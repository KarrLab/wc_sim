""" Ensure that wc-lang-encoded models are validate so that these examples don't diverge from wc-lang.

:Author: Jonathan Karr <karr@mssm.edu>
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2018-11-15
:Copyright: 2018, Karr Lab
:License: MIT
"""

import copy
import unittest
import wc_lang.io


class ValidateWcLangFilesTestCase(unittest.TestCase):
    FILES = (
        {'path': 'examples/transcription_translation_hybrid_model/model.xlsx'},
        {'path': 'examples/translation_metabolism_hybrid_model/model.xlsx'},
        {'path': 'tests/fixtures/2_species_1_reaction.xlsx'},
        {'path': 'tests/fixtures/2_species_1_reaction_with_rates_given_by_reactant_population.xlsx'},
        {'path': 'tests/fixtures/2_species_a_pair_of_symmetrical_reactions_rates_given_by_reactant_population.xlsx'},
        {'path': 'tests/fixtures/MetabolismAndGeneExpression.xlsx'},
        {'path': 'tests/fixtures/test_dry_model.xlsx'},
        {'path': 'tests/fixtures/test_dry_model_with_mass_computation.xlsx',
         'ignore_extra_models': True},
        {'path': 'tests/fixtures/test_dynamic_expressions.xlsx'},
        {'path': 'tests/fixtures/test_model.xlsx',
         'ignore_extra_models': True},
        {'path': 'tests/fixtures/test_model_for_access_species_populations.xlsx'},
        {'path': 'tests/fixtures/test_model_for_access_species_populations_steady_state.xlsx'},
        {'path': 'tests/fixtures/test_new_features_model.xlsx'},
        {'path': 'tests/fixtures/dynamic_tests/one_exchange_rxn_compt_growth.xlsx',
         'ignore_extra_models': True},
        {'path': 'tests/fixtures/dynamic_tests/stop_conditions.xlsx',
         'ignore_extra_models': True},
        {'path': 'tests/fixtures/dynamic_tests/one_reaction_linear.xlsx',
         'ignore_extra_models': True},
        {'path': 'tests/fixtures/dynamic_tests/template.xlsx',
         'ignore_extra_models': True},
        {'path': 'tests/fixtures/dynamic_tests/one_rxn_exponential.xlsx',
         'ignore_extra_models': True},
        {'path': 'tests/fixtures/dynamic_tests/static.xlsx',
         'ignore_extra_models': True},
        {'path': 'tests/submodels/fixtures/test_submodel.xlsx'},
        {'path': 'tests/submodels/fixtures/test_submodel_no_shared_species.xlsx'},
    )

    def test(self):
        errs = []
        for file in self.FILES:
            kwargs = copy.copy(file)
            kwargs.pop('path')
            try:
                # reader already does validation
                wc_lang.io.Reader().run(file['path'], validate=True, **kwargs)
            except ValueError as err:
                errs.append("File: {}\n{}".format(file['path'], str(err)))
        if errs:
            raise Exception('The following example model(s) are invalid:\n{}'.format('\n'.join(errs)))
