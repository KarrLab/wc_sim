""" Ensure that wc-lang-encoded models are validate so that
these examples don't diverge from wc-lang.

:Author: Jonathan Karr <jonrkarr@gmail.com>
:Date: 2018-11-15
:Copyright: 2018, Karr Lab
:License: MIT
"""

import unittest
import wc_lang.io


class ValidateWcLangFilesTestCase(unittest.TestCase):
    FILES = (
        {'path': 'tests/multialgorithm/submodels/fixtures/test_submodel.xlsx',
         'strict': True},
        {'path': 'tests/multialgorithm/submodels/fixtures/test_submodel_no_shared_species.xlsx',
         'strict': True},
        {'path': 'tests/multialgorithm/fixtures/2_species_1_reaction_with_rates_given_by_reactant_population.xlsx',
         'strict': True},
        {'path': 'tests/multialgorithm/fixtures/test_new_features_model.xlsx',
         'strict': True},
        {'path': 'tests/multialgorithm/fixtures/test_dry_model.xlsx',
         'strict': True},
        #{'path': 'tests/multialgorithm/fixtures/test_model_with_mass_computation.xlsx',
        # 'strict': False}, # contains formulae in extra columns/rows
        {'path': 'tests/multialgorithm/fixtures/2_species_a_pair_of_symmetrical_reactions_rates_given_by_reactant_population.xlsx',
         'strict': True},
        {'path': 'tests/multialgorithm/fixtures/test_model.xlsx',
         'strict': True},
        #{'path': 'tests/multialgorithm/fixtures/test_dry_model_with_mass_computation.xlsx',
        # 'strict': False}, # contains formulae in extra columns/rows
        {'path': 'tests/multialgorithm/fixtures/2_species_1_reaction.xlsx',
         'strict': True},
        {'path': 'tests/multialgorithm/fixtures/test_model_for_access_species_populations.xlsx',
         'strict': True},
        {'path': 'tests/multialgorithm/fixtures/test_model_for_access_species_populations_steady_state.xlsx',
         'strict': True},
        {'path': 'examples/transcription_translation_hybrid_model/model.xlsx',
         'strict': True},
        {'path': 'examples/translation_metabolism_hybrid_model/model.xlsx',
         'strict': True},
    )

    def test(self):
        errs = []
        for file in self.FILES:
            try:
                # reader already does validation
                wc_lang.io.Reader().run(file['path'], strict=file['strict'])
            except ValueError as err:
                errs.append(str(err))
        if errs:
            raise Exception('The following examples are invalid:\n  {}'.format('\n  '.join(errs)))
