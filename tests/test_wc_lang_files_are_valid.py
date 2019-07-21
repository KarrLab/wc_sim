""" Ensure that wc-lang-encoded models are validate so that these examples don't diverge from wc-lang.

:Author: Jonathan Karr <karr@mssm.edu>
:Date: 2018-11-15
:Copyright: 2018, Karr Lab
:License: MIT
"""

import unittest
import wc_lang.io


class ValidateWcLangFilesTestCase(unittest.TestCase):
    FILES = (
        'tests/multialgorithm/submodels/fixtures/test_submodel.xlsx',
        'tests/multialgorithm/submodels/fixtures/test_submodel_no_shared_species.xlsx',
        'tests/multialgorithm/fixtures/2_species_1_reaction_with_rates_given_by_reactant_population.xlsx',
        'tests/multialgorithm/fixtures/test_new_features_model.xlsx',
        'tests/multialgorithm/fixtures/test_dry_model.xlsx',
        #'tests/multialgorithm/fixtures/test_model_with_mass_computation.xlsx', # contains formulae in extra columns/rows
        'tests/multialgorithm/fixtures/2_species_a_pair_of_symmetrical_reactions_rates_given_by_reactant_population.xlsx',
        'tests/multialgorithm/fixtures/test_model.xlsx',
        #'tests/multialgorithm/fixtures/test_dry_model_with_mass_computation.xlsx', # contains formulae in extra columns/rows
        'tests/multialgorithm/fixtures/2_species_1_reaction.xlsx',
        'tests/multialgorithm/fixtures/test_model_for_access_species_populations.xlsx',
        'tests/multialgorithm/fixtures/test_model_for_access_species_populations_steady_state.xlsx',
        'examples/transcription_translation_hybrid_model/model.xlsx',
        'examples/translation_metabolism_hybrid_model/model.xlsx',
    )

    def test(self):
        errs = []
        for file in self.FILES:
            try:
                # reader already does validation
                wc_lang.io.Reader().run(file, validate=True)
            except ValueError as err:
                errs.append("File: {}\n{}".format(file, str(err)))
        if errs:
            raise Exception('The following example model(s) are invalid:\n{}'.format('\n'.join(errs)))
