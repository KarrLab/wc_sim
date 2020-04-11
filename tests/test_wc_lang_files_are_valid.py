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
        {'path': 'tests/submodels/fixtures/test_next_reaction_method_submodel.xlsx'},
        {'path': 'tests/submodels/fixtures/test_submodel.xlsx'},
        {'path': 'tests/submodels/fixtures/test_submodel_no_shared_species.xlsx'},
        {'path': 'tests/fixtures/verification/cases/multialgorithmic/00001/00001-wc_lang.xlsx'},
        {'path': 'tests/fixtures/verification/cases/multialgorithmic/00003/00003-wc_lang_1_submodel.xlsx'},
        {'path': 'tests/fixtures/verification/cases/multialgorithmic/00003/00003-wc_lang_2_submodels.xlsx'},
        {'path': 'tests/fixtures/verification/cases/multialgorithmic/00003/00003-wc_lang.xlsx'},
        {'path': 'tests/fixtures/verification/cases/multialgorithmic/00007/00007-wc_lang.xlsx'},
        {'path': 'tests/fixtures/verification/cases/multialgorithmic/00020/00020-wc_lang.xlsx'},
        {'path': 'tests/fixtures/verification/cases/multialgorithmic/00021/00021-wc_lang.xlsx'},
        {'path': 'tests/fixtures/verification/cases/multialgorithmic/00030/00007-wc_lang.xlsx'},
        {'path': 'tests/fixtures/verification/cases/multialgorithmic/00030/00030-wc_lang.xlsx'},
        {'path': 'tests/fixtures/verification/cases/semantic/00001/00001-wc_lang.xlsx'},
        {'path': 'tests/fixtures/verification/cases/semantic/00002/00002-wc_lang.xlsx'},
        {'path': 'tests/fixtures/verification/cases/semantic/00003/00003-wc_lang.xlsx'},
        {'path': 'tests/fixtures/verification/cases/semantic/00004/00004-wc_lang.xlsx'},
        {'path': 'tests/fixtures/verification/cases/semantic/00005/00005-wc_lang.xlsx'},
        {'path': 'tests/fixtures/verification/cases/semantic/00006/00006-wc_lang.xlsx'},
        {'path': 'tests/fixtures/verification/cases/semantic/00010/00010-wc_lang.xlsx'},
        {'path': 'tests/fixtures/verification/cases/semantic/00014/00014-wc_lang.xlsx'},
        {'path': 'tests/fixtures/verification/cases/semantic/00015/00015-wc_lang.xlsx'},
        {'path': 'tests/fixtures/verification/cases/semantic/00017/00017-wc_lang.xlsx'},
        {'path': 'tests/fixtures/verification/cases/semantic/00018/00018-wc_lang.xlsx'},
        {'path': 'tests/fixtures/verification/cases/semantic/00019/00019-wc_lang.xlsx'},
        {'path': 'tests/fixtures/verification/cases/semantic/00020/00020-wc_lang.xlsx'},
        {'path': 'tests/fixtures/verification/cases/semantic/00021/00021-wc_lang.xlsx'},
        {'path': 'tests/fixtures/verification/cases/semantic/00022/00022-wc_lang.xlsx'},
        {'path': 'tests/fixtures/verification/cases/semantic/00028/00028-wc_lang.xlsx'},
        {'path': 'tests/fixtures/verification/cases/semantic/00054/00054-wc_lang.xlsx'},
        {'path': 'tests/fixtures/verification/cases/stochastic/00001/00001-wc_lang.xlsx'},
        {'path': 'tests/fixtures/verification/cases/stochastic/00003/00003-wc_lang_1_submodel.xlsx'},
        {'path': 'tests/fixtures/verification/cases/stochastic/00003/00003-wc_lang_2_submodels.xlsx'},
        {'path': 'tests/fixtures/verification/cases/stochastic/00003/00003-wc_lang.xlsx'},
        {'path': 'tests/fixtures/verification/cases/stochastic/00004/00004-wc_lang.xlsx'},
        {'path': 'tests/fixtures/verification/cases/stochastic/00007/00007-wc_lang.xlsx'},
        {'path': 'tests/fixtures/verification/cases/stochastic/00007_hybrid/00007_hybrid-wc_lang_old.xlsx',
        'validate': False},
        {'path': 'tests/fixtures/verification/cases/stochastic/00007_hybrid/00007_hybrid-wc_lang.xlsx'},
        {'path': 'tests/fixtures/verification/cases/stochastic/00012/00012-wc_lang.xlsx'},
        {'path': 'tests/fixtures/verification/cases/stochastic/00020/00020-wc_lang.xlsx'},
        {'path': 'tests/fixtures/verification/cases/stochastic/00021/00021-wc_lang.xlsx'},
        {'path': 'tests/fixtures/verification/cases/stochastic/00030/00030-wc_lang.xlsx'},
        {'path': 'tests/fixtures/verification/cases/stochastic/00037/00037-wc_lang.xlsx'},
        {'path': 'tests/fixtures/verification/cases/stochastic/transcription_translation/transcription_translation-wc_lang.xlsx',
        'validate': False},
        {'path': 'tests/fixtures/verification/testing/hybrid/transcription_translation/transcription_translation_correct_ssa.xlsx',
        'validate': False},
        {'path': 'tests/fixtures/verification/testing/hybrid/transcription_translation/transcription_translation_hybrid.xlsx',
        'validate': False},
        {'path': 'tests/fixtures/verification/testing/hybrid/transcription_translation/transcription_translation-wc_lang_JK.xlsx',
        'validate': False},
        {'path': 'tests/fixtures/verification/testing/hybrid/translation_metabolism/translation_metabolism_correct_ssa.xlsx',
        'validate': False},
        {'path': 'tests/fixtures/verification/testing/hybrid/translation_metabolism/translation_metabolism_hybrid.xlsx',
        'validate': False},
        {'path': 'tests/fixtures/verification/testing/multialgorithmic/00007/00007-wc_lang.xlsx'},
        {'path': 'tests/fixtures/verification/testing/semantic/00001/00001-wc_lang.xlsx'},
        {'path': 'tests/fixtures/verification/testing/semantic/00004/00004-wc_lang.xlsx',
        'validate': False},
        {'path': 'tests/fixtures/verification/testing/semantic/00054/00054-wc_lang.xlsx'},
        {'path': 'tests/fixtures/verification/testing/stochastic/00001/00001-wc_lang.xlsx'},
        {'path': 'tests/fixtures/verification/testing/stochastic/00006/00006-wc_lang.xlsx'},
        {'path': 'tests/fixtures/verification/testing_ValidationSuite_run/stochastic/00001/00001-wc_lang.xlsx',
        'validate': False},
        {'path': 'tests/fixtures/verification/testing_ValidationSuite_run/stochastic/00006/00006-wc_lang.xlsx',
        'validate': False},
    )

    def test(self):
        errs = []
        for file in self.FILES:
            if file.get('validate', True):
                kwargs = copy.copy(file)
                kwargs.pop('path')
                if 'validate' in kwargs:
                    kwargs.pop('validate')
                try:
                    # reader already does validation
                    wc_lang.io.Reader().run(file['path'], validate=True, **kwargs)
                except ValueError as err:
                    errs.append("File: {}\n{}".format(file['path'], str(err)))
        if errs:
            raise Exception('The following example model(s) are invalid:\n{}'.format('\n'.join(errs)))
